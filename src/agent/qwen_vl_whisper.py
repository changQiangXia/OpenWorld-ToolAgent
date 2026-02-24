from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _dedup(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _extract_json_dict(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    if not raw:
        return {}

    starts = [i for i, ch in enumerate(raw) if ch == "{"]
    for start in starts:
        depth = 0
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : idx + 1]
                    try:
                        obj = json.loads(candidate)
                    except Exception:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return {}


def _normalize_tool_name(raw_tool: str, valid_tools: Sequence[str]) -> Optional[str]:
    token = str(raw_tool or "").strip().lower()
    if not token:
        return None
    token = token.strip("`\"' ")
    token = token.replace(" ", "_")
    token = token.replace("-", "_")

    normalized_to_canonical: Dict[str, str] = {}
    for tool in valid_tools:
        low = tool.lower()
        normalized_to_canonical[low] = tool
        normalized_to_canonical[low.replace("-", "_")] = tool

    if token in normalized_to_canonical:
        return normalized_to_canonical[token]

    # Accept partial matches only as fallback.
    for norm, canonical in normalized_to_canonical.items():
        if token and (token in norm or norm in token):
            return canonical
    return None


def parse_tool_decision(
    raw_text: str,
    candidate_tools: Sequence[str],
    unknown_token: str,
) -> Dict[str, Any]:
    valid_tools = _dedup([unknown_token] + [str(x) for x in candidate_tools if str(x) and str(x) != unknown_token])
    payload = _extract_json_dict(raw_text)
    text_lc = str(raw_text or "").lower()

    pred_tool: Optional[str] = None
    unknown_prob = None
    confidence = None
    is_unknown_json: Optional[bool] = None

    if payload:
        for key in ["pred_tool", "selected_tool", "tool", "answer"]:
            value = payload.get(key)
            if isinstance(value, str):
                pred_tool = _normalize_tool_name(value, valid_tools)
                if pred_tool:
                    break

        for key in ["unknown_prob", "unknown_probability", "unknown_score"]:
            unknown_prob = _safe_float(payload.get(key))
            if unknown_prob is not None:
                break

        for key in ["confidence", "tool_confidence", "score"]:
            confidence = _safe_float(payload.get(key))
            if confidence is not None:
                break

        if isinstance(payload.get("is_unknown"), bool):
            is_unknown_json = bool(payload["is_unknown"])

    if pred_tool is None:
        for tool in valid_tools:
            if tool == unknown_token:
                continue
            if tool.lower() in text_lc:
                pred_tool = tool
                break

    if pred_tool is None and (unknown_token.lower() in text_lc or "unknown" in text_lc):
        pred_tool = unknown_token

    if pred_tool is None:
        pred_tool = unknown_token

    if unknown_prob is None:
        unknown_prob = 0.88 if pred_tool == unknown_token else 0.12

    if is_unknown_json is True:
        unknown_prob = max(unknown_prob, 0.8)
    elif is_unknown_json is False:
        unknown_prob = min(unknown_prob, 0.4)

    unknown_prob = _clamp01(float(unknown_prob))

    if confidence is None:
        confidence = unknown_prob if pred_tool == unknown_token else (1.0 - unknown_prob)
    confidence = _clamp01(float(confidence))

    return {
        "pred_tool": pred_tool,
        "unknown_prob": float(unknown_prob),
        "confidence": float(confidence),
    }


@dataclass(frozen=True)
class ToolSelectionResult:
    pred_tool: str
    pred_tools: List[str]
    pred_tool_scores: List[float]
    unknown_prob: float
    confidence: float
    is_unknown_pred: bool
    raw_response: str
    prompt_text: str
    asr_text: str
    latency_ms: float


class QwenVLWhisperToolSelector:
    def __init__(
        self,
        project_root: Path,
        qwen_model_dir: Path,
        whisper_model_dir: Path,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        media_mode: str = "multimodal",
        video_frame_strategy: str = "first_frame",
        trust_remote_code: bool = True,
        use_flash_attention_2: bool = True,
        load_in_4bit: bool = False,
        max_new_tokens: int = 96,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        asr_chunk_length_s: float = 30.0,
        asr_batch_size: int = 8,
        asr_language: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.project_root = project_root
        self.qwen_model_dir = qwen_model_dir
        self.whisper_model_dir = whisper_model_dir
        self.device = str(device)
        self.dtype = str(dtype)
        self.media_mode = str(media_mode)
        self.video_frame_strategy = str(video_frame_strategy)
        self.trust_remote_code = bool(trust_remote_code)
        self.use_flash_attention_2 = bool(use_flash_attention_2)
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.asr_chunk_length_s = float(asr_chunk_length_s)
        self.asr_batch_size = int(asr_batch_size)
        self.asr_language = asr_language
        self.ffmpeg_bin = ffmpeg_bin

        self._qwen_model: Any = None
        self._qwen_processor: Any = None
        self._asr_pipeline: Any = None

    def _resolve_media_path(self, media_path: str) -> Path:
        p = Path(str(media_path))
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def _resolve_torch_dtype(self) -> Any:
        import torch

        name = self.dtype.lower()
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16", "half"}:
            return torch.float16
        return torch.float32

    def _infer_device_index(self) -> int:
        if self.device.startswith("cuda"):
            if ":" in self.device:
                try:
                    return int(self.device.split(":", 1)[1])
                except Exception:
                    return 0
            return 0
        return -1

    def _select_qwen_model_cls(self) -> Any:
        import transformers

        for cls_name in [
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]:
            if hasattr(transformers, cls_name):
                return getattr(transformers, cls_name)
        raise RuntimeError("No compatible Qwen model class found in transformers.")

    def _ensure_qwen_loaded(self) -> None:
        if self._qwen_model is not None and self._qwen_processor is not None:
            return

        import torch
        from transformers import AutoProcessor

        model_cls = self._select_qwen_model_cls()

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self._resolve_torch_dtype(),
        }
        if self.device.startswith("cuda"):
            # Force single-device placement based on runtime `device`.
            # In this project, `device_map="auto"` caused unstable gibberish outputs on multi-GPU.
            model_kwargs["device_map"] = {"": max(self._infer_device_index(), 0)}
        if self.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self._qwen_model = model_cls.from_pretrained(str(self.qwen_model_dir), **model_kwargs)
        self._qwen_processor = AutoProcessor.from_pretrained(
            str(self.qwen_model_dir),
            trust_remote_code=self.trust_remote_code,
        )

        if not self.device.startswith("cuda"):
            self._qwen_model.to(torch.device("cpu"))

    def _ensure_asr_loaded(self) -> None:
        if self._asr_pipeline is not None:
            return

        from transformers import pipeline

        device_index = self._infer_device_index()
        kwargs: Dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": str(self.whisper_model_dir),
            "device": device_index,
        }
        if device_index >= 0:
            kwargs["torch_dtype"] = self._resolve_torch_dtype()
        self._asr_pipeline = pipeline(**kwargs)

    def transcribe_audio(self, audio_path: Path) -> str:
        if not audio_path.exists():
            return ""

        self._ensure_asr_loaded()
        generate_kwargs: Dict[str, Any] = {}
        if self.asr_language:
            generate_kwargs["language"] = self.asr_language

        call_kwargs: Dict[str, Any] = {
            "chunk_length_s": self.asr_chunk_length_s,
            "batch_size": self.asr_batch_size,
            "return_timestamps": False,
        }
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs

        result = self._asr_pipeline(str(audio_path), **call_kwargs)
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

    def _load_image(self, image_path: Path) -> Any:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.convert("RGB").copy()

    def _extract_video_frame_path(self, video_path: Path) -> Optional[Path]:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                "select=eq(n\\,0)",
                "-vframes",
                "1",
                str(tmp_path),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode != 0 or (not tmp_path.exists()) or tmp_path.stat().st_size <= 0:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return None
            return tmp_path
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def _build_prompt(
        self,
        row: Dict[str, Any],
        candidate_tools: Sequence[str],
        unknown_token: str,
        asr_text: str,
    ) -> str:
        query = str(row.get("query", "")).strip()
        query_raw = str(row.get("query_raw", "")).strip()
        modality = str(row.get("modality", "unknown")).strip().lower()
        ambiguity_type = str(row.get("ambiguity_type", "unknown")).strip()
        tool_status = str(row.get("tool_status", "unknown")).strip()

        lines = [
            "You are an open-world tool router.",
            "Choose the best tool for the user request.",
            "Return strict JSON only:",
            '{"pred_tool":"<tool_or_unknown>","unknown_prob":0.0,"confidence":0.0}',
            "Rules:",
            f"1) pred_tool must be one of: {', '.join([unknown_token] + list(candidate_tools))}",
            "2) unknown_prob and confidence must be in [0,1]",
            "3) Prefer selecting one candidate tool when any candidate is plausible",
            "4) Use unknown token only when no candidate tool can solve the request",
            f"unknown_token: {unknown_token}",
            f"modality: {modality}",
            f"ambiguity_type: {ambiguity_type}",
            f"tool_status: {tool_status}",
            f"query: {query}",
        ]
        if query_raw:
            lines.append(f"query_raw: {query_raw[:400]}")
        if asr_text:
            lines.append(f"audio_transcript: {asr_text[:800]}")
        return "\n".join(lines)

    def _model_device(self) -> Any:
        import torch

        if self._qwen_model is None:
            return torch.device("cpu")
        if hasattr(self._qwen_model, "device"):
            return self._qwen_model.device
        try:
            return next(self._qwen_model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _as_file_uri(self, path: Path) -> str:
        return path.resolve().as_uri()

    def _generation_kwargs(self) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        return gen_kwargs

    def _decode_generated(self, model_inputs: Dict[str, Any], output_ids: Any) -> str:
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            return ""
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)]
        if not hasattr(self._qwen_processor, "batch_decode"):
            return ""
        decoded = self._qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded and str(decoded[0]).strip():
            return str(decoded[0]).strip()
        decoded_full = self._qwen_processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded_full[0]).strip() if decoded_full else ""

    def _generate_text_with_qwen_utils(
        self,
        prompt: str,
        vision_content: Optional[Dict[str, Any]],
    ) -> str:
        from qwen_vl_utils import process_vision_info

        user_content: List[Dict[str, Any]] = []
        if vision_content is not None:
            user_content.append(vision_content)
        user_content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": "You are a strict JSON-only assistant for tool routing."},
            {"role": "user", "content": user_content},
        ]
        chat_text = self._qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        processor_kwargs: Dict[str, Any] = {
            "text": [chat_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs
        model_inputs = self._qwen_processor(**processor_kwargs)
        device = self._model_device()
        for key, value in list(model_inputs.items()):
            if hasattr(value, "to"):
                model_inputs[key] = value.to(device)
        output_ids = self._qwen_model.generate(**model_inputs, **self._generation_kwargs())
        return self._decode_generated(model_inputs=model_inputs, output_ids=output_ids)

    def _generate_text_legacy(self, prompt: str, image: Optional[Any]) -> str:
        # Compatibility fallback when qwen_vl_utils is unavailable.
        try:
            messages = [
                {"role": "system", "content": "You are a strict JSON-only assistant for tool routing."},
                {
                    "role": "user",
                    "content": ([{"type": "image", "image": image}] if image is not None else [])
                    + [{"type": "text", "text": prompt}],
                },
            ]
            chat_text = self._qwen_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if image is not None:
                model_inputs = self._qwen_processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
            else:
                model_inputs = self._qwen_processor(text=[chat_text], return_tensors="pt", padding=True)
        except Exception:
            chat_text = prompt
            model_inputs = self._qwen_processor(text=[chat_text], return_tensors="pt", padding=True)
        device = self._model_device()
        for key, value in list(model_inputs.items()):
            if hasattr(value, "to"):
                model_inputs[key] = value.to(device)
        output_ids = self._qwen_model.generate(**model_inputs, **self._generation_kwargs())
        return self._decode_generated(model_inputs=model_inputs, output_ids=output_ids)

    def _generate_text(
        self,
        prompt: str,
        image: Optional[Any],
        vision_content: Optional[Dict[str, Any]],
    ) -> str:
        self._ensure_qwen_loaded()
        errors: List[str] = []
        try:
            decoded = self._generate_text_with_qwen_utils(prompt=prompt, vision_content=vision_content)
            if decoded:
                return decoded
            errors.append("qwen_vl_utils returned empty response")
        except Exception as exc:
            errors.append(f"qwen_vl_utils path failed: {exc}")
        try:
            decoded = self._generate_text_legacy(prompt=prompt, image=image)
            if decoded:
                return decoded
            errors.append("legacy path returned empty response")
        except Exception as exc:
            errors.append(f"legacy path failed: {exc}")
        raise RuntimeError("; ".join(errors))

    def _choose_candidate_fallback(
        self,
        row: Dict[str, Any],
        candidate_tools: Sequence[str],
        unknown_token: str,
        asr_text: str,
    ) -> str:
        filtered = [str(x) for x in candidate_tools if str(x) and str(x) != unknown_token]
        if not filtered:
            return unknown_token

        query = str(row.get("query", "")).strip().lower()
        query_raw = str(row.get("query_raw", "")).strip().lower()
        text_blob = f"{query} {query_raw} {asr_text}".lower()
        for tool in filtered:
            if tool.lower() in text_blob:
                return tool
        # Retriever-ordered top candidate is the most stable fallback.
        return filtered[0]

    def predict(
        self,
        row: Dict[str, Any],
        candidate_tools: Sequence[str],
        unknown_token: str,
        unknown_threshold: float,
    ) -> ToolSelectionResult:
        t0 = time.perf_counter()

        modality = str(row.get("modality", "unknown")).strip().lower()
        media_path = str(row.get("media_path", "")).strip()
        media_abs = self._resolve_media_path(media_path) if media_path else None

        image = None
        vision_content: Optional[Dict[str, Any]] = None
        cleanup_paths: List[Path] = []
        asr_text = ""

        if media_abs is not None and media_abs.exists():
            if modality == "audio":
                asr_text = self.transcribe_audio(media_abs)
            elif self.media_mode == "multimodal" and modality == "image":
                try:
                    image = self._load_image(media_abs)
                except Exception:
                    image = None
                vision_content = {"type": "image", "image": self._as_file_uri(media_abs)}
            elif self.media_mode == "multimodal" and modality == "video":
                if self.video_frame_strategy == "first_frame":
                    frame_path = self._extract_video_frame_path(media_abs)
                    if frame_path is not None and frame_path.exists():
                        cleanup_paths.append(frame_path)
                        vision_content = {"type": "image", "image": self._as_file_uri(frame_path)}
                        try:
                            image = self._load_image(frame_path)
                        except Exception:
                            image = None
                else:
                    vision_content = {"type": "video", "video": self._as_file_uri(media_abs)}

        prompt = self._build_prompt(
            row=row,
            candidate_tools=candidate_tools,
            unknown_token=unknown_token,
            asr_text=asr_text,
        )

        try:
            raw_response = self._generate_text(prompt=prompt, image=image, vision_content=vision_content)
        except Exception as exc:
            raw_response = json.dumps(
                {
                    "pred_tool": unknown_token,
                    "unknown_prob": 0.99,
                    "confidence": 0.60,
                    "error": str(exc)[:240],
                },
                ensure_ascii=True,
            )
        finally:
            for p in cleanup_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        parsed = parse_tool_decision(raw_response, candidate_tools=candidate_tools, unknown_token=unknown_token)

        unknown_prob = _clamp01(float(parsed["unknown_prob"]))
        confidence = _clamp01(float(parsed["confidence"]))
        base_pred_tool = str(parsed["pred_tool"])
        # Some runs return degenerate unknown outputs (e.g. unknown_prob~0.9 with confidence=0)
        # even when candidate tools are available. Fall back to retriever top candidate in this case.
        should_force_fallback = (
            base_pred_tool == unknown_token
            and (unknown_prob < float(unknown_threshold) or confidence <= 0.05)
        )
        if should_force_fallback:
            base_pred_tool = self._choose_candidate_fallback(
                row=row,
                candidate_tools=candidate_tools,
                unknown_token=unknown_token,
                asr_text=asr_text,
            )
            if base_pred_tool != unknown_token:
                unknown_prob = min(unknown_prob, max(0.0, float(unknown_threshold) - 0.01))
                confidence = max(confidence, 1.0 - unknown_prob)
        is_unknown_pred = (unknown_prob >= float(unknown_threshold)) or (base_pred_tool == unknown_token)
        pred_tool = unknown_token if is_unknown_pred else base_pred_tool

        ranked = [pred_tool] + [str(x) for x in candidate_tools if str(x) and str(x) != pred_tool]
        pred_tools = _dedup(ranked)
        if not pred_tools:
            pred_tools = [pred_tool]

        base_score = confidence if pred_tool != unknown_token else max(confidence, unknown_prob)
        pred_tool_scores: List[float] = []
        for idx in range(len(pred_tools)):
            pred_tool_scores.append(_clamp01(base_score - 0.08 * idx))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ToolSelectionResult(
            pred_tool=pred_tool,
            pred_tools=pred_tools,
            pred_tool_scores=pred_tool_scores,
            unknown_prob=unknown_prob,
            confidence=confidence,
            is_unknown_pred=is_unknown_pred,
            raw_response=raw_response,
            prompt_text=prompt,
            asr_text=asr_text,
            latency_ms=latency_ms,
        )
