from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.main_v1_data import (
    batch_encode_inputs,
    batch_targets,
    build_retriever,
    build_tool_vocab,
    load_rows,
)
from src.agent.main_v1_eval import evaluate_rows
from src.agent.main_v1_model import MainV1Model


class MainV1SmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unknown_token = "__unknown__"
        self.device = torch.device("cpu")

        train_path = PROJECT_ROOT / "data/splits/train.jsonl"
        dev_path = PROJECT_ROOT / "data/splits/dev.jsonl"
        self.train_rows = load_rows(train_path, limit=16)
        self.dev_rows = load_rows(dev_path, limit=12)

        self.tool_vocab = build_tool_vocab(self.train_rows, unknown_token=self.unknown_token)
        self.tool_to_idx = {t: i for i, t in enumerate(self.tool_vocab)}
        self.retriever = build_retriever(
            project_root=PROJECT_ROOT,
            tool_corpus_relpath="data/processed/tool_corpus.jsonl",
            tool_vocab=self.tool_vocab,
        )

        self.model = MainV1Model(
            num_tools=len(self.tool_vocab),
            text_dim=128,
            retrieval_dim=64,
            hidden_dim=96,
            modality_dim=16,
        ).to(self.device)

    def test_forward_shape(self) -> None:
        batch = self.train_rows[:8]
        qf, rf, mids, _ = batch_encode_inputs(
            rows=batch,
            text_dim=128,
            retrieval_dim=64,
            retriever=self.retriever,
            retriever_top_k=3,
            device=self.device,
        )
        out = self.model(query_features=qf, retrieval_features=rf, modality_ids=mids)

        self.assertEqual(tuple(out["tool_logits"].shape), (8, len(self.tool_vocab)))
        self.assertEqual(tuple(out["unknown_logit"].shape), (8, 1))

    def test_loss_is_finite(self) -> None:
        batch = self.train_rows[:8]
        qf, rf, mids, _ = batch_encode_inputs(
            rows=batch,
            text_dim=128,
            retrieval_dim=64,
            retriever=self.retriever,
            retriever_top_k=3,
            device=self.device,
        )
        tt, ut = batch_targets(
            rows=batch,
            tool_to_idx=self.tool_to_idx,
            unknown_token=self.unknown_token,
            device=self.device,
        )
        out = self.model(query_features=qf, retrieval_features=rf, modality_ids=mids)
        losses = self.model.compute_loss(outputs=out, tool_targets=tt, unknown_targets=ut)

        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertTrue(torch.isfinite(losses["tool_loss"]))
        self.assertTrue(torch.isfinite(losses["unknown_loss"]))

    def test_prediction_json_format(self) -> None:
        metrics, preds = evaluate_rows(
            model=self.model,
            rows=self.dev_rows,
            tool_vocab=self.tool_vocab,
            retriever=self.retriever,
            retriever_top_k=3,
            text_dim=128,
            retrieval_dim=64,
            unknown_token=self.unknown_token,
            unknown_threshold=0.55,
            top_k_tools=3,
            unknown_temperature=1.0,
            unknown_energy_weight=0.2,
            exp_id="test_exp",
            cost_per_request_usd=0.0,
            device=self.device,
        )
        self.assertEqual(len(preds), len(self.dev_rows))
        self.assertGreaterEqual(metrics["json_valid_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
