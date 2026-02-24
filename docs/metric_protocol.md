# Metric Protocol (Week1 Draft)

## Main Metrics
1. Tool Selection Accuracy
2. Hallucination Rate
3. Unknown Detection F1
4. Expected Calibration Error (ECE)
5. End-to-End Success Rate
6. Latency / Cost per request

## Required Slices
- Modality: text/image/audio/video
- Mapping: one-to-one / one-to-many
- Difficulty: >= 5 ambiguity types
- Tool status: stable/offline/replaced/new

## Statistics
- At least 3 seeds for key experiments
- Report mean + std (or variance)
- Mark statistical significance when feasible
