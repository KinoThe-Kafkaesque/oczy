# S1.4 HF Layer-L Hidden Probe Results

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Layers**: 24 decoder layers
- **Embedding dim**: 896
- **Corpus hash**: `e5b4ef66beb2`
- **Primary pooling**: `mean` (mean-pool over content tokens)
- **Mid-layer range**: 6-17 (25–75% depth)
- **Timestamp**: 2026-07-02T00:34:39.236374+00:00

| Layer | mean **← primary** | last | max |
|-------|-------|-------|-------|
| L0 | 0.4198 | 0.0080 | 0.0890 |
| L1 | 0.3012 | 0.0117 | 0.0739 |
| L2 | 0.0788 | 0.0230 | 0.0000 |
| L3 | 0.0789 | 0.0273 | 0.0000 |
| L4 | 0.0779 | 0.0606 | 0.0000 |
| L5 | 0.0952 | 0.0718 | 0.0000 |
| L6 | 0.0922 | 0.0761 | 0.0000 |
| L7 | 0.0857 | 0.0856 | 0.0000 |
| L8 | 0.0874 | 0.1039 | 0.0000 |
| L9 | 0.0829 | 0.1013 | 0.0000 |
| L10 | 0.0857 | 0.1214 | 0.0000 |
| L11 | 0.0876 | 0.1162 | 0.0000 |
| L12 | 0.0815 | 0.1071 | 0.0000 |
| L13 | 0.0829 | 0.1126 | 0.0000 |
| L14 | 0.0860 | 0.1351 | 0.0000 |
| L15 | 0.0812 | 0.1344 | 0.0000 |
| L16 | 0.0843 | 0.1288 | 0.0000 |
| L17 | 0.0792 | 0.1171 | 0.0000 |
| L18 | 0.0760 | 0.1241 | 0.0001 |
| L19 | 0.0719 | 0.1311 | 0.0001 |
| L20 | 0.0708 | 0.1272 | 0.0001 |
| L21 | 0.1219 | 0.1022 | 0.0647 |
| L22 | 0.1193 | 0.1074 | 0.0694 |
| L23 | 0.1753 | 0.1421 | 0.0909 |

## Verdict

- **Final-layer (mean) silhouette**: 0.1753
- **Max mid-layer (mean) silhouette**: 0.0922
- **Gap**: -0.0831 (threshold: +0.10)
- **Verdict**: **REFUTE**

H-L refuted: mid-layer hiddens do NOT cluster by concept better than the final layer (gap < +0.10). This confirms lane_03's refutation on a substrate that can see every layer.

---
*Pre-registered spec: `research/10-hf-layer-l-hidden-probe.md`*

---

## Secondary #3: LFM2.5-1.2B-Instruct

### S1.4 HF Layer-L Hidden Probe Results

- **Model**: `LiquidAI/LFM2.5-1.2B-Instruct`
- **Layers**: 16 decoder layers
- **Embedding dim**: 2048
- **Corpus hash**: `e5b4ef66beb2`
- **Primary pooling**: `mean` (mean-pool over content tokens)
- **Mid-layer range**: 4-11 (25–75% depth)
- **Timestamp**: 2026-07-02T00:34:43.309519+00:00

| Layer | mean **← primary** | last | max |
|-------|-------|-------|-------|
| L0 | 0.2385 | 0.0090 | 0.0671 |
| L1 | 0.1656 | 0.0048 | 0.0053 |
| L2 | 0.3524 | 0.0926 | 0.0062 |
| L3 | 0.3785 | 0.1175 | 0.0054 |
| L4 | 0.1649 | 0.0712 | 0.0051 |
| L5 | 0.3358 | 0.1706 | 0.0071 |
| L6 | 0.3158 | 0.1701 | 0.0154 |
| L7 | 0.1364 | 0.1298 | 0.0002 |
| L8 | 0.2552 | 0.1945 | 0.0003 |
| L9 | 0.0752 | 0.1773 | 0.0006 |
| L10 | 0.1458 | 0.2391 | 0.0009 |
| L11 | 0.1185 | 0.2212 | 0.0019 |
| L12 | 0.2145 | 0.2930 | 0.0050 |
| L13 | 0.2362 | 0.2682 | 0.0128 |
| L14 | 0.2096 | 0.2780 | 0.0221 |
| L15 | 0.2781 | 0.3274 | 0.0548 |

#### Verdict

- **Final-layer (mean) silhouette**: 0.2781
- **Max mid-layer (mean) silhouette**: 0.3358
- **Gap**: +0.0576 (threshold: +0.10)
- **Verdict**: **REFUTE**

H-L refuted: mid-layer hiddens do NOT cluster by concept better than the final layer (gap < +0.10). This confirms lane_03's refutation on a substrate that can see every layer.

---
*Pre-registered spec: `research/10-hf-layer-l-hidden-probe.md`*