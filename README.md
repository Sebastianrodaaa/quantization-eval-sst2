# Quantization Tradeoffs for a Production Sentiment Classifier

Overnight eval: INT8 dynamic quantization tradeoffs for CPU-bound sentiment serving — method, results, error analysis.

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SST-2](https://img.shields.io/badge/task-SST--2%20sentiment-1f6feb)](https://huggingface.co/datasets/nyu-mll/glue)
[![INT8 dynamic](https://img.shields.io/badge/quantization-INT8%20dynamic-f0883e)](https://pytorch.org/docs/stable/quantization.html)
[![CPU serving](https://img.shields.io/badge/serving-CPU-6e7681)](#results)
[![all-MiniLM-L6-v2](https://img.shields.io/badge/model-all--MiniLM--L6--v2-8b949e)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Question:** when serving a sentiment classifier on CPU-only infrastructure, how much accuracy do we actually give up by quantizing to INT8 — and is the compression worth it?

This eval answers it empirically: a lexicon baseline, a pretrained sentence embedding model (MiniLM) with centroid classification, and its INT8-quantized version, all scored on the same held-out test set with latency and model size measured on the same machine. Full run: [`eval-quantization-tradeoffs.ipynb`](eval-quantization-tradeoffs.ipynb).

## Table of contents

- [Method](#method)
- [Results](#results)
- [Error analysis](#error-analysis)
- [Conclusions](#conclusions)
- [Limitations](#limitations)
- [Reproducing](#reproducing)

## Method

- **Dataset:** SST-2 validation split (500 examples; first 300 used as calibration train set, last 200 held out for eval)
- **Baseline:** hand-crafted positive/negative lexicon + threshold
- **Model:** `all-MiniLM-L6-v2` sentence embeddings + nearest-centroid classification (centroids fit on the calibration split only, never on eval data)
- **Quantization:** PyTorch dynamic INT8 quantization (qnnpack engine) of all `nn.Linear` layers (37 quantized linear layers)
- **Metrics:** accuracy, precision/recall/F1, p50 latency (batch=1), on-disk model size
- **Error analysis:** fp32-vs-INT8 disagreement analysis + qualitative review of regression cases

## Results

| Model | Accuracy | Precision | Recall | F1 | p50 latency (b=1) | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexicon baseline | 58.0% | 0.809 | 0.272 | 0.407 | ~0 ms | ~0 MB |
| MiniLM fp32 (centroid) | 78.5% | 0.819 | 0.782 | 0.800 | 4.84 ms | 90.9 MB |
| MiniLM INT8 (centroid) | 76.0% | 0.804 | 0.746 | 0.774 | 4.64 ms | 47.7 MB |

Exact latency varies run-to-run by a few percent. Accuracy and F1 are stable across reruns. The results table in the notebook rounds latency to ~4–5 ms; the millisecond figures above are the measured p50 from this run.

## Error analysis

INT8 changes the label on **13 of 200** held-out examples: **9 regressions** (fp32 correct, INT8 wrong) and **4 accidental fixes** (fp32 wrong, INT8 right). **39 errors are shared** — both precisions get them wrong.

Regressions cluster on subtly positive reviews, negation, and long comparative sentences, where the embedding margin to the nearest centroid is thin:

| True | fp32 | INT8 | Sentence (truncated) |
| ---: | ---: | ---: | --- |
| 1 | 1 | 0 | broomfield turns his distinctive ` blundering ' style into something that could really hel |
| 1 | 1 | 0 | whether writer-director anne fontaine 's film is a ghost story , an account of a nervous b |
| 1 | 1 | 0 | a good piece of work more often than not . |
| 0 | 0 | 1 | with its dogged hollywood naturalism and the inexorable passage of its characters toward s |
| 0 | 0 | 1 | sticky sweet sentimentality , clumsy plotting and a rosily myopic view of life in the wwii |

Shared failures are the centroid head missing SST-2 subtlety, not the bit width. Examples both models miss:

| True | Both | Sentence (truncated) |
| ---: | ---: | --- |
| 0 | 1 | what was once original has been co-opted so frequently that it now seems pedestrian . |
| 0 | 1 | hit and miss as far as the comedy goes and a big ole ' miss in the way of story . |
| 1 | 0 | you 'll gasp appalled and laugh outraged and possibly , watching the spectacle of a promis |
| 0 | 1 | an unwise amalgam of broadcast news and vibes . |
| 1 | 0 | this tenth feature is a big deal , indeed -- at least the third-best , and maybe even a no |

## Conclusions

1. **INT8 dynamic quantization halves model size (90.9 → 47.7 MB, 1.9× smaller) at a cost of 2.5 accuracy points (78.5% → 76.0%).** For CPU-bound serving this is a real tradeoff, not a free lunch.
2. **Latency is a wash on this hardware** (p50 within measurement noise, ~4–5 ms both). Dynamic INT8 mainly buys memory and footprint here, not speed — worth checking per backend before promising latency wins.
3. **The 13 disagreements are not random.** They cluster on subtly positive reviews with negation or sarcasm and on long comparative sentences — exactly where embedding margins are thin, so small weight perturbations flip the nearest centroid.
4. **Shared error modes dominate** quantization-induced ones. Most errors are made by both fp32 and INT8: the bottleneck is the centroid classifier plus SST-2 subtlety, not the precision.

## Limitations

What this eval does not show:

- Nearest-centroid on a general embedding model is a deliberately simple head. A fine-tuned classifier would raise all numbers. This eval isolates the quantization delta, not SOTA.
- 200 held-out examples gives about a ±3% confidence interval on accuracy deltas of this size. Treat 2.5 points as indicative; run 5 seeds before believing exact deltas.
- Latency is single-thread CPU, batch=1. Production numbers depend on batching and the backend engine.

## Reproducing

- Runtime: Python 3.12, PyTorch 2.x (`qnnpack` engine; this run used torch 2.14.1), `transformers`, `numpy`.
- Data and weights are public: SST-2 via the Hugging Face datasets-server API, [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from the Hub.
- Open [`eval-quantization-tradeoffs.ipynb`](eval-quantization-tradeoffs.ipynb) and run all cells. A fresh run takes about 2 minutes on a laptop CPU.
