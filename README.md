# MIRAGE — Multimodal Image Retrieval via Augmented Generative Enrichment

A training-free hybrid pipeline for text-to-image retrieval that augments a frozen **CLIP** dense retriever with a **BM25** sparse index built over **BLIP-2** generated descriptions of each image, fused at query time.

> Jeremy Jornet, Eden Sasson — Technion, Israel Institute of Technology.
> Course project, *Advanced Information Retrieval*.

---

## Overview

Dense dual-encoders such as CLIP dominate text-to-image retrieval but compress away fine-grained descriptive details (color, counting, rare terms, in-image text). MIRAGE re-introduces these signals through a sparse lexical channel:

1. Each image is described offline by BLIP-2 — either via prompt-free captioning (**MIRAGE-Cap**) or via fixed VQA prompts (**MIRAGE-VQA**).
2. The resulting documents are indexed with BM25.
3. At query time, CLIP and BM25 are queried in parallel and their rankings are combined by one of four fusion strategies: **RRF**, **Borda**, **CombSUM**, or **Weighted** ($\alpha \in \{0.7, 0.9\}$).

The pipeline is fully training-free: no fine-tuning, no end-to-end joint encoder. BLIP generation is performed once, offline, and adds no retrieval-time latency.

## Key findings

- **Score-based fusion yields small but statistically significant early-rank gains** over CLIP (up to +1.3 R@3 on Flickr30k, +2.0 R@5 on VizWiz).
- **Rank-based fusion (RRF, Borda) collapses the dense signal**, particularly at R@1.
- **MIRAGE-VQA consistently outperforms MIRAGE-Cap** across all metrics and both datasets.
- The BM25 weight $\alpha$ acts as an explicit control over the rank depth at which lexical evidence intervenes.
- Aggregate gains mask a **redistribution effect**: a subset of queries is promoted to the top while another subset is pushed into a long tail, particularly pronounced on noisy imagery.

Full results, ablations, and qualitative analysis are reported in the paper.

## Datasets

| Dataset | # Images | Characteristics |
|---|---|---|
| [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) | 31,783 | Clean, diverse everyday scenes |
| [VizWiz-Captions](https://vizwiz.org/tasks-and-datasets/image-captioning/) | 23,954 | Blur, framing, exposure issues |

For each image we keep a single caption (the longest available) as the textual query.

## Models

- **Dense retriever:** [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
- **Generative VLM:** [`Salesforce/blip2-flan-t5-xl`](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
- **Sparse retriever:** Okapi BM25 (`rank_bm25`)
- **Vector index:** FAISS `IndexFlatIP` (exhaustive cosine search)

## Repository structure

```
.
├── advance_ir.ipynb     # Single end-to-end Colab notebook
└── README.md
```

The notebook is organized as a linear pipeline:

| Section | Stage |
|---|---|
| 1 | Dataset loading & annotation parsing (Flickr30k, VizWiz) |
| 2 | Dense indexing — CLIP image embeddings → FAISS |
| 3 | Sparse indexing — BLIP-2 generation (5 captions + 5 VQA answers per image) → BM25 |
| 4 | CLIP-only baseline ranking + BM25 Only |
| 5 | Hybrid ranking — RRF, Borda, CombSUM, Weighted ($\alpha=0.7, 0.9$) |
| 6 | Quantitative analysis — R@K, MRR, mean/median rank, improvement & no-degradation rates, significance tests |
| 7 | Visualizations — rank distributions (KDE) on a log axis |
| 8 | Qualitative analysis — improved/degraded queries with visual examples |
| 9 | *Bonus* — RM3 and WordNet query expansion (not discussed in the report) |

## Reproduction

The notebook is designed for **Google Colab with a GPU** and **Google Drive** storage.

1. **Mount Drive** and place datasets under `MyDrive/Advance-ir/{Flickr,VizWiz}/`:
    - Flickr30k images + `results.csv`
    - VizWiz training images + `train.json`
2. **Open** `advance_ir.ipynb` in Colab and run sections sequentially.
3. Set the `DATASET` variable in each `CONFIG` block to `"Flickr"` or `"VizWiz"` and re-run the indexing/ranking cells.

Each stage writes its outputs to Drive (FAISS index, `bm25_*.pkl`, `blip_{dataset}.json`, `{dataset}_annotations.json`) and is independently re-runnable. Heavy steps (CLIP embedding, BLIP generation) are cached and need to be run only once per dataset.


## License

Released for academic use. Please refer to the licenses of the underlying assets ([CLIP](https://github.com/openai/CLIP/blob/main/LICENSE), [BLIP-2](https://github.com/salesforce/LAVIS/blob/main/LICENSE.txt), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), [VizWiz](https://vizwiz.org/)).
