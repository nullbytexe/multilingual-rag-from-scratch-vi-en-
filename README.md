# 🌐 Multilingual RAG System — From Scratch

> Vietnamese question → English corpus → Vietnamese answer, built entirely from scratch without pre-trained retrievers.

---

## 📌 Overview

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions in **Vietnamese** and receive answers from an **English knowledge base**, with the final answer translated back to Vietnamese.

Every core component — the tokenizer, encoder, reranker, and BM25 — is built from scratch using PyTorch. The only pre-built components used are the translation models and the generative LLM.

---

## 🏗️ Architecture

```
Query (Vietnamese)
       ↓
[Module 1] Vi→En Translation      ← Helsinki-NLP/opus-mt-vi-en
       ↓
[Module 2] BPE Tokenizer          ← Trained from scratch on WikiText-103 (400K docs)
       ↓
[Module 3] Bi-Encoder             ← Transformer 6L/384d, fine-tuned on MS MARCO (500K pairs)
       ↓
[Module 4] FAISS Index            ← Dense retrieval (IVFFlat)
       +
[Module 5] BM25                   ← Sparse retrieval (from scratch)
       ↓   Hybrid Search (α = 0.6)
[Module 6] Cross-Encoder Reranker ← Transformer 6L/384d, MS MARCO
       ↓
[Module 7] Qwen2.5-1.5B-Instruct  ← Answer generation (local, no API key needed)
       ↓
[Module 8] En→Vi Translation      ← Helsinki-NLP/opus-mt-en-vi
       ↓
Answer (Vietnamese)
```

---

## ✨ Features

- **BPE Tokenizer** — trained from scratch on 400K lines of WikiText-103 with 10,000 merge rules
- **Transformer Bi-Encoder** — 6 layers, d_model=384, fine-tuned with InfoNCE loss on 500K MS MARCO pairs
- **BM25** — full Okapi BM25 implementation from scratch (no libraries)
- **Hybrid Retrieval** — weighted fusion of dense (FAISS) and sparse (BM25) scores
- **Cross-Encoder Reranker** — 6-layer Transformer trained on 200K MS MARCO pairs
- **Local LLM** — Qwen2.5-1.5B-Instruct with optional 4-bit quantization (works on ≤16GB VRAM)
- **Multilingual** — Marian MT for Vietnamese ↔ English translation
- **87K-passage corpus** — SQuAD v1.1 training split

---

## 📊 Evaluation Results

Evaluated on 100 random samples from SQuAD v1.1 validation split:

| Metric | Score | Notes |
|---|---|---|
| Token F1 | — | Main generation quality metric |
| BLEU-4 | — | Low by design (generative model, not extractive) |
| ROUGE-L | — | Longest common subsequence overlap |
| Recall@1 | — | Did top-1 chunk contain the answer? |
| Recall@5 | — | Did top-5 chunks contain the answer? |
| Recall@10 | — | Did top-10 chunks contain the answer? |
| MRR@10 | — | Mean Reciprocal Rank of correct chunk |

> **Note on Exact Match (EM):** EM = 0 is expected for generative models. Qwen2.5 generates full sentences, not extracted spans. Token F1 and Recall@K are more meaningful metrics for this system. See the notebook for a detailed explanation.

*(Run the notebook to populate actual scores — they are saved to `evaluation_results_v2.csv`)*

---

## 🗂️ Project Structure

```
multilingual-rag-from-scratch/
│
├── rag_improved_fixed.ipynb      # Main notebook — all modules and training
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── assets/                       # Evaluation charts (generated on run)
│   ├── evaluation_metrics_v2.png
│   ├── evaluation_summary_v2.png
│   └── retrieval_analysis_v2.png
│
└── .gitignore
```

> **Generated files** (not committed to Git — created when you run the notebook):
> `bpe_tokenizer_v2.json`, `bi_encoder_v2.pt`, `cross_encoder_v2.pt`, `evaluation_results_v2.csv`, `rag_summary_v2.json`

---

## ⚙️ Installation

### Requirements

- Python 3.9+
- CUDA GPU recommended (tested on Colab/Kaggle with T4/A100)
- ~15–20GB disk space for datasets and model weights
- ~8GB VRAM minimum (4-bit quantization auto-enabled if VRAM < 16GB)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-rag-from-scratch.git
cd multilingual-rag-from-scratch
pip install -r requirements.txt
```

### HuggingFace Login (required for Qwen2.5)

```bash
huggingface-cli login
```

You need to accept the model license at: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

---

## 🚀 Running the Notebook

### On Google Colab (recommended)

1. Upload `rag_improved_fixed.ipynb` to Colab
2. Set runtime to **GPU** (T4 or better)
3. Run `!huggingface-cli login` with your HF token
4. Run all cells in order

### On Kaggle

1. Upload the notebook
2. Enable GPU accelerator
3. Add your `HF_TOKEN` to Kaggle Secrets
4. Run all cells

### Locally

```bash
jupyter notebook rag_improved_fixed.ipynb
```

> First run will download ~8GB of model weights. Subsequent runs load from cache.

---

## 📦 Datasets Used

| Dataset | Purpose | Size |
|---|---|---|
| WikiText-103 | BPE tokenizer training | ~400K lines |
| MS MARCO v1.1 | Bi-Encoder + Cross-Encoder training | 500K / 200K pairs |
| SQuAD v1.1 train | RAG corpus | ~87K passages |
| SQuAD v1.1 validation | Evaluation | 100 sampled QA pairs |

All datasets are downloaded automatically via HuggingFace `datasets` on first run.

---

## 💬 Example Usage

```python
result = rag_pipeline('Khi nào sự kiện Super Bowl 50 được tổ chức?')

# Output:
# VI: Khi nào sự kiện Super Bowl 50 được tổ chức?
# EN: When was Super Bowl 50 held?
# Answer (EN): Super Bowl 50 was held on February 7, 2016.
# Answer (VI): Super Bowl 50 được tổ chức vào ngày 7 tháng 2 năm 2016.
```

---

## 🔧 Configuration

Key hyperparameters (in the notebook):

| Parameter | Value | Description |
|---|---|---|
| `D_MODEL` | 384 | Transformer hidden dimension |
| `N_LAYERS` | 6 | Number of Transformer encoder layers |
| `N_HEAD` | 8 | Number of attention heads |
| `alpha` | 0.6 | Hybrid retrieval weight (dense vs sparse) |
| `CHUNK_SIZE` | 150 | Words per document chunk |
| `CHUNK_OVERLAP` | 30 | Overlap between chunks |
| `top_k_retrieve` | 20 | Candidates from hybrid retrieval |
| `top_k_rerank` | 5 | Top chunks after cross-encoder reranking |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Transformers | HuggingFace `transformers` |
| Dense Index | FAISS (IVFFlat) |
| Datasets | HuggingFace `datasets` |
| Evaluation | NLTK, rouge-score, sacrebleu |
| Visualization | Matplotlib, Pandas |

---

## 📝 Notes and Limitations

- **EM = 0 is expected.** This system uses a generative model, not an extractive span predictor. See the in-notebook explanation under the evaluation section.
- **First run is slow.** BPE training (~400K lines), Bi-Encoder training (3 epochs, 500K pairs), and Cross-Encoder training (2 epochs, 400K pairs) can take several hours on a T4 GPU. Subsequent runs load from saved checkpoints.
- **Translation quality** affects retrieval. Errors in Vi→En translation propagate through the pipeline. Complex or idiomatic Vietnamese questions may produce lower-quality results.
- **No re-training needed.** All trained model files (`.pt`, `.json`) are saved locally after the first run.

---

## 🙋 About

Built as a learning project exploring how to construct a full RAG pipeline from first principles — without relying on pre-built retrieval frameworks like LangChain or LlamaIndex.

If you find this useful or have suggestions, feel free to open an issue or submit a pull request.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
