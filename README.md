# 🌐 Multilingual RAG System — From Scratch

> Vietnamese question → English corpus → Vietnamese answer, built entirely from scratch without pre-trained retrievers.

---

## 📌 Overview

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline that lets users ask questions in **Vietnamese** and get answers pulled from an **English knowledge base**, with the final answer translated back into Vietnamese.

Every core component — the tokenizer, encoder, reranker, and BM25 — is built from scratch using PyTorch. The only pre-built components are the translation models and the generative LLM.

Everything lives in a single notebook: **`rag_sys.ipynb`**.

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

## 📊 Evaluation

Evaluation is run inline in the notebook on a sample of 100 questions from the SQuAD v1.1 validation split, reporting:

| Metric    | Notes                                            |
| --------- | ------------------------------------------------ |
| Token F1  | Main generation quality metric                   |
| BLEU-4    | Low by design (generative model, not extractive) |
| ROUGE-L   | Longest common subsequence overlap               |
| Recall@1  | Did the top-1 chunk contain the answer?           |
| Recall@5  | Did the top-5 chunks contain the answer?          |
| Recall@10 | Did the top-10 chunks contain the answer?         |
| MRR@10    | Mean Reciprocal Rank of the correct chunk         |

> **Note on Exact Match (EM):** EM = 0 is expected for this system. Qwen2.5 generates full sentences rather than extracting spans, so Token F1 and Recall@K are the more meaningful metrics here — see the evaluation section inside the notebook for the full explanation.

Run the notebook end-to-end to populate the actual scores and generated charts.

---

## 🗂️ Repository Structure

```
multilingual-rag-from-scratch-vi-en/
│
├── rag_sys.ipynb      # Main (and only) notebook — all modules, training, and evaluation
├── README.md          # This file
└── LICENSE             # MIT license
```

> **Generated at runtime** (not committed to the repo — created when you run the notebook): trained tokenizer/model checkpoints, evaluation result CSVs, and evaluation charts. Since there's no `requirements.txt` yet, install dependencies as you hit `import` errors, or export your own `pip freeze` after your first successful run.

---

## ⚙️ Installation

### Requirements

- Python 3.9+
- CUDA GPU recommended (tested on Colab/Kaggle with T4/A100)
- ~15–20GB disk space for datasets and model weights
- ~8GB VRAM minimum (4-bit quantization recommended if VRAM < 16GB)

### Setup

```bash
git clone https://github.com/nullbytexe/multilingual-rag-from-scratch-vi-en.git
cd multilingual-rag-from-scratch-vi-en
```

Install the core dependencies used across the notebook:

```bash
pip install torch transformers datasets faiss-cpu sentencepiece \
            nltk rouge-score sacrebleu matplotlib pandas huggingface_hub
```

(Swap `faiss-cpu` for `faiss-gpu` if you have CUDA set up for it.)

### HuggingFace Login (required for Qwen2.5)

```bash
huggingface-cli login
```

You'll need to accept the model license at: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

---

## 🚀 Running the Notebook

### On Google Colab (recommended)

1. Upload `rag_sys.ipynb` to Colab
2. Set the runtime to **GPU** (T4 or better)
3. Run `!huggingface-cli login` with your HF token
4. Run all cells in order, top to bottom

### On Kaggle

1. Upload the notebook
2. Enable the GPU accelerator
3. Add your `HF_TOKEN` to Kaggle Secrets
4. Run all cells

### Locally

```bash
jupyter notebook rag_sys.ipynb
```

> First run downloads several GB of datasets and model weights. Subsequent runs are faster once things are cached locally.

---

## 📦 Datasets Used

| Dataset               | Purpose                             | Size                 |
| --------------------- | ------------------------------------ | --------------------- |
| WikiText-103          | BPE tokenizer training               | ~400K lines           |
| MS MARCO v1.1         | Bi-Encoder + Cross-Encoder training  | 500K / 200K pairs     |
| SQuAD v1.1 train      | RAG corpus                           | ~87K passages          |
| SQuAD v1.1 validation | Evaluation                           | 100 sampled QA pairs   |

All datasets are downloaded automatically via the HuggingFace `datasets` library on first run.

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

Key hyperparameters (set near the top of the notebook):

| Parameter        | Value | Description                                |
| ----------------- | ----- | ------------------------------------------- |
| `D_MODEL`         | 384   | Transformer hidden dimension                 |
| `N_LAYERS`        | 6     | Number of Transformer encoder layers         |
| `N_HEAD`          | 8     | Number of attention heads                    |
| `alpha`           | 0.6   | Hybrid retrieval weight (dense vs sparse)    |
| `CHUNK_SIZE`      | 150   | Words per document chunk                     |
| `CHUNK_OVERLAP`   | 30    | Overlap between chunks                       |
| `top_k_retrieve`  | 20    | Candidates from hybrid retrieval             |
| `top_k_rerank`    | 5     | Top chunks kept after cross-encoder reranking |

---

## 🛠️ Tech Stack

| Component     | Technology                     |
| -------------- | -------------------------------- |
| Deep Learning  | PyTorch                          |
| Transformers   | HuggingFace `transformers`       |
| Dense Index    | FAISS (IVFFlat)                  |
| Datasets       | HuggingFace `datasets`           |
| Evaluation     | NLTK, rouge-score, sacrebleu     |
| Visualization  | Matplotlib, Pandas               |

---

## 📝 Notes and Limitations

- **EM = 0 is expected.** This system uses a generative model, not an extractive span predictor — see the in-notebook explanation in the evaluation section.
- **First run is slow.** BPE training (~400K lines), Bi-Encoder training (500K pairs), and Cross-Encoder training (200K pairs) can take a while on a single T4 GPU. Subsequent runs are faster once checkpoints are cached.
- **Translation quality matters.** Errors in Vi→En translation propagate through the whole pipeline — complex or idiomatic Vietnamese questions may produce lower-quality results.
- **Single-notebook design.** There's currently no `requirements.txt` or packaged Python module — everything, including dependency installs, happens inside `rag_sys.ipynb`.

---

## 🙋 About

Built as a learning project exploring how to construct a full RAG pipeline from first principles — without relying on pre-built retrieval frameworks like LangChain or LlamaIndex.

If you find this useful or have suggestions, feel free to open an issue or submit a pull request.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
