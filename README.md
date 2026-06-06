# RAG Retrieval Benchmark

> Benchmarking RAG retrieval strategies across latency and quality metrics — every experiment tracked with MLflow on DagsHub.

**Live Experiments:** https://dagshub.com/rohit.sahay0660/RAG-retrieval-benchmark.mlflow

---

## What This Project Does

Most RAG tutorials stop at "it works." This project asks:
- **Which retrieval strategy is fastest?**
- **Does faster always mean better quality?**
- **How does chunk size affect answer relevancy and faithfulness?**

Each configuration is logged as an MLflow experiment with both latency
and RAGAS quality scores — making tradeoffs between speed and accuracy
explicitly measurable rather than assumed.

---

## Stack

| Layer | Tool |
|---|---|
| RAG Framework | LangChain |
| Vector Store | FAISS (baseline) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq (llama-3.1-8b-instant) |
| Evaluation | RAGAS |
| Experiment Tracking | MLflow + DagsHub |

---

## Strategies Benchmarked

**Chunking**
- Fixed 512 — 512 character chunks, 50 overlap (baseline)
- Fixed 256 — 256 character chunks, 30 overlap

**Retrieval**
- Naive top-k — standard cosine similarity search
- MMR — Maximal Marginal Relevance, balances relevance with diversity

**Evaluation Metrics (RAGAS)**
- Context Precision — were the retrieved chunks actually relevant?
- Faithfulness — is the answer grounded in context or hallucinated?
- Answer Relevancy — does the answer address what was asked?

---

## Results

Evaluated on the "Attention Is All You Need" paper across 3 queries.

| Strategy | Chunk Size | Avg Latency | Context Precision | Faithfulness | Answer Relevancy |
|---|---|---|---|---|---|
| **Naive** | **512** | **12.36ms** | **0.5463** | **0.6667** | **0.6059** |
| Naive | 256 | 19.40ms | 0.2333 | 0.6667 | 0.2993 |
| MMR | 512 | 19.89ms | 0.3241 | 0.6061 | 0.5772 |

*Bold = best performing configuration*

---

## Key Findings

**1. Smaller chunks degraded quality significantly**
chunk256 scored 57% lower on context precision (0.23 vs 0.55) compared
to chunk512 — while also being 57% slower. Smaller chunks lose sentence
context, making retrieved fragments semantically weaker even though there
are more of them.

**2. MMR added latency without quality gains on this corpus**
MMR is designed for large, diverse document sets where naive retrieval
returns redundant chunks. On a focused academic paper, naive retrieval
already returns varied chunks — MMR's diversity penalty hurt precision
without any benefit.

**3. Faithfulness was stable across all strategies (0.60–0.67)**
The LLM consistently grounded its answers in retrieved context regardless
of chunking or retrieval strategy. Hallucination risk did not meaningfully
change across configurations tested.

**4. Winning configuration: chunk512 + naive retrieval**
Fastest latency (12.36ms) AND highest scores across all three quality
metrics. For single-document RAG over academic papers, larger chunks
with naive retrieval is the optimal baseline.

---

## Project Structure

```
rag-retrieval-bench/
├── data/
│   └── sample_docs/            ← add your own PDF here
├── src/
│   ├── pipeline.py             ← load, chunk, embed, retrieve, generate
│   ├── chunking/               ← chunking strategy implementations
│   ├── retrieval/              ← retrieval strategy implementations
│   └── evaluation/
│       └── metrics.py          ← RAGAS evaluation wrapper
├── experiments/
│   └── run_experiment.py       ← MLflow + DagsHub experiment runner
├── .env.example
├── requirements.txt
└── README.md
```

---

## Run It Yourself

```bash
git clone https://github.com/rohitsahayy/RAG-retrieval-benchmark
cd RAG-retrieval-benchmark

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up environment variables:
```bash
cp .env.example .env
# add your GROQ_API_KEY to .env
```

Add any PDF to `data/sample_docs/` then run:
```bash
python experiments/run_experiment.py
```

View results locally:
```bash
mlflow ui --port 5500
# open http://127.0.0.1:5500
```

Or view all tracked experiments publicly:
**https://dagshub.com/rohit.sahay0660/RAG-retrieval-benchmark.mlflow**

---