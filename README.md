# rag-retrieval-bench

Benchmarking RAG retrieval strategies across latency and quality metrics.
Tracks every experiment with MLflow — comparing chunking strategies, retrieval 
methods, and their effect on answer quality using RAGAS evaluation.

---

## What This Project Does

Most RAG tutorials stop at "it works." This project asks:
- **How well does it work?**
- **Which retrieval strategy is fastest?**
- **Does faster always mean better quality?**

Each experiment is tracked in MLflow with both latency and RAGAS quality scores,
making tradeoffs between speed and accuracy explicitly measurable.

---

## Stack

| Layer | Tool |
|---|---|
| RAG Framework | LangChain |
| Vector Store | FAISS (baseline) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq (llama-3.1-8b-instant) |
| Evaluation | RAGAS |
| Experiment Tracking | MLflow |

---

## Retrieval Strategies Benchmarked

- **Naive top-k** — baseline similarity search
- **MMR** — Maximal Marginal Relevance, balances relevance with diversity

## Chunking Strategies

- **Fixed 512** — 512 character chunks, 50 overlap
- **Fixed 256** — 256 character chunks, 30 overlap

---

## Results

Evaluated on the "Attention Is All You Need" paper across 3 queries.

| Strategy | Chunk Size | Avg Latency | Context Precision | Faithfulness | Answer Relevancy |
|---|---|---|---|---|---|
| Naive | 512 | 12.36ms | 0.5463 | 0.6667 | 0.6059 |
| Naive | 256 | 19.40ms | 0.2333 | 0.6667 | 0.2993 |
| MMR | 512 | 19.89ms | 0.3241 | 0.6061 | 0.5772 |

---

## Key Findings

**1. Smaller chunks degraded quality significantly**
chunk256 scored 57% lower on context precision (0.23 vs 0.55) compared to
chunk512 — while also being slower. Smaller chunks lose sentence context,
making retrieved fragments semantically weaker.

**2. MMR added latency without quality gains on this corpus**
MMR is designed for diverse document sets. On a focused academic paper,
naive retrieval already returns varied chunks — MMR's diversity penalty
hurt precision without benefit.

**3. Faithfulness was stable across all strategies (0.60–0.67)**
The LLM consistently grounded answers in retrieved context regardless
of strategy. Hallucination risk is low across all configurations tested.

**4. Winning configuration: chunk512 + naive retrieval**
Fastest latency (12.36ms) AND highest quality scores across all three metrics.

---

## Project Structure

```
rag-retrieval-bench/
├── data/
│   └── sample_docs/        ← add your own PDF here
├── src/
│   ├── pipeline.py         ← load, chunk, embed, retrieve, generate
│   ├── chunking/           ← chunking strategies
│   ├── retrieval/          ← retrieval strategies  
│   └── evaluation/
│       └── metrics.py      ← RAGAS evaluation
├── experiments/
│   └── run_experiment.py   ← MLflow experiment runner
├── .env.example
├── requirements.txt
└── README.md
```

---

## Run It Yourself

```bash
git clone https://github.com/rohitsahayy/rag-retrieval-bench
cd rag-retrieval-bench

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key:
```bash
cp .env.example .env
# edit .env and add GROQ_API_KEY=your_key_here
```

Add a PDF to `data/sample_docs/` then run:
```bash
python experiments/run_experiment.py
```

View results:
```bash
mlflow ui --port 5500
# open http://127.0.0.1:5500
```

---

## Roadmap

- [ ] Qdrant vector store with HNSW indexing
- [ ] Semantic chunking strategy
- [ ] Redis semantic cache
- [ ] HyDE retrieval strategy
- [ ] FastAPI backend
- [ ] Live comparison UI