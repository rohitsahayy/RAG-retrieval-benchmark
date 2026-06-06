import sys
import time
import os
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from src.pipeline import load_document,chunk,build_vectorstore,build_qdrant_vectorstore,retrieve,generate_answer
from src.evaluation.metrics import evaluate_rag

# dagshub connection 
import dagshub
dagshub.init(
    repo_owner="rohit.sahay0660",
    repo_name="RAG-retrieval-benchmark",
    mlflow=True
)

# -----------------------------------------------------------
# WHAT IS MLFLOW DOING HERE?
#
# MLflow tracks "runs" — each run is one experiment attempt.
# You log: parameters (what you tried) + metrics (how it performed)
# Then MLflow UI lets you compare all runs in a table/chart.
#
# Think of it like a lab notebook that never loses your results.
# -----------------------------------------------------------

DOCUMENT_PATH = "data/sample_docs/attention_mech.pdf"

TEST_QUERIES = [
    "What is the attention mechanism?",
    "How does multi-head attention work?",
    "What are the limitations of this approach?",
]

def run_experiment(
        chunk_size:int,
        chunk_overlap:int,
        retrieval_strategy :str,
        vector_store: str = "faiss",
        top_k:int =5
):
    # Each call to this function = one MLflow run
    # run_name makes it readable in the UI

    run_name = f"chunk{chunk_size}_overlap{chunk_overlap}_strategy{retrieval_strategy}_{vector_store}"

    with mlflow.start_run(run_name=run_name):
        # --- LOG PARAMETERS ---
        # Parameters = what you chose / configured
        # These are the "inputs" to your experiment

        mlflow.log_param("chunk_size",chunk_size)
        mlflow.log_param("chunk_overlap",chunk_overlap)
        mlflow.log_param("retrieval_strategy",retrieval_strategy)
        mlflow.log_param("vector_store", vector_store)  
        mlflow.log_param("top_k",top_k)
        mlflow.log_param("embedding_model","all-MiniLM-L6-v2")
        mlflow.log_param("document",DOCUMENT_PATH)

        # RUN Pipeline
        docs = load_document(DOCUMENT_PATH)
        chunks = chunk(docs,chunk_size,chunk_overlap)

        if vector_store=="faiss":
            store = build_vectorstore(chunks)
        elif vector_store=="qdrant":
            store = build_qdrant_vectorstore(chunks)

        #LOG metrics
        mlflow.log_metric("num_chunks",len(chunks))

        # Run all test queries, collect latencies
        latencies = []
        all_contexts = []
        all_answers = []

        for query in TEST_QUERIES:
            results,latency_ms = retrieve(store,query,k=top_k,strategy=retrieval_strategy)
            latencies.append(latency_ms)

            answer = generate_answer(query,results)

            context_texts = [doc.page_content for doc in results]
            all_contexts.append(context_texts)
            all_answers.append(answer)
            
        # Summary metrics across all queries
        avg_latency = round(sum(latencies) / len(latencies), 2)
        max_latency = round(max(latencies), 2)
        min_latency = round(min(latencies), 2)

        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_metric("max_latency_ms", max_latency)
        mlflow.log_metric("min_latency_ms", min_latency)

        # --- RUN RAGAS EVALUATION ---
        print(f"\n[RAGAS] Evaluating quality scores...")
        ragas_scores = evaluate_rag(
            queries=TEST_QUERIES,
            retrieved_contexts=all_contexts,
            generated_answers=all_answers
        )

        # log RAGAS metrics :
        mlflow.log_metric("context_precision", ragas_scores["context_precision"])
        mlflow.log_metric("faithfulness",      ragas_scores["faithfulness"])
        mlflow.log_metric("answer_relevancy",  ragas_scores["answer_relevancy"])

        print(f"\n{'='*50}")
        print(f"Run: {run_name}")
        print(f"Chunks: {len(chunks)}")
        print(f"Avg latency: {avg_latency}ms")
        print(f"Max latency: {max_latency}ms")
        print(f"Context Precision:{ragas_scores['context_precision']}")
        print(f"Faithfulness:     {ragas_scores['faithfulness']}")
        print(f"{'='*50}\n")

if __name__ =="__main__":
    mlflow.set_experiment("rag-retrieval-bench")

    # We'll run 5 experiments back to back
    # Each one changes ONE variable so results are comparable

    run_experiment(
        chunk_size=512,
        chunk_overlap=50,
        retrieval_strategy="naive"
    )

    run_experiment(
        chunk_size=256,
        chunk_overlap=30,
        retrieval_strategy="naive"
    )

    run_experiment(
        chunk_size=512,
        chunk_overlap=50,
        retrieval_strategy="mmr"
    )

    # chunk 512 + overlap 50 + retrieve strategy "naive" + VDB Qdrant
    run_experiment(chunk_size=512, 
                   chunk_overlap=50,
                   retrieval_strategy="naive", 
                   vector_store="qdrant")

    # chunk 512 + overlap 50 + retrieve strategy "mmr" + VDB Qdrant
    run_experiment(chunk_size=512, 
                   chunk_overlap=50,
                   retrieval_strategy="mmr",   
                   vector_store="qdrant")
