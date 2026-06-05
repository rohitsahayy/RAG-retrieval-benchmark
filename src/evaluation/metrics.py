import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

def evaluate_rag(
        queries:list[str],
        retrieved_contexts:list[list[str]],
        generated_answers:list[str]
)->dict:
    """
    queries ->list of questions asked
    retrieved_context -> for each query , list of chunk texts retrieved
    generated_answers -> LLM answer for each query  

    Returns dict of metric score
    """

    llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
    )

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # RAGAS expects a HuggingFace Dataset format
    # Each row = one query with its context and answer

    data = {
        "user_input":queries,
        "retrieved_contexts":retrieved_contexts,
        "response":generated_answers
    }

    dataset = Dataset.from_dict(data)

    # RunConfig — fixes Groq rate limit timeouts
    # timeout=120  → wait up to 120s per LLM call
    # max_retries=5 → retry failed calls before giving up
    run_config = RunConfig(timeout=120, max_retries=5,max_workers=1)

    results = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithoutReference(),
            faithfulness,
            answer_relevancy
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    # Debug — print actual keys so we know exactly what RAGAS returned
    results_df = results.to_pandas()
    print(f"\n[RAGAS] Available columns: {list(results_df.columns)}")

    # Helper to safely extract score — returns None if key missing
    def get_score(key: str) -> float | None:
        if key in results_df.columns:
            val = results_df[key].mean()
            return round(float(val), 4)
        print(f"[RAGAS] Warning: '{key}' not found in results")
        return None

    scores = {
        "context_precision": get_score("llm_context_precision_without_reference"),
        "faithfulness":      get_score("faithfulness"),
        "answer_relevancy":  get_score("answer_relevancy"),
    }

    print("\n--- RAGAS SCORES ---")
    for metric, score in scores.items():
        print(f"{metric:25s} → {score}")

    return scores
