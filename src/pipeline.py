import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------
# STEP 1: LOAD
# -----------------------------------------------------------
# Takes a file path, returns a list of LangChain Document objects.
# Each Document has .page_content (text) and .metadata (page number etc.)
# We support PDF and plain text for now.

def load_document(file_path:str)-> list[Document]:
    path = Path(file_path)

    if path.suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif path.suffix == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file path: {path.suffix}")
    
    documents = loader.load()
    print(f"[LOAD] Loaded {len(documents)} page(s) from {path.name}")
    return documents


# -----------------------------------------------------------
# STEP 2: CHUNK
# -----------------------------------------------------------
# Takes documents, returns smaller chunks.
# chunk_size     = max characters per chunk
# chunk_overlap  = how many characters bleed into the next chunk
#
# WHY overlap? So a sentence that falls at a boundary
# doesn't lose context. Think of it like a sliding window.

def chunk(
        documents:list[Document],
        chunk_size : int = 512,
        chunk_overlap :int = 50,
)-> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[CHUNK] Split into {len(chunks)} chunks"
          f"(size={chunk_size},overlap={chunk_overlap}")
    return chunks

# -----------------------------------------------------------
# STEP 3: EMBED + INDEX
# -----------------------------------------------------------
# Takes chunks, converts each to a vector, stores in FAISS.
# 
# WHY vectors? Computers can't compare meaning of text directly.
# Vectors let us find "semantically similar" chunks using
# math (cosine similarity) instead of keyword matching.

def build_vectorstore(chunks:list[Document])->FAISS:
    print("[EMBED] Loading embedding model..")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print(f"[EMBED] Embedding {len(chunks)} chunks...")
    start = time.time()
    vectorstore = FAISS.from_documents(chunks,embedding)
    elapsed = round((time.time()-start)*1000,2)

    print(f"[EMBED] Done in {elapsed} ms")
    return vectorstore


# -----------------------------------------------------------
# STEP 4: RETRIEVE
# -----------------------------------------------------------
# Given a query string, finds the top-k most relevant chunks.
# 
# This is the step we benchmark.
# We measure: how long it takes + how good the results are.
#
# Returns: (results, latency_ms)

def retrieve(
        vectorstore:FAISS,
        query:str,
        k:int = 5
) -> tuple[list[Document],float]:
    
    start = time.time()
    results = vectorstore.similarity_search(query,k=k)
    latency_ms = round((time.time()-start)*1000,2)

    print(f"[RETRIEVE] Query : {query[:50]}...")
    print(f"[RETRIEVE] Got {len(results)} chunks in {latency_ms}ms")
    return results,latency_ms

# -----------------------------------------------------------
# QUICK SANITY TEST
# -----------------------------------------------------------
# Run this file directly to verify the pipeline works
# before plugging in different strategies.

if __name__ == "__main__":
    import sys

    if len(sys.argv)<2:
        print("Usage : python src/pipeline.py <path_to_document>")
        sys.exit(1)

    file_path = sys.argv[1]

    docs = load_document(file_path)
    chunks = chunk(docs,chunk_size=512,chunk_overlap=50)
    store = build_vectorstore(chunks)
    results,latency = retrieve(store,query="Summarize this document")

    print(f"\n--- TOP RESULT ---")
    print(results[0].page_content[:300])
    print(f"\nLatency: {latency}ms")