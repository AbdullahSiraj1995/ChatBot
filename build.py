import os
import shutil
import time
import requests
import torch
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- LangChain & Chroma Components ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 

# --- 1. CONFIGURATION ---
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_URL = "https://www.ecc.de/en/risk-management/"
VECTORSTORE_PATH = "data/vectorstore_local"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_PAGES = 15

def log(message, icon="ℹ️"):
    print(f"{icon} {time.strftime('%H:%M:%S')} - {message}")

# --- 2. CRAWLER LOGIC ---
def crawl_site(base_url, max_pages=10):
    log(f"Starting Scraper on: {base_url}", "🕸️")
    to_visit = {base_url}
    visited = set()
    documents = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited: continue
            
        log(f"Scraping: {url}", "📄")
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            visited.add(url)
            
            soup = BeautifulSoup(res.text, 'html.parser')
            # Extract main text (targeting 'main' tag if available for cleaner data)
            main_area = soup.find('main') or soup.body
            text = main_area.get_text(separator=' ', strip=True)
            
            documents.append(Document(page_content=text, metadata={"source": url}))
            
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href']).split('#')[0].rstrip('/')
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    if full_url not in visited:
                        to_visit.add(full_url)
                        
        except Exception as e:
            log(f"Failed to scrape {url}: {e}", "⚠️")
    return documents

# --- 3. BUILD VECTOR STORE ---
def build_local_vector_store():
    log("Starting RAG Ingestion Process", "🏗️")

    # Clean start
    if os.path.exists(VECTORSTORE_PATH):
        log(f"Removing old vector store at {VECTORSTORE_PATH}", "🧹")
        shutil.rmtree(VECTORSTORE_PATH)

    # Step A: Crawl
    raw_docs = crawl_site(BASE_URL, MAX_PAGES)
    if not raw_docs:
        log("No documents captured. Exiting.", "❌")
        return None

    # Step B: Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(raw_docs)
    log(f"Created {len(chunks)} chunks from {len(raw_docs)} pages.")

    # Step C: Embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )

    # Step D: Initialize Chroma
    log(f"Creating ChromaDB at {VECTORSTORE_PATH}...", "🧠")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    log("RAG Ingestion Complete!", "✅")
    return vector_store

# --- 4. VERIFICATION TESTS ---
def run_post_build_tests(vector_store, test_queries):
    if not vector_store:
        print("\n❌ Vector store not available.")
        return

    print("\n" + "="*80)
    print("🧪 --- Running Verification Tests --- 🧪")
    print("="*80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- [Test {i}/{len(test_queries)}] QUERY: \"{query}\"")
        try:
            results = vector_store.similarity_search_with_score(query, k=2)
            if not results:
                print("⚠️ No relevant documents found.")
            else:
                for doc, score in results:
                    # Chroma scores are typically L2 distance (lower is better for some models)
                    print(f" ✅ [Score: {score:.4f}] [Source: {doc.metadata.get('source')}]")
                    print(f"    Snippet: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"💥 Test Error: {e}")

if __name__ == "__main__":
    store = build_local_vector_store()
    
    # Updated queries relevant to the ECC Risk Management site
    test_queries = [
        "What are the core components of ECC risk management?",
        "How does the clearing house handle margin requirements?",
        "What is the default management process?",
        "Information about credit risk and liquidity risk."
    ]
    
    run_post_build_tests(store, test_queries)