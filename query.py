import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION ---
load_dotenv()
VECTORSTORE_PATH = "data/vectorstore_local"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

def format_docs(docs):
    """Formats retrieved documents for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)

def launch_rag():
    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Change to 'cuda' if GPU is available
    )
    
    # Load Chroma (The Modern Way)
    if not os.path.exists(VECTORSTORE_PATH):
        print(f"❌ Error: {VECTORSTORE_PATH} not found. Run the build script first.")
        return

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH, 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

    # --- 2. THE PROMPT ---
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
    to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # --- 3. THE CHAIN (MODERN LCEL) ---
    # This replaces 'create_retrieval_chain' and 'create_stuff_documents_chain'
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ System Ready (LCEL)! Type 'exit' to stop.\n" + "-"*40)

    while True:
        query = input("\nMake a Query: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
            
        print("🔍 Thinking...")
        try:
            # In LCEL, we just invoke the chain with the string input
            response = rag_chain.invoke(query)
            print(f"\n🤖 **Answer:**\n{response}")
            
            # If you still want to see sources, you'll need to call retriever separately
            # as the LCEL chain above specifically outputs the parsed string.
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    launch_rag()