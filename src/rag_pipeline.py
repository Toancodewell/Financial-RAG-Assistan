import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file
api_key = os.getenv("groq_api_key")

# LOAD VECTOR DATABASE

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LOAD VECTOR DATABASE

vectorstore = Chroma(
    persist_directory="../Data/chroma_samsung_db",
    embedding_function=embed_model,
    collection_name="samsung_financials"
)

# Retrieve more docs for reranking
retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

# LOAD RERANKER (Cross Encoder)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_docs(query, docs, top_k=4):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked_docs[:top_k]]

# ==========================================================
# 4️⃣ FORMAT DOCUMENTS
# ==========================================================

def format_docs(docs):
    return "\n\n".join(
        f"Source file: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )
# RETRIEVE + RERANK PIPELINE

def retrieve_and_rerank(query: str):
    docs = retriever.invoke(query)
    reranked_docs = rerank_docs(query, docs, top_k=4)
    return format_docs(reranked_docs)

# LLM SETUP

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=api_key
)
# PROMPT TEMPLATE

template = """You are a professional financial analysis assistant.

You MUST answer strictly based on the provided context.
Do NOT use outside knowledge.

If the answer is not found in the context, say:
"I could not find specific data for this request."

Context:
{context}

Question:
{question}

Requirements:
- Provide structured and clear information.
- Always mention the source file name.
- Do not speculate.
- Include units for all numerical data (KRW, USD, %, etc.).

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

#  BUILD FINAL RAG CHAIN

rag_chain = (
    {
        "context": RunnableLambda(retrieve_and_rerank),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
def ask_question(query: str):
    try:
        return rag_chain.invoke(query)
    except Exception as e:
        return f"Error: {str(e)}"


# Run : uvicorn fastapi_server:app --reload 