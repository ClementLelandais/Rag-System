"""
RAG Backend FastAPI - Version SINGLETON (anti-double-chargement)
"""
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import torch
from huggingface_hub import login
from pathlib import Path

# Charge .env automatiquement
load_dotenv()

if os.environ.get("CONDA_PREFIX"):
    os.environ["LD_LIBRARY_PATH"] = (
        os.environ["CONDA_PREFIX"] + "/lib:" +
        os.environ.get("LD_LIBRARY_PATH", "")
    )

if 'RAG_COMPONENTS' not in globals():
    print("[SINGLETON] Chargement composants RAG...")

    from langchain_groq import ChatGroq
    from src.backend.embedder import Embedder
    from src.backend.vector_store import MultiVectorStore
    from src.backend.rag import RAG
    from src.backend.reranker import Reranker

    login(token=os.getenv("HF_TOKEN"))

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    GROQ_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device détecté: {DEVICE}")

    print("Loading Embedder...")
    embedder = Embedder(model_name=EMBEDDING_MODEL, device=DEVICE)

    print("Loading MultiVectorStore...")
    vector_store = MultiVectorStore(datasets_path="./data")

    # Log datasets chargés
    stats = vector_store.get_stats()
    print(f"Datasets chargés : {stats['datasets']} | {stats['total_chunks']} chunks")

    print("Loading LLM (Groq)...")
    llm = ChatGroq(
        model=GROQ_LLM_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
        max_tokens=2000,
    )

    print("Loading Reranker...")
    reranker = Reranker(
        model_name=RERANKER_MODEL,
        device=torch.device(DEVICE),
    )

    print("Initializing RAG...")
    rag = RAG(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        reranker=reranker,
        top_k=5,
        num_candidates=20,
    )

    RAG_COMPONENTS = {
        'embedder': embedder,
        'vector_store': vector_store,
        'llm': llm,
        'reranker': reranker,
        'rag': rag,
        'device': DEVICE,
        'llm_model': GROQ_LLM_MODEL,
    }
    print("[SINGLETON] TOUS composants chargés !")
else:
    print("[SINGLETON] Composants déjà chargés - skip")


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_source(s: str) -> str:
    """Rend les IDs de sources lisibles."""
    return (s.replace("bsard_doc_", "BSARD #")
             .replace("syntec_doc_", "Syntec #")
             .replace("alloprof_", "Alloprof #"))


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Acollab API", version="1.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    rag = RAG_COMPONENTS['rag']
    result = rag.ask(req.query)
    answer = result['answer']

    if 'Pas dans les documents' in answer:
        return QueryResponse(answer="Aucune information trouvée.", sources=[])
    if '---' in answer:
        answer = answer.split('---')[0].strip()
    if 'QUESTION:' in answer:
        answer = answer.split('QUESTION:')[0].strip()

    sources = [format_source(s) for s in result.get('sources', [])]
    return QueryResponse(answer=answer, sources=sources)

@app.get("/")
def read_root():
    return {
        "RAG LIVE": True,
        "docs": "/docs",
        "device": RAG_COMPONENTS['device'],
        "llm_model": RAG_COMPONENTS['llm_model'],
        "reranker": getattr(RAG_COMPONENTS['reranker'], 'model_name', 'N/A'),
        "datasets": RAG_COMPONENTS['vector_store'].get_stats()['datasets'],
        "test": "POST /query {'query': 'CDI résiliation Syntec'}",
    }

@app.get("/health")
def health_check():
    stats = RAG_COMPONENTS['vector_store'].get_stats()
    return {
        "status": "healthy",
        "device": RAG_COMPONENTS['device'],
        "reranker_active": RAG_COMPONENTS['reranker'] is not None,
        "datasets": stats['datasets'],
        "total_chunks": stats['total_chunks'],
    }

@app.post("/debug_search")
async def debug_search(request: dict):
    """DEBUG : Chunks retrieval direct avec reranking"""
    query = request["query"]
    rag = RAG_COMPONENTS['rag']
    docs = rag.retrieve(query)
    return {
        "query": query,
        "reranker_active": rag.reranker is not None,
        "num_candidates": rag.num_candidates,
        "top_k": rag.top_k,
        "found_chunks": len(docs),
        "top_chunks": [
            {
                "rank": i + 1,
                "dataset": doc.get("dataset", "?"),
                "rerank_score": doc.get("rerank_score", "N/A"),
                "score": doc.get("score", "N/A"),
                "content": (doc.get("text") or doc.get("chunk_text") or "")[:200],
                "title": doc.get("title", ""),
            }
            for i, doc in enumerate(docs[:5])
        ],
    }


# ── Gradio ────────────────────────────────────────────────────────────────────
import gradio as gr

def gradio_chat(message, history):
    if not message.strip():
        return ""

    rag = RAG_COMPONENTS['rag']

    if history:
        conversation = ""
        for user_msg, bot_msg in history[:]:
            bot_clean = bot_msg.split("\n\n")[0] if bot_msg else ""
            conversation += f"Utilisateur: {user_msg}\nAssistant: {bot_clean}\n\n"
        full_query = f"{conversation}Utilisateur: {message}"
    else:
        full_query = message

    result = rag.ask(full_query)
    answer = result['answer']

    if 'Pas dans les documents' in answer:
        answer = "Rien trouvé dans les documents."
    elif '---' in answer:
        answer = answer.split('---')[0].strip()
    elif 'QUESTION:' in answer:
        answer = answer.split('QUESTION:')[0].strip()

    # Sources lisibles
    sources = result.get('sources', [])
    if sources:
        answer += f"\n\n**Sources** ({len(sources)}):"
        for i, s in enumerate(sources[:5], 1):
            answer += f"\n{i}. {format_source(s)}"

    return answer

demo = gr.ChatInterface(
    fn=gradio_chat,
    title="RAG System avec Datasets locaux",
    description="Droit du travail (Syntec & Belgique), éducation québécoise",
    examples=[
        ["CDI résiliation Syntec"],
        ["Succession Belgique"],
        ["Bail logement Bruxelles"],
        ["Accord sujet verbe grammaire"],
    ],
    type="messages",   
)

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=1, reload=False)   
