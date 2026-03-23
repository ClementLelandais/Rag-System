"""
RAG Backend FastAPI - Version SINGLETON (anti-double-chargement)
"""
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import torch
from huggingface_hub import login

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
    GROQ_LLM_MODEL  = os.getenv("LLM_MODEL",       "llama-3.1-8b-instant")
    RERANKER_MODEL  = os.getenv("RERANKER_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device détecté: {DEVICE}")

    print("Loading Embedder...")
    embedder = Embedder(model_name=EMBEDDING_MODEL, device=DEVICE)

    print("Loading MultiVectorStore...")
    vector_store = MultiVectorStore(datasets_path="./data")
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
        'embedder':     embedder,
        'vector_store': vector_store,
        'llm':          llm,
        'reranker':     reranker,
        'rag':          rag,
        'device':       DEVICE,
        'llm_model':    GROQ_LLM_MODEL,
    }
    print("[SINGLETON] TOUS composants chargés !")
else:
    print("[SINGLETON] Composants déjà chargés - skip")


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_source(s: str) -> str:
    return (s.replace("bsard_doc_",  "BSARD #")
             .replace("syntec_doc_", "Syntec #")
             .replace("alloprof_",   "Alloprof #"))

def ask_rag(query: str, history: list = None) -> tuple[str, list]:
    """Interroge le RAG et retourne (answer, sources)."""
    rag = RAG_COMPONENTS['rag']

    # Contexte conversationnel (3 derniers échanges)
    full_query = query
    if history:
        conversation = ""
        for msg in history[-6:]:  # 3 échanges = 6 messages
            role    = msg.get("role", "")
            content = msg.get("content", "")
            # Retire les sources de l'historique
            content_clean = content.split("\n\n**Sources**")[0]
            if role == "user":
                conversation += f"Utilisateur: {content_clean}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content_clean}\n"
        full_query = f"{conversation}\nUtilisateur: {query}"

    result  = RAG_COMPONENTS['rag'].ask(full_query)
    answer  = result['answer']
    sources = result.get('sources', [])

    if 'Pas dans les documents' in answer:
        answer = "Rien trouvé dans les documents."
    if '---' in answer:
        answer = answer.split('---')[0].strip()
    if 'QUESTION:' in answer:
        answer = answer.split('QUESTION:')[0].strip()

    return answer, sources


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Acollab API", version="1.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    answer, sources = ask_rag(req.query)
    return QueryResponse(
        answer=answer,
        sources=[format_source(s) for s in sources],
    )

@app.get("/")
def read_root():
    return {
        "RAG LIVE":  True,
        "docs":      "/docs",
        "device":    RAG_COMPONENTS['device'],
        "llm_model": RAG_COMPONENTS['llm_model'],
        "reranker":  getattr(RAG_COMPONENTS['reranker'], 'model_name', 'N/A'),
        "datasets":  RAG_COMPONENTS['vector_store'].get_stats()['datasets'],
    }

@app.get("/health")
def health_check():
    stats = RAG_COMPONENTS['vector_store'].get_stats()
    return {
        "status":          "healthy",
        "device":          RAG_COMPONENTS['device'],
        "reranker_active": RAG_COMPONENTS['reranker'] is not None,
        "datasets":        stats['datasets'],
        "total_chunks":    stats['total_chunks'],
    }

@app.post("/debug_search")
async def debug_search(request: dict):
    query = request["query"]
    rag   = RAG_COMPONENTS['rag']
    docs  = rag.retrieve(query)
    return {
        "query":           query,
        "reranker_active": rag.reranker is not None,
        "num_candidates":  rag.num_candidates,
        "top_k":           rag.top_k,
        "found_chunks":    len(docs),
        "top_chunks": [
            {
                "rank":         i + 1,
                "dataset":      doc.get("dataset", "?"),
                "rerank_score": doc.get("rerank_score", "N/A"),
                "score":        doc.get("score", "N/A"),
                "content":      (doc.get("text") or doc.get("chunk_text") or "")[:200],
                "title":        doc.get("title", ""),
            }
            for i, doc in enumerate(docs[:5])
        ],
    }


# ── Gradio avec sidebar historique ───────────────────────────────────────────
import gradio as gr

def _init_state():
    return {"conversations": {"Chat 1": []}, "current": "Chat 1", "next_idx": 2}

def _choices(state):
    return list(state["conversations"].keys())

def new_chat(state):
    name = f"Chat {state['next_idx']}"
    state["conversations"][name] = []
    state["current"]  = name
    state["next_idx"] += 1
    return state, gr.update(value=name, choices=_choices(state)), []

def load_chat(selected, state):
    if selected and selected in state["conversations"]:
        state["current"] = selected
    return state, state["conversations"][state["current"]]

def clear_current(state):
    state["conversations"][state["current"]] = []
    return state, []

def delete_current(state):
    cur = state["current"]
    if cur in state["conversations"]:
        del state["conversations"][cur]
    if not state["conversations"]:
        state = _init_state()
    else:
        state["current"] = _choices(state)[0]
    return (
        state,
        gr.update(value=state["current"], choices=_choices(state)),
        state["conversations"][state["current"]],
    )

def send_message(user_message, state):
    if not user_message.strip():
        return "", state["conversations"][state["current"]], state

    cur     = state["current"]
    history = state["conversations"][cur]

    # Appel RAG avec historique
    answer, sources = ask_rag(user_message, history)

    # Ajoute les sources à la réponse
    if sources:
        answer += "\n\n**Sources :**"
        for i, s in enumerate(sources[:5], 1):
            answer += f"\n{i}. {format_source(s)}"

    # Sauvegarde dans l'historique
    state["conversations"][cur].append({"role": "user",      "content": user_message})
    state["conversations"][cur].append({"role": "assistant", "content": answer})

    return "", state["conversations"][cur], state


with gr.Blocks(title="Akelio - RAG System", fill_height=True) as demo:
    initial_state = _init_state()
    st = gr.State(initial_state)

    gr.Markdown("## 🤖 Akelio - RAG System")

    with gr.Sidebar():
        gr.Markdown("### 💬 Conversations")
        btn_new = gr.Button("➕ Nouveau chat", variant="primary")
        chat_list = gr.Radio(
            label="Historique",
            choices=_choices(initial_state),
            value="Chat 1",
            interactive=True,
        )
        gr.Markdown("---")
        btn_clear  = gr.Button("🗑️ Vider ce chat",     variant="secondary")
        btn_delete = gr.Button("❌ Supprimer ce chat",  variant="stop")

    chatbot = gr.Chatbot(
        label="Chatbot",
        height=500,
        value=[],
        type="messages",
        show_copy_button=True,
    )
    msg = gr.Textbox(
        placeholder="Pose ta question…",
        show_label=False,
        autofocus=True,
    )

    # Interactions
    msg.submit(send_message,    [msg, st],           [msg, chatbot, st])
    btn_new.click(new_chat,     st,                  [st, chat_list, chatbot])
    chat_list.change(load_chat, [chat_list, st],     [st, chatbot])
    btn_clear.click(clear_current,  st,              [st, chatbot])
    btn_delete.click(delete_current, st,             [st, chat_list, chatbot])

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=1, reload=False)
