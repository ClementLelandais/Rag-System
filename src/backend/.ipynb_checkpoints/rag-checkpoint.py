"""
Module RAG (Retrieval-Augmented Generation)
===============

Ce module fournit la classe `RAG` pour implémenter un système de
Retrieval-Augmented Generation (RAG) combinant un embedder, une base de
vecteurs, et un modèle de langage de grande taille (LLM) pour répondre
aux requêtes en s'appuyant sur des documents pertinents récupérés.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RAG:
    """
    Classe pour le système de Retrieval-Augmented Generation (RAG).

    Attributes
    ----------
    embedder : Embedder
        Instance de la classe Embedder pour générer des embeddings de texte.
    vector_store : VectorStore
        Instance de la classe VectorStore pour la recherche de documents.
    llm : LLM
        Instance de la classe LLM pour générer des réponses basées sur le contexte.
    reranker : Reranker, optional
        Instance de la classe Reranker pour réordonner les documents récupérés.
    top_k : int
        Nombre de documents contextuels à utiliser pour générer la réponse.
    num_candidates : int
        Nombre de documents candidats à récupérer avant le réordonnancement.
        Doit être > top_k pour que le reranker ait un effet.
    dtype : str
        Type de données pour les embeddings (par exemple 'float32').
    """

    def __init__(
        self,
        embedder,
        vector_store,
        llm,
        reranker=None,
        top_k: int = 5,
        num_candidates: int = 20,   # FIX : était 5 = top_k → reranker sans effet
        dtype: str = "float32",
    ):
        self.embedder       = embedder
        self.vector_store   = vector_store
        self.llm            = llm
        self.reranker       = reranker
        self.top_k          = top_k
        self.num_candidates = max(num_candidates, top_k * 3)  # garantit num_candidates > top_k
        self.dtype = "float32"
        if hasattr(self.llm, 'top_k'):
            self.llm.top_k = top_k

        if self.reranker is None:
            logger.warning(
                "RAG initialisé sans reranker — "
                "le ranking se basera uniquement sur la similarité cosinus FAISS."
            )
        else:
            logger.info(
                "RAG initialisé avec reranker (%s) | top_k=%d | num_candidates=%d",
                getattr(self.reranker, 'model_name', type(self.reranker).__name__),
                self.top_k,
                self.num_candidates,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # RETRIEVE
    # ──────────────────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> list:
        q_emb = self.embedder.embed(query)
        
        raw_docs = self.vector_store.search(q_emb, self.num_candidates)
        
        results = []
        if self.reranker is not None:
            try:
                results = self.reranker.apply(query, raw_docs)
            except Exception as e:
                print(f"Reranker failed: {e}, fallback to raw_docs")
                results = raw_docs
        else:
            results = raw_docs
        
        return results[:self.top_k]


    # ──────────────────────────────────────────────────────────────────────────
    # ASK
    # ──────────────────────────────────────────────────────────────────────────
    def ask(self, query: str) -> dict:
        top_docs = self.retrieve(query)

        context_parts = []
        for doc in top_docs:
            context_parts.append(doc.get('text') or doc.get('chunk_text') or '')
        context = "\n---\n".join(context_parts)

        prompt = f"""Tu es un assistant juridique et éducatif. Réponds à la question en te basant sur les documents fournis.

        Instructions :
        - Réponds en français de manière fluide et naturelle.
        - Ne cite JAMAIS les sources dans ta réponse. Aucune référence, aucun numéro, aucun nom de document.
        - Synthétise directement les informations.
        
        QUESTION: {query}
        
        DOCUMENTS:
        {context}
        
        RÉPONSE:"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
        except Exception:
            response = self.llm.invoke(prompt)
        answer   = response.content if hasattr(response, 'content') else str(response)

        return {
            "answer":  answer.strip()[:2000],
            "sources": [doc.get('title', 'Document sans titre') for doc in top_docs],
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXEMPLE D'INITIALISATION — à adapter dans ton main.py / app.py
# ══════════════════════════════════════════════════════════════════════════════
#
# import torch
# from reranker import Reranker
# from rag import RAG
#
# reranker = Reranker(
#     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# )
#
# rag = RAG(
#     embedder=embedder,
#     vector_store=vector_store,
#     llm=llm,
#     reranker=reranker,       
#     top_k=5,
#     num_candidates=20,       
# )