#!/usr/bin/env python3
"""
Évaluation RAG — Mistral LLM-as-a-Judge + RAGAS
================================================

Deux niveaux d'évaluation :
  1. LLM-as-a-Judge via Mistral API  →  scores custom (pertinence, fidélité, complétude)
  2. RAGAS                           →  métriques avec ET sans ground truth

Métriques RAGAS sans ground truth :
  - faithfulness         : la réponse est-elle fidèle aux contextes ?
  - answer_relevancy     : la réponse répond-elle à la question ?
  - context_precision    : les contextes sont-ils précis ?

Métriques RAGAS avec ground truth :
  - answer_correctness   : la réponse est-elle correcte par rapport à la vérité terrain ?
  - answer_similarity    : similarité sémantique avec la ground truth

Structure attendue du dataset annoté (JSON) :
  [
    {
      "question": "...",
      "ground_truth": "...",         ← réponse de référence
      "reference_contexts": ["..."]  ← optionnel, contextes de référence
    },
    ...
  ]

Usage :
  python eval_ragas.py --dataset data/eval_dataset.json --output results/eval_results.json
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

# ── Compatibilité imports projet ──────────────────────────────────────────────
sys.path.append('src')
sys.path.append('src/backend')

from backend.embedder import Embedder
from backend.vector_store import MultiVectorStore
from backend.rag import RAG

# ── RAGAS ─────────────────────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness,
    answer_similarity,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. MISTRAL LLM-AS-A-JUDGE
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """Tu es un évaluateur expert de systèmes RAG (Retrieval-Augmented Generation).
Tu dois évaluer objectivement la qualité des réponses générées selon trois critères précis.
Réponds UNIQUEMENT avec un objet JSON valide, sans markdown ni explication."""

JUDGE_USER_TEMPLATE = """Évalue la réponse RAG suivante sur 3 critères (score de 0 à 10) :

QUESTION : {question}

CONTEXTES RÉCUPÉRÉS :
{contexts}

RÉPONSE GÉNÉRÉE :
{answer}

GROUND TRUTH :
{ground_truth}

Retourne exactement ce JSON (et rien d'autre) :
{{
  "pertinence": <0-10>,       // La réponse répond-elle bien à la question ?
  "fidelite": <0-10>,         // La réponse est-elle fidèle aux contextes fournis (pas d'hallucination) ?
  "completude": <0-10>,       // La réponse couvre-t-elle les éléments clés de la ground truth ?
  "justification": "<1 phrase courte>"
}}"""


class MistralJudge:
    """
    Évaluateur LLM-as-a-Judge basé sur l'API Mistral.
    Utilise mistral-large pour des jugements fiables.
    """

    def __init__(self, model: str = "mistral-large-latest", max_retries: int = 3):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("Variable MISTRAL_API_KEY manquante dans .env")
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def judge(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """
        Envoie une requête de jugement à Mistral et parse la réponse JSON.

        Retourne un dict avec les clés :
          pertinence, fidelite, completude, score_global, justification
        """
        ctx_text = "\n---\n".join(
            f"[DOC{i+1}] {c[:400]}" for i, c in enumerate(contexts[:5])
        )
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question,
            contexts=ctx_text,
            answer=answer[:600],
            ground_truth=ground_truth[:400],
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,   # Déterministe pour la reproductibilité
                    max_tokens=256,
                )
                raw = response.choices[0].message.content.strip()

                # Nettoyage si le modèle entoure de ```json ... ```
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                scores = json.loads(raw)

                # Score global = moyenne pondérée (fidélité compte double)
                scores["score_global"] = round(
                    (scores["pertinence"] + 2 * scores["fidelite"] + scores["completude"]) / 4,
                    2,
                )
                return scores

            except json.JSONDecodeError as e:
                logger.warning(f"JSON invalide (tentative {attempt+1}): {e}")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Erreur API (tentative {attempt+1}): {e}")
                time.sleep(2)

        # Fallback si toutes les tentatives échouent
        logger.error("Échec du jugement après %d tentatives", self.max_retries)
        return {
            "pertinence": 0, "fidelite": 0, "completude": 0,
            "score_global": 0, "justification": "Erreur API",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. ÉVALUATION RAGAS
# ══════════════════════════════════════════════════════════════════════════════

def build_ragas_dataset(samples: list[dict]) -> Dataset:
    """
    Convertit la liste d'échantillons enrichis en Dataset HuggingFace
    compatible RAGAS.

    Chaque sample doit contenir :
      question, answer, contexts, ground_truth (optionnel)
    """
    data = {
        "question": [s["question"]     for s in samples],
        "answer": [s["answer"]       for s in samples],
        "contexts": [s["contexts"]     for s in samples],
        "ground_truth": [s.get("ground_truth", "") for s in samples],
    }
    return Dataset.from_dict(data)


def run_ragas_evaluation(
    samples: list[dict],
    mistral_api_key: str,
    with_ground_truth: bool = True,
) -> dict:
    """
    Lance l'évaluation RAGAS avec Mistral comme LLM et embedder.

    Métriques :
      - Sans ground truth : faithfulness, answer_relevancy, context_precision
      - Avec ground truth  : + answer_correctness, answer_similarity
    """
    logger.info("Configuration RAGAS avec Mistral...")

    # LLM et embeddings Mistral pour RAGAS
    langchain_llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=mistral_api_key,
        temperature=0.0,
    )
    langchain_emb = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=mistral_api_key,
    )

    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_emb = LangchainEmbeddingsWrapper(langchain_emb)

    # Sélection des métriques
    metrics = [faithfulness, answer_relevancy, context_precision]
    if with_ground_truth:
        metrics += [answer_correctness, answer_similarity]

    # Injection LLM/embeddings dans chaque métrique
    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    dataset = build_ragas_dataset(samples)

    logger.info("Lancement évaluation RAGAS sur %d échantillons...", len(samples))
    results = evaluate(dataset=dataset, metrics=metrics)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def load_eval_dataset(path: str) -> list[dict]:
    """
    Charge le dataset d'évaluation annoté depuis un fichier JSON.

    Format attendu :
      [{"question": "...", "ground_truth": "...", "reference_contexts": [...]}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Dataset chargé : %d questions", len(data))
    return data


def run_full_evaluation(
    dataset_path: str,
    output_path: str,
    embedder_model: str = "BAAI/bge-m3",
    data_dir: str = "./data",
    top_k: int = 5,
):
    """
    Pipeline d'évaluation complète :
      1. Chargement du dataset annoté
      2. Génération des réponses via le pipeline RAG
      3. Jugement via Mistral LLM-as-a-Judge
      4. Évaluation RAGAS (avec et sans ground truth)
      5. Export JSON des résultats
    """
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise EnvironmentError("Variable MISTRAL_API_KEY manquante dans .env")

    # ── Chargement composants RAG ──────────────────────────────────────────
    logger.info("Initialisation du pipeline RAG...")
    embedder = Embedder(embedder_model, "cpu")
    vector_store = MultiVectorStore(data_dir)
    rag = RAG(embedder, vector_store, llm=None, top_k=top_k)

    judge = MistralJudge()

    eval_data = load_eval_dataset(dataset_path)
    enriched = []   # Données pour RAGAS
    judge_scores = []   # Scores LLM-as-a-Judge

    # ── Boucle d'évaluation ───────────────────────────────────────────────
    for i, item in enumerate(eval_data):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        logger.info("\n[%d/%d] Question : %s", i + 1, len(eval_data), question)

        # 1. Retrieval
        retrieved = rag.retrieve(question)
        contexts = [d["chunk_text"] for d in retrieved]

        # 2. Génération de la réponse (si LLM disponible dans le RAG)
        if rag.llm is not None:
            result = rag.ask(question)
            answer = result["answer"]
        else:
            # Fallback : concaténation des chunks les plus pertinents
            answer = " ".join(contexts[:2])[:500]
            logger.warning("Pas de LLM dans RAG — réponse = concaténation des chunks")

        logger.info("Réponse : %s...", answer[:80])

        # 3. LLM-as-a-Judge
        judge_result = judge.judge(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
        )
        judge_scores.append(judge_result)
        logger.info(
            "  🏛️  Judge → pertinence=%.1f | fidélité=%.1f | complétude=%.1f | global=%.1f",
            judge_result["pertinence"],
            judge_result["fidelite"],
            judge_result["completude"],
            judge_result["score_global"],
        )

        # 4. Collecte pour RAGAS
        enriched.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })

    # ── RAGAS ─────────────────────────────────────────────────────────────
    has_ground_truth = any(e["ground_truth"] for e in enriched)
    ragas_results = run_ragas_evaluation(
        samples=enriched,
        mistral_api_key=mistral_api_key,
        with_ground_truth=has_ground_truth,
    )

    # ── Agrégation scores LLM-Judge ───────────────────────────────────────
    avg_judge = {
        "pertinence": round(np.mean([s["pertinence"]   for s in judge_scores]), 2),
        "fidelite": round(np.mean([s["fidelite"]     for s in judge_scores]), 2),
        "completude": round(np.mean([s["completude"]   for s in judge_scores]), 2),
        "score_global": round(np.mean([s["score_global"] for s in judge_scores]), 2),
    }

    # ── Rapport final ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("RÉSULTATS D'ÉVALUATION RAG")
    print("═" * 60)

    print("\nLLM-AS-A-JUDGE (Mistral) :")
    print(f"Pertinence   : {avg_judge['pertinence']:.1f} / 10")
    print(f"Fidélité     : {avg_judge['fidelite']:.1f} / 10")
    print(f"Complétude   : {avg_judge['completude']:.1f} / 10")
    print(f"Score global : {avg_judge['score_global']:.1f} / 10")

    print("\nRAGAS :")
    ragas_dict = ragas_results.to_pandas().mean().to_dict() if hasattr(ragas_results, "to_pandas") else dict(ragas_results)
    for metric, value in ragas_dict.items():
        print(f"{metric:<25} : {float(value):.4f}")

    # ── Export JSON ────────────────────────────────────────────────────────
    output = {
        "config": {
            "dataset": dataset_path,
            "embedder_model": embedder_model,
            "judge_model": "mistral-large-latest",
            "top_k": top_k,
            "n_questions": len(eval_data),
        },
        "llm_judge": {
            "averages": avg_judge,
            "per_sample": [
                {"question": enriched[i]["question"], **judge_scores[i]}
                for i in range(len(enriched))
            ],
        },
        "ragas": {k: round(float(v), 4) for k, v in ragas_dict.items()},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("\nRésultats exportés → %s", output_path)
    return output


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation RAG : Mistral Judge + RAGAS")
    parser.add_argument(
        "--dataset", type=str, default="data/eval_dataset.json",
        help="Chemin vers le dataset annoté (JSON)",
    )
    parser.add_argument(
        "--output", type=str, default="results/eval_results.json",
        help="Fichier de sortie des résultats",
    )
    parser.add_argument(
        "--embedder", type=str, default="BAAI/bge-m3",
        help="Modèle d'embedding (HuggingFace)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Dossier racine des données (index FAISS, chunks...)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Nombre de documents récupérés par le RAG",
    )
    args = parser.parse_args()

    run_full_evaluation(
        dataset_path = args.dataset,
        output_path = args.output,
        embedder_model= args.embedder,
        data_dir = args.data_dir,
        top_k = args.top_k,
    )
