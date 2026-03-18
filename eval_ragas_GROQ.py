#!/usr/bin/env python3
"""
Évaluation RAG — Groq LLM-as-a-Judge + RAGAS
=============================================
Usage :
  python eval_ragas.py --dataset data/eval_dataset.json
  python eval_ragas.py --reset-checkpoint ...
"""
import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv

sys.path.append('src')
sys.path.append('src/backend')
from backend.embedder import Embedder
from backend.vector_store import MultiVectorStore
from backend.rag import RAG
from backend.reranker import Reranker

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, AnswerCorrectness
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
GROQ_JUDGE_MODEL = "llama-3.3-70b-versatile"   # Judge : qualité max, tokens réduits
GROQ_RAGAS_MODEL = "qwen/qwen3-32b"            # RAGAS : meilleur raisonnement, quota séparé
EMBED_MODEL      = os.environ.get("HF_EMBED_MODEL", "BAAI/bge-m3")
RERANKER_MODEL   = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SLEEP_BETWEEN    = 3.0
CHECKPOINT_FILE  = "results/checkpoint.json"

# ~2500 à 1200 tokens par appel judge
JUDGE_CTX_CHARS    = 80   # chars par doc contexte  (était 150)
JUDGE_ANSWER_CHARS = 150  # chars pour la réponse   (était 300)
JUDGE_GT_CHARS     = 100  # chars pour ground truth (était 200)
JUDGE_MAX_DOCS     = 3    # nb docs contexte         (était 5)

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(enriched, judge_scores):
    Path("results").mkdir(exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"enriched": enriched, "judge_scores": judge_scores},
                  f, ensure_ascii=False, indent=2)

def load_checkpoint(total):
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        enriched = data.get("enriched",     [None] * total)
        judge_scores = data.get("judge_scores", [None] * total)

        # Remet à None les items en erreur pour les re-run
        reset_count = 0
        for i, score in enumerate(judge_scores):
            if score and score.get("justification") in ("Erreur", "Erreur judge"):
                judge_scores[i] = None
                enriched[i] = None
                reset_count += 1
        if reset_count:
            logger.warning("%d items en erreur remis à None pour être re-run", reset_count)

        done = sum(1 for e in enriched if e is not None)
        logger.info("Checkpoint trouvé : %d/%d questions déjà traitées", done, total)
        return enriched, judge_scores
    return [None] * total, [None] * total

def reset_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        logger.info("Checkpoint supprimé")

# ══════════════════════════════════════════════════════════════════════════════
# GROQ LLM-AS-A-JUDGE
# ══════════════════════════════════════════════════════════════════════════════
JUDGE_SYSTEM_PROMPT = """Tu es un évaluateur RAG expert et impartial.
Note 3 critères de 0 à 10 :
- pertinence : la réponse répond-elle à la QUESTION ?
- fidelite : tout vient-il des CONTEXTES ? (pas d'invention)
- completude : la réponse est-elle exhaustive ?

REPONDS UNIQUEMENT ce JSON :
{"pertinence": X, "fidelite": X, "completude": X, "justification": "1 phrase"}"""

JUDGE_USER_TEMPLATE = """QUESTION : {question}
CONTEXTES : {contexts}
REPONSE : {answer}
REFERENCE : {ground_truth}
JSON:"""


class GroqJudge:
    def __init__(self, model=GROQ_JUDGE_MODEL, api_key=GROQ_API_KEY, max_retries=5):
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante !")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        logger.info("GroqJudge : '%s' | contexte : %d chars/doc × %d docs",
                    model, JUDGE_CTX_CHARS, JUDGE_MAX_DOCS)

    def judge(self, question, contexts, answer, ground_truth):
        
        ctx_text = "\n".join(
            f"[{i+1}] {c[:JUDGE_CTX_CHARS]}"
            for i, c in enumerate(contexts[:JUDGE_MAX_DOCS])
        )
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question[:200],
            contexts=ctx_text,
            answer=answer[:JUDGE_ANSWER_CHARS],
            ground_truth=ground_truth[:JUDGE_GT_CHARS],
        )
        for attempt in range(self.max_retries):
            try:
                time.sleep(SLEEP_BETWEEN)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=120,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                scores = json.loads(raw)
                scores["score_global"] = round(
                    (scores["pertinence"] + 2 * scores["fidelite"] + scores["completude"]) / 4, 2
                )
                return scores
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    wait = 60 * (attempt + 1)
                    logger.warning("Rate limit Token Per Day, attente %ds...", wait)
                    time.sleep(wait)
                else:
                    logger.warning("Erreur (tentative %d): %s", attempt + 1, e)
                    time.sleep(2)

        # Lève exception : item reste None, retenté au prochain run
        raise RuntimeError(f"Judge échoué après {self.max_retries} tentatives")


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS
# ══════════════════════════════════════════════════════════════════════════════
def build_ragas_dataset(samples):
    return Dataset.from_dict({
        "question":     [s["question"]               for s in samples],
        "answer":       [s["answer"]                 for s in samples],
        "contexts":     [s["contexts"]               for s in samples],
        "ground_truth": [s.get("ground_truth", "")  for s in samples],
    })

def run_ragas_evaluation(samples, with_ground_truth=True):
    logger.info("RAGAS avec '%s' + embeddings locaux...", GROQ_RAGAS_MODEL)

    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

    langchain_llm = ChatGroq(
        model=GROQ_RAGAS_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=1024,
        n=1,
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_emb = RagasHFEmbeddings(model_name=EMBED_MODEL)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
    ]
    if with_ground_truth:
        metrics.append(AnswerCorrectness(llm=ragas_llm, embeddings=ragas_emb))

    dataset = build_ragas_dataset(samples)
    logger.info("RAGAS sur %d échantillons...", len(samples))

    return evaluate(
        dataset=dataset,
        metrics=metrics,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def load_eval_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Dataset chargé : %d questions", len(data))
    return data

def process_single_item(item, idx, total, rag, judge):
    question = item["question"]
    ground_truth = item.get("ground_truth", "")
    logger.info("[%d/%d] %s", idx + 1, total, question[:80])

    retrieved = rag.retrieve(question)

    contexts = [
        d.get("text") or d.get("chunk_text") or d.get("content") or ""
        for d in retrieved
    ]

    logger.info("TOP 5 pour '%s':", question[:30])
    for i, doc in enumerate(retrieved[:5]):
        logger.info("#%d [%s] %.3f — %s", i+1,
                    doc.get('dataset', '?'),
                    doc.get('score', 0),
                    (doc.get('text') or doc.get('chunk_text') or '')[:80])

    if rag.llm is not None:
        result = rag.ask(question)
        answer = result["answer"]
    else:
        answer = " ".join(contexts[:2])[:500]

    logger.info("%s...", answer[:60])

    judge_result = judge.judge(
        question=question, contexts=contexts,
        answer=answer, ground_truth=ground_truth,
    )
    logger.info(
        "pertinence=%.1f | fidelite=%.1f | completude=%.1f | global=%.1f",
        judge_result["pertinence"], judge_result["fidelite"],
        judge_result["completude"], judge_result["score_global"],
    )

    return {
        "question": question, "answer": answer,
        "contexts": contexts, "ground_truth": ground_truth,
    }, judge_result


def run_full_evaluation(
    dataset_path,
    output_path,
    embedder_model = "BAAI/bge-m3",
    data_dir       = "./data",
    top_k          = 5,
    num_candidates = 20,
    parallel       = False,
    reset          = False,
):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY manquante !")
    if reset:
        reset_checkpoint()

    logger.info("Initialisation pipeline RAG...")
    logger.info("Judge : %s (~%d tokens/question)", GROQ_JUDGE_MODEL,
                JUDGE_MAX_DOCS * JUDGE_CTX_CHARS // 4 + JUDGE_ANSWER_CHARS // 4 + 200)
    logger.info("   RAGAS : %s", GROQ_RAGAS_MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Embedder(embedder_model, str(device))
    vector_store = MultiVectorStore(data_dir)

    stats = vector_store.get_stats()
    logger.info("Datasets : %s | %d chunks total", stats['datasets'], stats['total_chunks'])

    for ds_name, store in vector_store.stores.items():
        sample = (store['entities'][0].get('text') or '')[:80] if store['entities'] else ''
        logger.info("'%s' : %d chunks | ex: %s", ds_name, len(store['entities']), sample)

    rag_llm  = ChatGroq(model=GROQ_JUDGE_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.0)
    reranker = Reranker(model_name=RERANKER_MODEL, device=device)
    rag = RAG(
        embedder=embedder,
        vector_store=vector_store,
        llm=rag_llm,
        reranker=reranker,
        top_k=top_k,
        num_candidates=num_candidates,
    )

    # Test retrieval au démarrage
    test_docs = rag.retrieve("bail logement Bruxelles")
    logger.info("TEST retrieval: %d docs | 1er: [%s] %s",
                len(test_docs),
                test_docs[0].get('dataset', '?') if test_docs else '?',
                (test_docs[0].get('text') or '')[:80] if test_docs else 'VIDE')

    judge = GroqJudge(model=GROQ_JUDGE_MODEL)
    eval_data = load_eval_dataset(dataset_path)
    total = len(eval_data)

    enriched, judge_scores = load_checkpoint(total)

    for i, item in enumerate(eval_data):
        if enriched[i] is not None:
            logger.info("[%d/%d] Skip", i + 1, total)
            continue
        try:
            enriched[i], judge_scores[i] = process_single_item(item, i, total, rag, judge)
        except RuntimeError as e:
            # Judge échoué : item reste None, retenté au prochain run
            logger.error("Judge échoué item %d : %s — sera retenté", i, e)
            save_checkpoint(enriched, judge_scores)
            continue
        except Exception as e:
            logger.error("Erreur pipeline item %d : %s", i, e)
            enriched[i] = {
                "question": item["question"], "answer": "",
                "contexts": [], "ground_truth": item.get("ground_truth", ""),
            }
            judge_scores[i] = {
                "pertinence": 0, "fidelite": 0, "completude": 0,
                "score_global": 0, "justification": "Erreur pipeline",
            }
        save_checkpoint(enriched, judge_scores)

    done = sum(1 for e in enriched if e is not None)
    logger.info("%d/%d questions traitées", done, total)

    # Filtre les None pour ne pas fausser les moyennes
    valid_enriched = [e for e in enriched if e is not None]
    valid_judge_scores = [s for s in judge_scores if s is not None]

    has_ground_truth = any(e.get("ground_truth") for e in valid_enriched)
    ragas_results = run_ragas_evaluation(samples=valid_enriched, with_ground_truth=has_ground_truth)

    avg_judge = {
        "pertinence": round(np.mean([s["pertinence"] for s in valid_judge_scores]), 2),
        "fidelite": round(np.mean([s["fidelite"]  for s in valid_judge_scores]), 2),
        "completude": round(np.mean([s["completude"] for s in valid_judge_scores]), 2),
        "score_global": round(np.mean([s["score_global"] for s in valid_judge_scores]), 2),
    }

    print("\n" + "=" * 60)
    print("RESULTATS D'EVALUATION RAG")
    print("=" * 60)
    print(f"\nLLM-AS-A-JUDGE [{GROQ_JUDGE_MODEL}] ({len(valid_judge_scores)}/{total}) :")
    print(f"===Pertinence=== : {avg_judge['pertinence']:.1f} / 10")
    print(f"===Fidelite=== : {avg_judge['fidelite']:.1f} / 10")
    print(f"===Completude=== : {avg_judge['completude']:.1f} / 10")
    print(f"===Score global=== : {avg_judge['score_global']:.1f} / 10")

    print(f"\nRAGAS [{GROQ_RAGAS_MODEL}] :")
    df = ragas_results.to_pandas()
    ragas_dict = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            ragas_dict[col] = float(df[col].mean())
    for metric, value in ragas_dict.items():
        print(f"   {metric:<25} : {value:.4f}")

    output = {
        "config": {
            "dataset": dataset_path,
            "embedder_model": embedder_model,
            "reranker_model": RERANKER_MODEL,
            "judge_model": GROQ_JUDGE_MODEL,
            "ragas_model": GROQ_RAGAS_MODEL,
            "top_k": top_k,
            "num_candidates": num_candidates,
            "n_questions": total,
            "n_valid": len(valid_judge_scores),
            "judge_token_budget": {
                "ctx_chars": JUDGE_CTX_CHARS,
                "answer_chars": JUDGE_ANSWER_CHARS,
                "gt_chars": JUDGE_GT_CHARS,
                "max_docs": JUDGE_MAX_DOCS,
            },
        },
        "llm_judge": {
            "averages": avg_judge,
            "per_sample": [
                {"question": valid_enriched[i]["question"], **valid_judge_scores[i]}
                for i in range(len(valid_enriched))
            ],
        },
        "ragas": {k: round(float(v), 4) for k, v in ragas_dict.items()},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Résultats exportés -> %s", output_path)
    reset_checkpoint()
    return output


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation RAG : Groq Judge + RAGAS")
    parser.add_argument("--dataset", type=str, default="data/eval_dataset.json")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    parser.add_argument("--embedder", type=str, default="BAAI/bge-m3")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--reranker-model", type=str, default=RERANKER_MODEL)
    parser.add_argument("--judge-model", type=str, default=GROQ_JUDGE_MODEL,
                        help="Modèle Groq pour le judge (défaut: llama-3.3-70b-versatile)")
    parser.add_argument("--ragas-model", type=str, default=GROQ_RAGAS_MODEL,
                        help="Modèle Groq pour RAGAS (défaut: qwen/qwen3-32b)")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--reset-checkpoint", action="store_true")
    args = parser.parse_args()

    GROQ_JUDGE_MODEL = args.judge_model
    GROQ_RAGAS_MODEL = args.ragas_model
    RERANKER_MODEL = args.reranker_model

    run_full_evaluation(
        dataset_path = args.dataset,
        output_path = args.output,
        embedder_model = args.embedder,
        data_dir = args.data_dir,
        top_k = args.top_k,
        num_candidates = args.num_candidates,
        parallel = not args.no_parallel,
        reset = args.reset_checkpoint,
    )
