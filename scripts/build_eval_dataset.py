#!/usr/bin/env python3
"""
Convertisseur datasets -> format eval_ragas.py
==============================================

Lit les datasets existants (bsard.csv, alloprof.json, syntec.csv)
et génère un fichier eval_dataset.json prêt à l'emploi.

Usage :
    python build_eval_dataset.py
    python build_eval_dataset.py --max 50 # Limite à 50 questions
    python build_eval_dataset.py --source bsard # Un seul dataset
"""

import json
import argparse
import pandas as pd
from pathlib import Path

# ── Chemins par défaut (adaptez si nécessaire) ────────────────────────────────
DATA_DIR = Path("./data")
CHUNKS_DIR = DATA_DIR / "chunks"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "eval_dataset.json"


# ══════════════════════════════════════════════════════════════════════════════
# CONVERTISSEURS PAR SOURCE
# ══════════════════════════════════════════════════════════════════════════════

def build_chunk_index(chunks_path: Path) -> dict:
    """
    Construit un index chunk_id → content depuis un fichier chunks JSON.
    Utilisé pour retrouver les contextes de référence via should_retrieve.

    Structure attendue :
      [{"id": "..._chunk_0", "content": "...", "source_id": "...", "chunk_index": 0}, ...]
    """
    if not chunks_path.exists():
        return {}

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Index par chunk_index (position dans la liste = identifiant entier)
    index = {}
    for i, chunk in enumerate(chunks):
        index[i] = chunk.get("content", "")

        # Index aussi par source_id pour retrouver par UUID
        source_id = chunk.get("source_id", "")
        if source_id not in index:
            index[source_id] = []
        index[source_id].append(chunk.get("content", ""))

    print(f"Index chunks : {len(chunks)} chunks chargés")
    return index, chunks


def convert_bsard(csv_path: Path, chunk_index: dict, chunks: list, max_items: int) -> list:
    """
    Convertit bsard.csv en format eval_ragas.

    Colonnes utilisées :
      - question : question
      - answer : ground_truth
      - should_retrieve : liste d'indices de chunks de référence (optionnel)
    """
    df = pd.read_csv(csv_path).head(max_items)
    samples = []

    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("answer", "")).strip()

        # Récupération des contextes de référence via should_retrieve
        reference_contexts = []
        should_retrieve = row.get("should_retrieve", "")
        if pd.notna(should_retrieve) and str(should_retrieve).strip():
            try:
                indices = json.loads(str(should_retrieve).replace("'", '"'))
                for idx in indices:
                    if isinstance(idx, int) and idx < len(chunks):
                        content = chunks[idx].get("content", "")
                        if content:
                            reference_contexts.append(content[:600])
            except Exception:
                pass  # should_retrieve mal formé, on ignore

        if question and ground_truth:
            samples.append({
                "question":           question,
                "ground_truth":       ground_truth,
                "reference_contexts": reference_contexts,
                "source":             "bsard",
            })

    print(f"BSARD : {len(samples)} questions converties")
    return samples


def convert_alloprof(json_path: Path, chunks_path: Path, max_items: int) -> list:
    """
    Convertit alloprof.json → format eval_ragas.

    Structure alloprof.json (raw) :
      [{"uuid": "...", "cleaned_text": "...", "answer": "...", "title": "..."}, ...]

    On génère des paires question/réponse à partir des champs disponibles.
    Si le champ 'question' n'existe pas, on utilise le titre comme question
    et le cleaned_text/answer comme ground_truth.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Charge les chunks alloprof pour les contextes de référence
    chunk_lookup = {}
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            alloprof_chunks = json.load(f)
        for chunk in alloprof_chunks:
            src = str(chunk.get("source_id", ""))
            chunk_lookup.setdefault(src, []).append(chunk.get("content", ""))

    samples = []
    for item in data[:max_items]:
        uuid = str(item.get("uuid", ""))

        # Champ question (peut s'appeler question, title, titlr...)
        question = (
            item.get("question") or
            item.get("title") or
            item.get("titlr") or
            ""
        ).strip()

        # Ground truth = réponse de référence
        ground_truth = (
            item.get("cleaned_text") or
            item.get("answer") or
            ""
        ).strip()

        # Contextes de référence = chunks issus du même document
        reference_contexts = chunk_lookup.get(uuid, [])[:3]
        reference_contexts = [c[:600] for c in reference_contexts]

        if question and ground_truth:
            samples.append({
                "question": question,
                "ground_truth": ground_truth[:800],
                "reference_contexts": reference_contexts,
                "source": "alloprof",
            })

    print(f"Alloprof : {len(samples)} questions converties")
    return samples


def convert_syntec(csv_path: Path, max_items: int) -> list:
    """
    Convertit syntec.csv → format eval_ragas.
    Adaptez les noms de colonnes si nécessaire.
    """
    df = pd.read_csv(csv_path).head(max_items)
    samples = []

    # Détection automatique des colonnes question/réponse
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "question" in col_lower or "query" in col_lower:
            col_map.setdefault("question", col)
        if "answer" in col_lower or "reponse" in col_lower or "réponse" in col_lower:
            col_map.setdefault("ground_truth", col)

    if "question" not in col_map or "ground_truth" not in col_map:
        print(f"Colonnes non détectées dans syntec.csv. Colonnes disponibles : {list(df.columns)}")
        print("Éditez convert_syntec() pour mapper manuellement.")
        return []

    for _, row in df.iterrows():
        question = str(row[col_map["question"]]).strip()
        ground_truth = str(row[col_map["ground_truth"]]).strip()
        if question and ground_truth:
            samples.append({
                "question":           question,
                "ground_truth":       ground_truth,
                "reference_contexts": [],
                "source":             "syntec",
            })

    print(f"SYNTEC : {len(samples)} questions converties")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def build_eval_dataset(source_filter: str = None, max_items: int = 100):
    """
    Construit le dataset d'évaluation en agrégeant toutes les sources disponibles.
    """
    all_samples = []

    # ── BSARD ─────────────────────────────────────────────────────────────────
    bsard_csv = RAW_DIR / "bsard.csv"
    bsard_chunks = CHUNKS_DIR / "bsard.json"

    if (source_filter is None or source_filter == "bsard") and bsard_csv.exists():
        print("\nChargement BSARD...")
        if bsard_chunks.exists():
            chunk_index, chunks = build_chunk_index(bsard_chunks)
        else:
            chunk_index, chunks = {}, []
            print("Pas de chunks BSARD — contextes de référence vides")
        all_samples += convert_bsard(bsard_csv, chunk_index, chunks, max_items)

    # ── ALLOPROF ──────────────────────────────────────────────────────────────
    alloprof_raw = RAW_DIR / "alloprof.json"
    alloprof_chunks = CHUNKS_DIR / "alloprof.json"

    if (source_filter is None or source_filter == "alloprof") and alloprof_raw.exists():
        print("\nChargement Alloprof...")
        all_samples += convert_alloprof(alloprof_raw, alloprof_chunks, max_items)

    # ── SYNTEC ────────────────────────────────────────────────────────────────
    syntec_csv = RAW_DIR / "syntec.csv"

    if (source_filter is None or source_filter == "syntec") and syntec_csv.exists():
        print("\nChargement SYNTEC...")
        all_samples += convert_syntec(syntec_csv, max_items)

    if not all_samples:
        print("\nAucun dataset trouvé. Vérifiez les chemins dans DATA_DIR.")
        return

    # ── Export ────────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\nDataset généré : {OUTPUT_PATH}")
    print(f"Total : {len(all_samples)} questions")

    # Résumé par source
    from collections import Counter
    counts = Counter(s["source"] for s in all_samples)
    for src, n in counts.items():
        has_ctx = sum(1 for s in all_samples if s["source"] == src and s["reference_contexts"])
        print(f"{src:<12} : {n} questions ({has_ctx} avec contextes de référence)")

    print(f"\nPrêt pour : python eval_ragas.py --dataset {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertit les datasets en format eval_ragas")
    parser.add_argument("--max",    type=int, default=100,  help="Nombre max de questions par source")
    parser.add_argument("--source", type=str, default=None, help="Source unique : bsard | alloprof | syntec")
    args = parser.parse_args()

    build_eval_dataset(source_filter=args.source, max_items=args.max)
