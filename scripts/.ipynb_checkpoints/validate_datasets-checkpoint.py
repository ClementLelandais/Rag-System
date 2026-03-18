#!/usr/bin/env python3
"""Validation des datasets RAG"""
import json, faiss, numpy as np, sys
from pathlib import Path

DATA_DIR = Path("./data")
ERRORS   = []
WARNINGS = []

def ok(msg):  print(f"{msg}")
def err(msg): print(f"{msg}"); ERRORS.append(msg)
def warn(msg):print(f"{msg}"); WARNINGS.append(msg)

# ── 1. Structure dossiers ─────────────────────────────────────────────────────
print("\nSTRUCTURE DOSSIERS")
for folder in ["index", "chunks", "raw"]:
    p = DATA_DIR / folder
    if p.exists(): 
        ok(f"data/{folder}/ présent")
    else:      
        err(f"data/{folder}/ MANQUANT")

# ── 2. Cohérence par dataset ──────────────────────────────────────────────────
print("\n📊 DATASETS")
faiss_files = list((DATA_DIR / "index").glob("*.faiss"))

if not faiss_files:
    err("Aucun fichier .faiss trouvé dans data/index/")
else:
    for faiss_path in faiss_files:
        name = faiss_path.stem
        print(f"\n[{name}]")

        # FAISS
        try:
            index = faiss.read_index(str(faiss_path))
            ok(f"FAISS chargé : {index.ntotal} vecteurs, dim={index.d}")
        except Exception as e:
            err(f"FAISS illisible : {e}"); continue

        # Chunks
        chunks_path = DATA_DIR / "chunks" / f"{name}.json"
        if not chunks_path.exists():
            err(f"chunks/{name}.json MANQUANT"); continue

        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        ok(f"Chunks chargés : {len(chunks)} entrées")

        # Cohérence taille
        if index.ntotal != len(chunks):
            err(f"Taille incohérente : {index.ntotal} vecteurs vs {len(chunks)} chunks")
        else:
            ok(f"Tailles cohérentes ({index.ntotal})")

        # Clés obligatoires
        required_keys = ["id", "source_id", "text"]
        sample = chunks[0]
        missing = [k for k in required_keys if k not in sample]
        if missing: 
            err(f"Clés manquantes dans chunks : {missing}")
        else:      
            ok(f"Clés obligatoires présentes : {required_keys}")

        # Textes vides
        empty = sum(1 for c in chunks if not c.get("text", "").strip())
        if empty > 0: 
            warn(f"{empty} chunks avec texte vide")
        else:      
            ok("Aucun chunk vide")

        # Longueur moyenne
        avg_len = sum(len(c.get("text","")) for c in chunks) / len(chunks)
        ok(f"Longueur moyenne des textes : {avg_len:.0f} chars")
        if avg_len < 50:  warn("Textes très courts — vérifier le contenu")
        if avg_len > 2000: warn("Textes très longs — peut dégrader le retrieval")

        # Raw optionnel
        raw_path = DATA_DIR / "raw" / f"{name}.json"
        if raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                raw = json.load(f)
            ok(f"Raw chargé : {len(raw)} entrées")
        else:
            warn(f"raw/{name}.json absent (optionnel)")

# ── 3. Test retrieval rapide ──────────────────────────────────────────────────
print("\nTEST RETRIEVAL")
try:
    sys.path.append('src')
    from backend.vector_store import MultiVectorStore
    from backend.embedder import Embedder
    import torch

    vs = MultiVectorStore(str(DATA_DIR))
    stats = vs.get_stats()
    ok(f"MultiVectorStore : {stats['loaded']} datasets, {stats['total_chunks']} chunks")

    embedder = Embedder("BAAI/bge-m3", "cpu", debug=False)
    q_emb = embedder.embed("heures supplémentaires droit belge")
    results  = vs.search(q_emb, top_k=3)

    if results:
        ok(f"Retrieval OK : {len(results)} résultats")
        for i, r in enumerate(results):
            print(f"DOC{i+1} [{r.get('dataset','?')}] : {r.get('text','')[:80]}...")
    else:
        err("Retrieval retourne 0 résultats")

except Exception as e:
    err(f"Erreur retrieval : {e}")

# ── 4. Résumé ─────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print(f{len(ERRORS) == 0 and 'TOUT OK' or ''}")
print(f{len(ERRORS)} erreur(s)")
print(f"{len(WARNINGS)} avertissement(s)")
if ERRORS:
    print("\nErreurs à corriger :")
    for e in ERRORS: print(f"  - {e}")
