#!/usr/bin/env python3
"""Réindexation complète des 3 datasets"""
import json, faiss, numpy as np, torch, sys
from pathlib import Path
from tqdm import tqdm

sys.path.append('src')
from backend.embedder import Embedder

DATA_DIR = Path("./data")
EMBED_MODEL = "BAAI/bge-m3"
BATCH_SIZE  = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

embedder = Embedder(EMBED_MODEL, str(device), batch_size=BATCH_SIZE, debug=False)

# Vérifie que l'embedder fonctionne
test_emb = embedder.embed("test")
print(f"Test embed: shape={test_emb.shape}, norme={np.linalg.norm(test_emb):.4f}")
assert np.linalg.norm(test_emb) > 0.1, "Embedder retourne des zéros !"

def build_index(name):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    
    chunks_path = DATA_DIR / "chunks" / f"{name}.json"
    if not chunks_path.exists():
        print(f"{chunks_path} introuvable"); return
    
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"{len(chunks)} chunks chargés")
    
    texts = [c.get("text", c.get("chunk_text", "")) for c in chunks]
    
    # Vérifie textes non vides
    empty = sum(1 for t in texts if not t.strip())
    print(f"{'Attention' if empty else 'Valide'} Textes vides: {empty}/{len(texts)}")
    
    # Génère embeddings batch par batch avec vérification
    all_embeddings = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Embedding {name}"):
        batch = texts[start:start + BATCH_SIZE]
        emb = embedder.embed(batch)
        
        
        if torch.is_tensor(emb):
            emb = emb.cpu().numpy()
        emb = np.array(emb, dtype=np.float32)
        
        # Vérifie normes
        normes = np.linalg.norm(emb, axis=1)
        if normes.mean() < 0.1:
            print(f"Batch {start}: normes anormalement basses ({normes.mean():.4f})")
        
        all_embeddings.append(emb)
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings: shape={embeddings.shape}")
    print(f"Norme moyenne: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"Min/Max: {embeddings.min():.4f} / {embeddings.max():.4f}")
    
    # Construit index FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    out_path = DATA_DIR / "index" / f"{name}.faiss"
    faiss.write_index(index, str(out_path))
    print(f"Index sauvegardé: {out_path} ({index.ntotal} vecteurs)")
    
    # Vérifie l'index en recherchant
    q = embeddings[0:1].copy()
    D, I = index.search(q, 3)
    print(f"Test recherche: distances={D[0]}, indices={I[0]}")
    assert I[0][0] == 0, "Le premier résultat devrait être lui-même !"
    print(f"Index valide !")

# Reconstruit les 3 datasets
for name in ["syntec", "alloprof", "bsard"]:
    build_index(name)

print("\nRéindexation terminée !")
