"""
Module Vector Store
===============

Ce module fournit la classe `VectorStore` pour gérer le stockage et la récupération
d'embeddings vectoriels à l'aide de la bibliothèque FAISS.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Classe pour gérer le stockage et la récupération d'embeddings vectoriels.

    Attributes
    ----------
    index_path : str
        Chemin vers le fichier de l'index FAISS.    
    entites : list of dict
        Liste des entités documentaires associées aux embeddings.
    index : faiss.Index
        Index FAISS pour la recherche d'embeddings
    """

    def __init__(self, index_path: str):
        """
        Initialise le VectorStore avec le chemin de l'index FAISS.

        Paramètres
        ----------
        index_path : str
            Chemin vers le fichier de l'index FAISS.
        """
        self.index_path = index_path
        self.entites = []
        self.index = None

    def add(self, embeddings: np.ndarray, metadata: list):
        """
        Ajoute des embeddings et leurs métadonnées à l'index FAISS.

        Paramètres
        ----------
        embeddings : numpy.ndarray
            Matrice d'embeddings de forme (n_samples, dimension).
        metadata : list of dict
            Liste des métadonnées associées (même ordre que embeddings).
        """
        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
        
        # Ajout batch FAISS
        self.index.add(embeddings)
        
        # Ajout métadonnées
        self.entites.extend(metadata)
        
        print(f"Ajouté {len(metadata)} documents")

    def save(self):
        """
        Sauvegarde l'index FAISS sur le disque.
        """
        if self.index is None:
            raise ValueError("Index vide, rien à sauvegarder")
        
        faiss.write_index(self.index, self.index_path)
        print(f"Index sauvegardé : {self.index_path} ({self.index.ntotal} vecteurs)")

    def load(self, raw_path: str, chunks_path: str, meta_path: str, index_path: str):
        """
        Charge les entités documentaires et l'index FAISS depuis les fichiers.

        Paramètres
        ----------
        raw_path : str
            Chemin vers le fichier JSON contenant les documents bruts.
        chunks_path : str
            Chemin vers le fichier JSON contenant les chunks de texte.
        meta_path : str
            Chemin vers le fichier JSON contenant les métadonnées des chunks.
        index_path : str    
            Chemin vers le fichier de l'index FAISS.
        """
        
        with open(raw_path, 'r', encoding='utf-8') as f:
            answers = json.load(f)

        answers_map = {}
        for obj in answers:
            uuid = str(obj.get('uuid'))

            answers_map[uuid] = {
                "text": obj.get('cleaned_text') or obj.get('answer'),
                "title": obj.get('titlr') or "Document sans titre",
                "topic": obj.get('topic') or "unknow", 
            }

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_raw = json.load(f)
        chunks_list = [c['content'] for c in chunks_raw]

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_raw = json.load(f)

        entites = []
        for idx, chunk in enumerate(chunks_list):
            metadata = meta_raw[idx]
            uuid = str(metadata['source_id'])

            doc_info = answers_map.get(uuid, {})

            entites.append({
                "doc_uuid": uuid,
                "chunk_text": chunk,
                "answer_text": doc_info.get("text"),
                "title": doc_info.get("title"),
                "topic": doc_info.get("topic"),
                "chunk_id": idx  
            })

        self.entites = entites
        self._load_faiss(index_path)

    def _load_faiss(self, index_path: str):
        """
        Charge l'index FAISS depuis le fichier.
        
        Paramètres
        ----------
        index_path : str
            Chemin vers le fichier de l'index FAISS.
        """
        try:
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        except:
            self.index = faiss.read_index(index_path)
        print(f"Index chargé : {self.index.ntotal} vecteurs, dim={self.index.d}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> tuple:
        """
        Recherche les k embeddings les plus similaires.

        Paramètres
        ----------
        query_embedding : numpy.ndarray
            Embedding de la requête (1, dimension).
        k : int
            Nombre de résultats à retourner.

        Retourne
        -------
        tuple: (distances, indices, résultats)
            Distances L2, indices FAISS, entités correspondantes.
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(
            query_embedding, k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.entities):
                entity = self.entities[idx].copy()
                entity['distance'] = float(distances[0][i])
                results.append(entity)
        return results    
        
    def get_stats(self) -> dict:
        """
        Retourne des statistiques sur le vector store.
        
        Retourne
        -------
        dict: Statistiques (nb docs, dimension, etc.)
        """
        if self.index is None:
            return {"error": "Index non chargé"}
        return {
            "total_entities": len(self.entites),
            "index_size": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": self.index.__class__.__name__
        }

class MultiVectorStore:
    """VectorStore multi-datasets (scan auto ./data/index/*.faiss)"""
    
    def __init__(self, datasets_path: str = "./data"):
        self.datasets_path = Path(datasets_path)
        self.stores = {}
        self.load_all_datasets()

    def _load_raw_flexible(self, raw_path, max_items):
        """JSON ou CSV : liste dicts uniformes"""
        import pandas as pd
        
        if raw_path.suffix == '.json':
            with open(raw_path, 'r', encoding='utf-8') as f:
                return json.load(f)[:max_items]
        else:  # CSV
            df = pd.read_csv(raw_path)
            return df.to_dict('records')[:max_items]

    def load_all_datasets(self):
        """Charge TOUS datasets ./data/index/*.faiss"""
        index_dir = self.datasets_path / "index"
        if not index_dir.exists():
            print("Dossier index manquant:", index_dir)
            return
            
        for index_path in index_dir.glob("*.faiss"):
            ds_name = index_path.stem
            print(f"Chargement {ds_name}...")
            
            # 1. CHUNKS obligatoires (JSON)
            chunks_path = self.datasets_path / "chunks" / f"{ds_name}.json"
            if not chunks_path.exists():
                print(f"Skip {ds_name}: pas de chunks")
                continue
                
            with open(chunks_path, 'r', encoding='utf-8') as f:
                entities = json.load(f)
            
            # 2. RAW optionnel (JSON ou CSV)
            raw_path = self.datasets_path / "raw" / f"{ds_name}.json"
            if not raw_path.exists():
                raw_path = self.datasets_path / "raw" / f"{ds_name}.csv"
                
            if raw_path.exists():
                raw_data = self._load_raw_flexible(raw_path, len(entities))
                for i, entity in enumerate(entities):
                    if i < len(raw_data):
                        entity.update({
                            'title': raw_data[i].get('title', raw_data[i].get('titlr', f"{ds_name}_doc_{i}")),
                            'answer_text': raw_data[i].get('cleaned_text', raw_data[i].get('answer', ''))
                        })
            
            # 3. META optionnel (JSON)
            meta_path = self.datasets_path / "meta" / f"{ds_name}.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_list = json.load(f)
                    for i, entity in enumerate(entities):
                        if i < len(meta_list):
                            entity['source_id'] = meta_list[i].get('source_id', f"{ds_name}_{i}")
                except Exception as e:
                    print(f"Meta {ds_name} ignoré:", e)
            
            # 4. FALLBACK titres/source_id
            for i, entity in enumerate(entities):
                entity.setdefault('title', f"{ds_name}_doc_{i}")
                entity.setdefault('source_id', f"{ds_name}_{i}")
            
            # 5. FAISS index
            try:
                self.stores[ds_name] = {
                    'index': faiss.read_index(str(index_path)),
                    'entities': entities
                }
                print(f"{ds_name}: {len(entities)} chunks")
            except Exception as e:
                print(f"Erreur index {ds_name}:", e)
            for ds_name, store in self.stores.items():
                sample = store['entities'][0]
                logger.info("Dataset '%s' : %d chunks | ex: %s", 
                            ds_name, len(store['entities']), str(sample.get('text',''))[:80])

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = np.ascontiguousarray(query_emb.reshape(1, -1))
    
        all_results = []
        
        n_datasets = len(self.stores)
        for ds_name, store in self.stores.items():
            try:
                k_per_ds = min(top_k, len(store['entities']))
                distances, indices = store['index'].search(query_emb, k_per_ds)
                for i, idx in enumerate(indices[0]):
                    if idx < len(store['entities']):
                        entity = store['entities'][idx].copy()
                        entity['dataset'] = ds_name
                        entity['score']   = float(-distances[0][i])
                        all_results.append(entity)
            except Exception as e:
                print(f"Erreur search {ds_name}:", e)

        return sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_k * n_datasets]
    
    def get_stats(self):
        """Stats tous datasets"""
        total = sum(len(store['entities']) for store in self.stores.values())
        return {
            "datasets": list(self.stores.keys()),
            "total_chunks": total,
            "loaded": len(self.stores)
        }