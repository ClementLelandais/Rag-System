"""
Module Embedder
===============

Ce module fournit la classe `Embedder` pour générer des embeddings de texte
à l'aide de modèles pré-entraînés de la bibliothèque Transformers de Hugging Face.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional  

class Embedder:
    """
    Classe pour générer des embeddings de texte avec des modèles pré-entraînés.

    Cette classe utilise un modèle de Transformers pour encoder du texte en vecteurs
    numériques (embeddings), et normalise ces vecteurs pour faciliter les comparaisons
    (cosine similarity, recherche de nearest neighbors, etc.).

    Attributes
    ----------
    model_name : str
        Nom du modèle pré-entraîné Hugging Face à utiliser.
    device : torch.device
        Dispositif (CPU ou GPU) sur lequel le modèle est exécuté.
    tokenizer : transformers.AutoTokenizer
        Tokenizer associé au modèle pré-entraîné.
    model : transformers.AutoModel
        Modèle pré-entraîné pour générer les embeddings.
    """

    def __init__(self, model_name: str, device: torch.device,
                 embedding_dtype: str = "float32", debug: bool = True,
                 batch_size: int = 128):
        
        self.embedding_dtype = "float32"        
        self.debug = debug
        self.batch_size = batch_size
        
        # FORCE GPU si disponible
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
            torch_dtype = torch.float16
            device_map = "auto"
        else:
            self.device = torch.device("cpu")
            print("GPU indisponible : CPU")
            torch_dtype = torch.float32
            device_map = None
        
        if self.debug:
            print("=" * 60)
            print(f"[DEBUG] Modèle: {model_name}")
            print(f"[DEBUG] Device: {self.device}")
            print(f"[DEBUG] Precision: {torch_dtype}")
            print("=" * 60)
        
        # CHARGEMENT UNIQUE DU MODÈLE
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        ).eval()
        
        if self.debug:
            print(f"[DEBUG] Modèle chargé en {time.time() - t0:.2f}s")
        
        self.documents = []
    
            
    def embed(self, texts):
        """Embed texts → numpy array NORMALISÉ"""
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)  # Évite division par 0
            
        return embeddings
        
    def _get_document_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes.

        Args:
            texts (list[str]) : Liste de textes à encoder.

        Returns:
            np.ndarray : Matrice numpy des embeddings (shape : [n_texts, embedding_dim]).
        """
        all_embeddings = []
        total = len(texts)
        num_batches = (total + self.batch_size - 1) // self.batch_size

        if self.debug:
            print("\n[DEBUG] ===== Début de la génération des embeddings =====")
            print(f"[DEBUG] Nombre total de textes : {total}")
            print(f"[DEBUG] Nombre de batches : {num_batches}, batch_size={self.batch_size}")

        for start_idx in range(0, total, self.batch_size):
            batch_texts = texts[start_idx:start_idx + self.batch_size]
            batch_num = start_idx // self.batch_size + 1
            progress = (start_idx + len(batch_texts)) / total * 100

            lengths = [len(t.split()) for t in batch_texts]
            avg_length = sum(lengths) / len(lengths)

            if self.debug:
                print(f"\n[DEBUG] Batch {batch_num}/{num_batches} - {len(batch_texts)} docs ({progress:.2f}% du total)")
                print(f"[DEBUG] Longueur moyenne des textes : {avg_length:.1f} mots")
                print(f"[DEBUG] Exemple premier texte : {batch_texts[0][:80]}{'...' if len(batch_texts[0]) > 80 else ''}")

            t1 = time.time()
            inputs = self.tokenizer(self.documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 
            token_time = time.time() - t1

            if self.debug:
                print(f"[DEBUG] Tokenization terminée en {token_time:.2f}s")
                print(f"[DEBUG] input_ids shape : {inputs['input_ids'].shape}, attention_mask shape : {inputs['attention_mask'].shape}")

            t2 = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            model_time = time.time() - t2

            if self.debug:
                print(f"[DEBUG] Passage modèle OK en {model_time:.2f}s")
                print(f"[DEBUG] Embeddings batch shape : {embeddings.shape}")

            all_embeddings.append(embeddings.cpu().numpy())

        embeddings_np = np.vstack(all_embeddings)

        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / (norms + 1e-8)
        
        if self.debug:
            print(f"[DEBUG] Embeddings NORMALISÉS : norms: {np.linalg.norm(embeddings_np, axis=1)[:3].round(2)}")
        
        return embeddings_np

    def add_documents(self, documents: list[str] | str):
        """
        Ajoute un ou plusieurs documents à la liste.

        Args:
            documents (list[str] | str) : Document(s) à ajouter.
        """
        if isinstance(documents, str):
            documents = [documents]
        self.documents.extend(documents)
        if self.debug:
            print(f"[DEBUG] +{len(documents)} document(s) ajouté(s). Total actuel : {len(self.documents)}")

    def clear_documents(self):
        """
        Vide la liste des documents.
        """
        count = len(self.documents)
        self.documents = []
        if self.debug:
            print(f"[DEBUG] {count} document(s) supprimé(s). La liste est maintenant vide.")

    def get_embeddings(self) -> np.ndarray:
        """
        Retourne les embeddings de tous les documents ajoutés.

        Returns:
            np.ndarray : Matrice des embeddings.
        """
        if not self.documents:
            if self.debug:
                print("[DEBUG] Aucun document à encoder — arrêt.")
            return np.array([], dtype=self.embedding_dtype)

        if self.debug:
            print("\n[DEBUG] ===== Lancement de get_embeddings() =====")
            print(f"[DEBUG] Documents à encoder : {len(self.documents)}")

        embeddings = self._get_document_embeddings(self.documents)
        if self.debug:
            print(f"[DEBUG] Génération des embeddings pour {len(self.documents)} documents")
            print("[DEBUG] ===== Fin de get_embeddings() =====\n")

        return embeddings

    def load(self, path: str):
        """
        Charge des documents depuis un fichier texte.

        Args:
            path (str) : Chemin vers le fichier texte.
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            self.add_documents(content)
        if self.debug:
            print(f"[DEBUG] Document chargé depuis {path}")

    def generate_in_file(self, path: str):
        """
        Génére et sauvegarde les embeddings dans un fichier .npy batch par batch.
        Chaque batch est écrit dans le fichier sans charger tous les embeddings en mémoire.
        """
        if not self.documents:
            if self.debug:
                print("[DEBUG] Aucun document à encoder : arrêt.")
            return

        total = len(self.documents)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        if self.debug:
            print(f"[DEBUG] Sauvegarde par stream des embeddings vers '{path}' ({total} docs, {num_batches} batches)")

        # Ouvrir un fichier .npy pour écriture "appendable" via memmap
        first_batch = True
        for start_idx in range(0, total, self.batch_size):
            batch_texts = self.documents[start_idx:start_idx + self.batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Ecriture batch
            if first_batch:
                # Création du fichier .npy avec la taille exacte pour le total
                fp = np.lib.format.open_memmap(path, mode='w', dtype=self.embedding_dtype,
                                               shape=(total, embeddings.shape[1]))
                first_batch = False
            fp[start_idx:start_idx + embeddings.shape[0]] = embeddings

            if self.debug:
                print(f"[DEBUG] Batch {start_idx // self.batch_size + 1}/{num_batches} écrit dans le fichier.")

        if self.debug:
            print(f"[DEBUG] Tous les embeddings ont été sauvegardés dans '{path}'")

    def search(self, query: str, k: int = 5):
        """
        Recherche les k documents les plus proches du query.
        Args:
            query (str) : Texte de la requête
            k (int) : nombre de résultats à retourner
        Returns:
            results (list[str]), distances (list[float])
        """
        if not self.documents:
            return [], []

        # Générer l'embedding de la query
        query_emb = self._get_document_embeddings([query])
        doc_embs = self.get_embeddings()

        # Calculer similarité cosinus
        sims = cosine_similarity(query_emb, doc_embs)[0]
        top_k_idx = sims.argsort()[::-1][:k]

        results = [self.documents[i] for i in top_k_idx]
        distances = [sims[i] for i in top_k_idx]

        return results, distances

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts et retourne embeddings normalisés"""
        if isinstance(texts, str):
            texts = [texts]
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        
        embeddings = model.encode(
            texts, 
            normalize_embeddings=True,  
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
    
    def encode_similarity(self, text1: str, text2: str) -> float:
        """Similarité cosinus entre 2 textes"""
        emb1 = self.encode([text1])
        emb2 = self.encode([text2])
        return np.dot(emb1[0], emb2[0])

