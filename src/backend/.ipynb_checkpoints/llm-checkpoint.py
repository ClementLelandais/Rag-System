"""
Module LLM (Large Language Model)
===============

Ce module fournit la classe `LLM` pour interagir avec des modèles de langage
de grande taille (LLM) afin de générer des réponses contextuelles basées sur
les documents fournis.
"""

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import torch
from collections import OrderedDict
import os

class LLM:
    """
    Classe pour interagir avec un modèle de langage de grande taille (LLM).

    Attributes
    ----------
    model_name : str
        Nom du modèle pré-entraîné Hugging Face à utiliser.
    device : torch.device
        Dispositif (CPU ou GPU) sur lequel le modèle est exécuté.
    top_k : int
        Nombre de documents contextuels à utiliser pour générer la réponse.
    model : transformers.AutoModel
        Modèle pré-entraîné pour générer des réponses basées sur le contexte.
    tokenizer : transformers.AutoTokenizer
        Tokenizer associé au modèle pré-entraîné.
    """

    def __init__(
        self, 
        model_name: str, 
        device: torch.device,
        top_k: int = 5,
    ):
        """
        Initialise le tokenizer et le modèle pré-entraîné.

        Paramètres
        ----------
        model_name : str
            Nom du modèle pré-entraîné Hugging Face à utiliser.
        device : torch.device
            Dispositif (CPU ou GPU) sur lequel exécuter le modèle.
        top_k : int
            Nombre de documents contextuels à utiliser pour générer la réponse.
        """

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k

        # Charger le LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = os.getenv("HF_TOKEN"), trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if "cuda" in str(self.device) else None
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.model.eval()

    def build_context(self, results):
        """
        Construit un contexte textuel à partir des résultats de recherche.
        
        Paramètres
        ----------
        results : list of dict
            Liste des résultats de recherche contenant les documents pertinents.

        Retourne
        -------
        context : str
            Contexte textuel construit à partir des documents.
        """
        docs = OrderedDict()

        for r in results:
            uuid = r['doc_uuid']

            if uuid not in docs:
                docs[uuid] = {
                    "title": r["title"],
                    "topic": r["topic"],
                    "content": (r["answer_text"] or r["chunk_text"])[:1200] + "…"
                }

            if len(docs) >= self.top_k:
                break

        context = ""
        for d in docs.values():
            context += f"[{d['title']}]\n{d['content']}\n\n"
        
        return context.strip(), docs
    
    def generate(self, prompt, context=None, max_tokens=None):
        """Génération universelle - compatible TOUS modèles"""
        
        # Prompt complet
        full_prompt = f"{prompt}\n{context}" if context else prompt
        
        try:
            # Tokenization 
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            # Inputs avec attention_mask
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.device)
            
            # Génération 
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100 if max_tokens is None else max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False  # Déterministe
                )
            
            # Décode la partie générée
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = full_text[len(full_prompt):].strip()
            
            return generated if generated else full_text
            
        except Exception as e:
            print(f"LLM generate error: {e}")
            return "Erreur génération."

