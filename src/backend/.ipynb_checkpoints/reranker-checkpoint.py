"""
Module Reranker
===============
Réordonne les résultats de recherche avec un cross-encoder.
Compatible transformers >= 4.50 (text= au lieu de texts=).
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class Reranker:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device     = device
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        # AutoModelForSequenceClassification : retourne .logits directement
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
        ).to(self.device)
        self.model.eval()

    def apply(self, query: str, results: list, max_length: int = 512) -> list:
        if not results:
            return results

        pairs = [
            (query, (r.get('text') or r.get('chunk_text') or r.get('content') or '')[:1500])
            for r in results
        ]

        # FIX : text= au lieu de texts= (compatible transformers >= 4.50)
        tokens = self.tokenizer(
            text=[p[0] for p in pairs],
            text_pair=[p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**tokens).logits

        # cross-encoder ms-marco retourne 1 logit par paire = score de pertinence
        scores = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, 1]

        for r, s in zip(results, scores):
            r['rerank_score'] = float(s)

        return sorted(results, key=lambda r: r['rerank_score'], reverse=True)