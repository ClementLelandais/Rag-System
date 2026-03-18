"""
Module API Client
===============

Ce module fournit la classe `APIClient` pour interagir avec l'API
backend du système RAG (Retrieval-Augmented Generation).
"""

import requests
import os
from dotenv import load_dotenv

class APIClient:
    """
    Classe pour interagir avec l'API backend du système RAG.

    Attributes
    ----------
    url : str
        URL de l'API backend.
    """
    def __init__(self):
        load_dotenv()
        self.url = os.getenv("API_URL")

    def query(self, query: str):
        """
        Envoie une requête de type "query" à l'API backend.

        Paramètres
        ----------
        query : str
            La requête textuelle à envoyer à l'API.
            
        Retourne
        -------
        str
            La réponse textuelle de l'API.
        """
        resp = requests.post(f"{self.url}/query", json={"query": query})
        if resp.status_code == 200:
            return resp.json().get("answer", "Erreur: réponse vide.")
        else:
            return f"Erreur: {resp.text}"
    
        