import os
import sys

# Ajouter le projet au path
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'Système RAG'
author = '2026, C. Lelandais'
copyright = '2026, Acollab, 2026, C. Lelandais'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',          # Génère la doc depuis les docstrings
    'sphinx.ext.napoleon',         # Support des docstrings style Google / NumPy
    'sphinx_autodoc_typehints',    # Affiche automatiquement les types
    'sphinx.ext.viewcode'          # Ajoute un lien vers le code source
]

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'  # pour la doc en français

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Thème moderne, similaire à Read the Docs
html_static_path = ['_static']

# Mocker les imports pour éviter les erreurs si les packages ne sont pas installés
autodoc_mock_imports = [
    "fastapi", 
    "transformers", 
    "gradio", 
    "torch"
]
