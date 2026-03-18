#  Akelio - Projet RAG

Dans le cadre du Master 1 en Intelligence Artificielle, nous sommes amenés à réaliser un projet en partenariat avec une entreprise externe. Cette entreprise dispose d’une application de gestion de projets (*Acollab*) permettant à ses utilisateurs de communiquer, s’organiser, stocker des documents et collaborer efficacement en équipe.

Dans ce contexte, **Akelio** souhaite explorer la faisabilité de l’intégration d’un système intelligent basé sur le RAG (Retrieval-Augmented Generation), afin d’améliorer les capacités de recherche d’information et d’offrir une assistance plus pertinente et contextuelle aux utilisateurs de la plateforme.

## Acteurs du projet

### Membres du groupe:

- **[Thanina Benmammar](mailto:thanina.bemmammar.etu@univ-lemans.fr)**
- **[Fares Khleifi](mailto:fares.khleifi.etu@univ-lemans.fr)**
- **[Kilian Pousse](mailto:kilian.pousse.etu@univ-lemans.fr)**
- **[Clément Lelandais](mailto:clement.lelandais.etu@univ-lemans.fr)**

### Tuteurs de projet

- **[Nicolas Dugué](mailto:nicolas.dugue@univ-lemans.fr)** - *enseignant*
- **[Guillaume Louvel](mailto:louvel@akelio.fr)** - *client*

## Arborescence
```
RAG-System/
 ├─── backend/
 │    ├─── embedder.py      # Module d'embeddings des documents
 │    ├─── llm.py           # Module sur le Language Model
 │    ├─── main.py          # Main du Backend
 │    ├─── rag.py           # Regroupement de tous les modules
 │    ├─── reranker.py      # Réevaluation des rangs 
 │    └─── vector_store.py  # Module du vector 
 ├─── data/                 # Données utilisées pour la démo
 ├─── docs/
 │    ├─── build/           # Resultat du build de la documentation
 │    └─── source/          # Dossier source de la documentation
 └─── frontend
      ├─── api_client.py    # Module qui gère les requêtes API 
      └─── app.py           # Interface utilisateur
```

## Lancer le projet

Afin de lancer le projet, il vous surffit de copier coller le `.env.example` en `.env`. Les instructions à suivre sont dans le fichier concerné.

Ensuite il vous voudra installer python 3.10 ainsi que les packages présents dans le `requirements.txt`. Dans ce dernier, vous y retrouverez tous les instructions à suivre.

### Lancer l'interface (frontend):
```bash
python -m frontend
```

### Lancer le backend:
```bash
python -m backend [<option>]
```

#### Liste des options:
- **`--host <host>`:** Adresse d’écoute du serveur (par défaut : valeur définie dans le fichier `.env`, sinon `0.0.0.0`).

- **`--port <port>`:** Port du serveur (par défaut : valeur définie dans le fichier `.env`, sinon `8000`).

- **`--dev`:** Active le mode développeur (par défaut : `False`). Ce mode active notamment le rechargement automatique du serveur.

### Documentation
La documentation complète du projet Système RAG se trouve dans le dossier `docs/`. Elle est organisée en plusieurs pages:
- Une page principale (`index.html`)
- Une page pour le backend et une page pour le frontend
- Une page par module

#### Génération de la documentation
Afin de générer la documentation, il est nécessaire d’avoir préalablement les packages installés:
```bash
pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints
pip install fastapi transformers torch gradio faiss-cpu pydantic python-dotenv
```

Ensuite, pour générer la doc:
```bash
cd docs
make html
```

Les pages HTML générées se trouvent dans `docs/build/html`.
Ouvrez `docs/build/html/index.html` dans votre navigateur pour consulter la documentation.

## **Commit**
Les commits doivent suivre [ConventionnalCommits](https://www.conventionalcommits.org/en/v1.0.0/) et doivent être écrits en minuscule et en utilisant la convention suivante:
```
<type>([<scope>]): <description>
```
Les types de commit sont les suivants :
- feat: Ajout d'une nouvelle fonctionnalité
- fix: Correction d'un bug
- docs: Modification de la documentation
- style: Modification du style du code
- refactor: Modification du code sans changer son comportement
- perf: Modification du code pour améliorer les performances
- test: Ajout de tests ou modification des tests existants
- chore: Modification du build system ou des dépendances
- ci: Modification des fichiers de configuration du CI
- revert: Revert d'un commit précédent
- wip: Work in progress
- merge: Merge d'une branche
- release: Release d'une version
- hotfix: Correction d'un bug critique
- other: Autre
- init: Initialisation du projet