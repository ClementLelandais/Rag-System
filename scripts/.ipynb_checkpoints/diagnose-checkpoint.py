#!/usr/bin/env python3
"""
diagnose_rag.py — Diagnostic local du retriever RAG depuis le checkpoint ou le dataset.
Usage :
  python diagnose_rag.py # checkpoint par défaut
  python diagnose_rag.py --checkpoint results/checkpoint.json --n 10
  python diagnose_rag.py --checkpoint data/eval_dataset.json  # dataset brut
"""
import json
import argparse
import textwrap
import random
from pathlib import Path


def load_checkpoint(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format 1 : checkpoint enrichi  {"enriched": [...]}
    if isinstance(data, dict) and "enriched" in data:
        raw = data["enriched"]
    # Format 2 : dataset brut [{...}, ...]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"Format non reconnu : {list(data.keys())}")

    samples = [e for e in raw if e is not None and e.get("contexts")]
    if not samples:
        samples = [e for e in raw if e is not None and e.get("question")]
        print(f"Aucun contexte trouvé — questions seulement ({len(samples)} samples)\n")
    else:
        print(f"{len(samples)} samples avec contextes chargés\n")
    return samples


def analyze_sample(s: dict, idx: int):
    question = s.get("question", "")
    answer = s.get("answer", "")
    contexts = s.get("contexts", [])
    ground_truth = s.get("ground_truth", "")

    print("=" * 70)
    print(f"SAMPLE n°{idx}")
    print("=" * 70)
    print(f"\nQUESTION :\n {question}")
    print(f"\nRÉPONSE ({len(answer)} chars) :\n {textwrap.shorten(answer, 200)}")
    if ground_truth:
        print(f"\nGROUND TRUTH :\n {textwrap.shorten(ground_truth, 200)}")

    print(f"\nCONTEXTES RÉCUPÉRÉS ({len(contexts)} chunks) :")
    q_words = set(question.lower().split())

    for i, ctx in enumerate(contexts):
        ctx_str = str(ctx)
        n_chars = len(ctx_str)
        n_words = len(ctx_str.split())
        preview = textwrap.shorten(ctx_str, 150)
        ctx_words = set(ctx_str.lower().split())
        overlap = q_words & ctx_words
        overlap_pct = len(overlap) / max(len(q_words), 1) * 100

        print(f"\n  [{i+1}] {n_chars} chars / {n_words} mots — "
              f"overlap question : {overlap_pct:.0f}% ({len(overlap)} mots communs)")
        print(f" Mots communs : {', '.join(list(overlap)[:8])}")
        print(f" Aperçu : {preview}")

    print("\nSIGNAUX :")
    if not contexts:
        print("Aucun contexte récupéré !")
    else:
        sizes = [len(str(c)) for c in contexts]
        if max(sizes) > 3000:
            print(f"Chunks très longs (max {max(sizes)} chars)")
        if min(sizes) < 100:
            print(f"Chunks très courts (min {min(sizes)} chars)")

        answer_words  = set(answer.lower().split())
        ctx_all_words = set(" ".join(str(c) for c in contexts).lower().split())
        coverage = len(answer_words & ctx_all_words) / max(len(answer_words), 1) * 100
        if coverage < 40:
            print(f{coverage:.0f}% des mots de la réponse dans les contextes → risque hallucination")
        else:
            print(f"{coverage:.0f}% des mots de la réponse couverts par les contextes")

        overlaps = [len(q_words & set(str(c).lower().split())) for c in contexts]
        best_pos = overlaps.index(max(overlaps)) + 1
        if best_pos > 1:
            print(f"Meilleur chunk en position {best_pos} (pas en position 1) → ranking KO")
        else:
            print(f"Meilleur chunk en position 1")
    print()


def global_stats(samples: list):
    print("\n" + "=" * 70)
    print("STATISTIQUES GLOBALES")
    print("=" * 70)

    if not samples:
        print("Aucun sample à analyser")
        return

    n_contexts = [len(s.get("contexts", [])) for s in samples]
    ctx_sizes  = [len(str(c)) for s in samples for c in s.get("contexts", [])]

    print(f"\nNombre de contextes par sample :")
    print(f"min={min(n_contexts)}  max={max(n_contexts)}  "
          f"moy={sum(n_contexts)/len(n_contexts):.1f}")

    if ctx_sizes:
        print(f"\nTaille des chunks (chars) :")
        print(f"min={min(ctx_sizes)}  max={max(ctx_sizes)}  "
              f"moy={sum(ctx_sizes)/len(ctx_sizes):.0f}")
        over_3k = sum(1 for s in ctx_sizes if s > 3000)
        print(f"chunks > 3000 chars : {over_3k}/{len(ctx_sizes)} "
              f"({over_3k/len(ctx_sizes)*100:.0f}%)")

    best_not_first = 0
    for s in samples:
        contexts = s.get("contexts", [])
        question = s.get("question", "")
        if not contexts:
            continue
        q_words  = set(question.lower().split())
        overlaps = [len(q_words & set(str(c).lower().split())) for c in contexts]
        if overlaps and overlaps.index(max(overlaps)) > 0:
            best_not_first += 1

    print(f"\nRanking du retriever :")
    print(f"Meilleur chunk PAS en position 1 : "
          f"{best_not_first}/{len(samples)} ({best_not_first/len(samples)*100:.0f}%)")
    if best_not_first / len(samples) > 0.5:
        print("Problème de ranking confirmé : un reranker résoudrait ça")
    else:
        print("Ranking correct")

    # Taux de réponses "je ne trouve pas"
    refusals = sum(
        1 for s in samples
        if "ne trouve pas" in s.get("answer", "").lower()
        or "not found" in s.get("answer", "").lower()
    )
    if refusals:
        print(f"\nRéponses 'je ne trouve pas' : {refusals}/{len(samples)} "
              f"({refusals/len(samples)*100:.0f}%) : prompt trop restrictif ou retriever KO")
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnostic RAG")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoint.json")
    parser.add_argument("--n", type=int, default=5,
                        help="Nombre de samples à afficher (défaut: 5)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Index d'un sample spécifique")
    parser.add_argument("--stats-only",  action="store_true",
                        help="Stats globales uniquement")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Fichier non trouvé : {args.checkpoint}")
        return

    samples = load_checkpoint(args.checkpoint)
    global_stats(samples)

    if not args.stats_only:
        if args.sample is not None:
            analyze_sample(samples[args.sample], args.sample)
        else:
            chosen = random.sample(samples, min(args.n, len(samples)))
            for i, s in enumerate(chosen):
                analyze_sample(s, i)


if __name__ == "__main__":
    main()