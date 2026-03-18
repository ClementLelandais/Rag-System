import pytrec_eval
import json, sys
sys.path.append('src')
from backend.embedder import Embedder
from backend.vector_store import MultiVectorStore

# Charge dataset
with open('data/eval_dataset.json') as f:
    dataset = json.load(f)

embedder = Embedder('BAAI/bge-m3', 'cpu')
vector_store = MultiVectorStore('./data')

# Construit qrels et run
qrels = {}
run = {}

for item in dataset[:50]:   
    qid = item['question'][:50]  
    
    # Docs pertinents
    qrels[qid] = {doc_id: 1 for doc_id in item.get('should_retrieve', [])}
    
    # Retrieval
    import numpy as np
    q_emb = np.ascontiguousarray(embedder.embed(item['question']), dtype=np.float32)
    results = vector_store.search(q_emb, 10)
    
    # Scores retrieval
    run[qid] = {
        r.get('id', f"doc_{i}"): float(-r.get('score', 0))
        for i, r in enumerate(results)
    }

# Évaluation
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {'map', 'ndcg', 'recall_5', 'P_5'}
)
results = evaluator.evaluate(run)

# Moyennes
for metric in ['map', 'ndcg', 'recall_5', 'P_5']:
    avg = sum(r[metric] for r in results.values()) / len(results)
    print(f"{metric}: {avg:.4f}")