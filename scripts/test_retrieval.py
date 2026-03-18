import faiss, numpy as np, json
from transformers import AutoTokenizer, AutoModel
import torch

device = "cuda"
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Test query
query = "diabète insulinémie"
inputs = tokenizer([query], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    query_emb = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)


# Recherche
index = faiss.read_index("data/index/alloprof.faiss")
D, I = index.search(query_emb.astype('float32'), k=5)
print("Top 5 résultats:")

with open("data/chunks/alloprof/chunks.json") as f:
    chunks = json.load(f)
for i, idx in enumerate(I[0]):
    print(f"{i+1}. {chunks[idx]['text'][:100]}... (score: {D[0][i]:.3f})")
