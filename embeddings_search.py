# embeddings_search.py
from sentence_transformers import SentenceTransformer, util
import torch

# 1) Load model (first run downloads model files)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2) Example dataset (you can change these sentences)
sentences = [
    "I love learning AI",
    "Machine learning is amazing",
    "The sky is blue",
    "AI helps build smart apps",
    "I enjoy solving coding problems"
]

# 3) Encode dataset into embeddings (vector form)
embeddings = model.encode(sentences, convert_to_tensor=True)

# 4) The query you want to search
query = "AI is awesome and useful"
query_emb = model.encode(query, convert_to_tensor=True)

# 5) Cosine similarity between query and each dataset sentence
cosine_scores = util.cos_sim(query_emb, embeddings)  # shape: [1, N]

# 6) Sort and print results (highest similarity first)
scores = cosine_scores[0].cpu().numpy()
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print("Query:", query)
print("\nTop matches:")
for idx, score in ranked:
    print(f"{score:.4f}  →  {sentences[idx]}")
