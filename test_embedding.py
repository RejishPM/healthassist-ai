from app.embedding.factory import get_embedder

embedder = get_embedder()

vector = embedder.get_embedding(
    "Metformin is commonly used for Type 2 diabetes."
)

print("Embedding dimensions:", len(vector))
print("First 5 values:", vector[:5])