import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# =========================
# Paths
# =========================
input_path = Path(
    r"D:\Users\moels\PycharmProjects\chat_1\backend\Data\cleaned_data\unified_legal_schema_final.json"
)

faiss_index_path = Path(
    r"\backend\Data\vector_db\faiss_index_muffakir.bin"
)

metadata_path = Path(
    r"\backend\Data\vector_db\metadata_muffakir.pkl"
)

faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

# =========================
# Load Data
# =========================
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📄 Loaded {len(data)} legal chunks")

# =========================
# Load Embedding Model
# =========================
model = SentenceTransformer(
    "mohamed2811/Muffakir_Embedding",
    device="cpu"   # غيريها cuda لو عندك GPU
)

print("✅ Muffakir Embedding model loaded")

vectors = []
metadata_list = []

# =========================
# Embedding Loop
# =========================
for item in data:
    text = item.get("text", "").strip()
    if not text:
        continue

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True   # 🔥 مهم جدًا
    )

    vectors.append(embedding.astype(np.float32))

    metadata_list.append({
        "uid": item.get("uid"),
        "text": text,                     # النص نفسه
        "doc_type": item.get("doc_type"),
        "law_name": item.get("law_name"),
        "article_number": item.get("article_number"),
        "text_role": item.get("text_role"),
        "topic": item.get("topic"),
        "source_file": item.get("source_file")
    })

# =========================
# Safety Check
# =========================
assert len(vectors) == len(metadata_list), "❌ Vectors / metadata mismatch"

vectors = np.vstack(vectors)

# =========================
# Build FAISS Index (Cosine Similarity)
# =========================
dim = vectors.shape[1]

index = faiss.IndexFlatIP(dim)   # 👈 Inner Product = Cosine (after normalization)
index.add(vectors)

print(f"✅ FAISS index built")
print(f"🔢 Dimension: {dim}")
print(f"📦 Total vectors: {index.ntotal}")

# =========================
# Save Index & Metadata
# =========================
faiss.write_index(index, str(faiss_index_path))

with open(metadata_path, "wb") as f:
    pickle.dump(metadata_list, f)

print("💾 FAISS index + metadata saved successfully")
print(f"📍 Index: {faiss_index_path}")
print(f"📍 Metadata: {metadata_path}")
