import faiss
import pickle
import numpy as np
import torch
import os
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# Paths (Dynamic Setup)
# =========================
# BASE_DIR بتخرج خطوتين من src للوصول لفولدر backend الرئيسي
BASE_DIR = Path(__file__).resolve().parent.parent

# مسارات الـ Vector DB (موجودة في Data/vector_db)
VECTOR_DB_PATH = BASE_DIR / "Data" / "vector_db" / "faiss_index_muffakir.bin"
METADATA_PATH  = BASE_DIR / "Data" / "vector_db" / "metadata_muffakir.pkl"

# =========================
# Load FAISS + Metadata
# =========================
def load_vector_db():
    if not VECTOR_DB_PATH.exists():
        print(f"⚠️ Warning: Index file not found at {VECTOR_DB_PATH}")
        # ممكن ترجع None وتعمل handling في الـ api.py لو حابب
    
    index = faiss.read_index(str(VECTOR_DB_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# =========================
# Load Embedding Model
# =========================
def load_embedding_model():
    # هيتحمل أوتوماتيك من Hugging Face Hub
    return SentenceTransformer("mohamed2811/Muffakir_Embedding", device="cpu")

# =========================
# Load Generation Model
# =========================
def load_generation_model(model_choice):
    if model_choice == "qwen":
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        return pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=220,
                        do_sample=False,
                        temperature=0.0)
    else:
        # ملاحظة: Ollama يتطلب تشغيل محلي للسيرفر، لو على Hugging Face يفضل استخدام نماذج Transformers
        from langchain_community.llms import Ollama
        return Ollama(model=model_choice, temperature=0.0, top_p=0.1, num_predict=600)

# =========================
# Embedding & Intent
# =========================
def embed_text(text, model):
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def infer_intent(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["عقوبة", "يعاقب", "الحبس", "الغرامة"]):
        return "penalty"
    if any(w in q for w in ["تعريف", "ما هو", "مفهوم"]):
        return "definition"
    if any(w in q for w in ["إجراءات", "كيف", "متى"]):
        return "procedure"
    return "general"

# =========================
# Retrieval with Multi-Similarity & Reranking
# =========================
def retrieve_candidates(query, index, metadata, emb_model, top_n=30, similarity_type="cosine"):
    q_vec = embed_text(query, emb_model).reshape(1, -1).astype(np.float32)

    # البحث في FAISS
    D, I = index.search(q_vec, min(top_n, index.ntotal))

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < len(metadata):
            meta = metadata[idx].copy()
            meta["_score"] = float(score)
            results.append(meta)

    # ريرانكينج بسيط بناءً على المقياس
    results.sort(key=lambda x: x["_score"], reverse=(similarity_type == "cosine"))
    return results[:top_n]

# =========================
# Filter by Intent
# =========================
def filter_by_intent(results, intent):
    if intent == "penalty":
        filtered = [r for r in results if r.get("topic") == "penalty"]
        return filtered if filtered else results[:10]
    return results[:10]

# =========================
# Group by Article
# =========================
def group_by_article(chunks):
    grouped = defaultdict(list)
    for c in chunks:
        key = (c["law_name"], c["article_number"])
        grouped[key].append(c["text"])
    
    articles = []
    for (law, article), texts in grouped.items():
        articles.append({
            "law_name": law,
            "article_number": article,
            "text": "\n".join(texts)
        })
    return articles

# =========================
# Build Smart Context
# =========================
def build_legal_context(articles, max_chars=2000):
    context = ""
    for art in articles:
        block = f"{art['law_name']} – {art['article_number']}:\n{art['text']}\n\n"
        if len(context) + len(block) > max_chars:
            break
        context += block
    return context

# =========================
# Generate Answer with Guard
# =========================
def generate_answer(generator, context, question, sources):
    prompt = f"""
أنت مساعد قانوني مصري.

⚠️ قواعد إلزامية:
- استخدم النصوص التالية فقط.
- لا تذكر أي مادة أو عقوبة غير موجودة.
- إذا لم تجد نصًا صريحًا، اكتب حرفيًا:
  "لا يوجد نص صريح في المصادر المرفقة يحدد ذلك."

النصوص القانونية:
{context}

السؤال:
{question}

الإجابة القانونية:
"""
    if not sources:
        return "❌ لا يوجد نص قانوني صريح في قاعدة البيانات."

    try:
        if hasattr(generator, "__call__"):
            out = generator(prompt)
            if isinstance(out, list) and "generated_text" in out[0]:
                answer = out[0]["generated_text"].split("الإجابة القانونية:")[-1].strip()
            else:
                answer = str(out)
        else:
            # Ollama / Langchain
            answer = generator.invoke(prompt)
    except Exception as e:
        answer = f"❌ خطأ أثناء التوليد: {str(e)}"

    return answer
