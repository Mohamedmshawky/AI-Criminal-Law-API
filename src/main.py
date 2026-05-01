import faiss
import pickle
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

# =========================
# Paths
# =========================
VECTOR_DB_PATH = Path(r"\backend\Data\vector_db\faiss_index_muffakir.bin")
METADATA_PATH  = Path(r"\backend\Data\vector_db\metadata_muffakir.pkl")
# =========================
# Load FAISS + Metadata
# =========================
@st.cache_resource
def load_vector_db():
    index = faiss.read_index(str(VECTOR_DB_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# =========================
# Load Embedding Model
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("mohamed2811/Muffakir_Embedding", device="cpu")

# =========================
# Load Generation Model
# =========================
@st.cache_resource
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
        from langchain_community.llms import Ollama
        return Ollama(model=model_choice,
                    temperature=0.3,top_p= 0.1,
                    num_predict= 600
                     )

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
# Retrieval
# =========================
def retrieve_candidates(query, index, metadata, emb_model, top_n=30):
    q_vec = embed_text(query, emb_model)
    scores, ids = index.search(np.array([q_vec], dtype=np.float32), min(top_n, index.ntotal))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        meta = metadata[idx]
        if meta.get("text"):
            meta["_score"] = float(score)
            results.append(meta)
    results.sort(key=lambda x: x["_score"], reverse=True)
    return results

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
# Build Context
# =========================
def build_legal_context(articles, max_chars=1800):
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
                # Ollama LLM
                answer = generator.invoke(prompt)
    except Exception as e:
            answer = f"❌ خطأ أثناء التوليد: {str(e)}"

    return answer

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="⚖️ المساعد القانوني المصري", layout="wide")
st.title("⚖️ المساعد القانوني المصري")
st.markdown("<center style='color:gray'>مشروع تخرج – جامعة القاهرة</center>", unsafe_allow_html=True)

index, metadata = load_vector_db()
emb_model = load_embedding_model()
model_choice = st.selectbox(
    "🤖 اختر نموذج الإجابة:",
    ["qwen", "gemma3:4b", "llama3:8b"]
)

gen_model = load_generation_model(model_choice)

query = st.text_area("📝 اكتب سؤالك القانوني:", height=100)
if st.button("📨 إرسال"):
    if query.strip():
        with st.spinner("🔍 جاري التحليل القانوني..."):
            intent = infer_intent(query)
            candidates = retrieve_candidates(query, index, metadata, emb_model)
            filtered = filter_by_intent(candidates, intent)
            articles = group_by_article(filtered)
            context = build_legal_context(articles)
            answer = generate_answer(gen_model, context, query, articles)
        st.subheader("📌 الإجابة القانونية")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
        if articles:
            st.subheader("📚 النصوص المستخدمة")
            for src in articles:
                with st.expander(f"{src['law_name']} – {src['article_number']}"):
                    st.write(src["text"])
    else:
        st.error("❌ برجاء كتابة سؤال")
