import json
from pathlib import Path
import re

# =========================
# Paths (تعديل المسار ليكون ديناميكي فقط)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "Data" / "cleaned_data" / "unified_legal_schema.json"
output_path = BASE_DIR / "Data" / "cleaned_data" / "unified_legal_schema_final.json"

output_path.parent.mkdir(parents=True, exist_ok=True)

# =========================
# Utilities
# =========================
def clean_text(text: str) -> str:
    """تنظيف النص القانوني بدون كسر المعنى"""
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_legal_text(text, max_chars=700):
    """
    تقسيم النص القانوني:
    - نحاول نفصل عند ؛ . ؟
    - نحافظ على السياق
    """
    sentences = re.split(r'(?<=[.؛؟])\s+', text)
    chunks = []
    buffer = ""

    for s in sentences:
        if len(buffer) + len(s) <= max_chars:
            buffer += (" " if buffer else "") + s
        else:
            chunks.append(buffer)
            buffer = s

    if buffer:
        chunks.append(buffer)

    return chunks

def detect_topic(text: str) -> str:
    """تصنيف قانوني أوضح"""
    topic_map = {
        "penalty": ["يعاقب", "العقوبة", "السجن", "الحبس", "الغرامة", "الإعدام"],
        "definition": ["يقصد", "المقصود", "تعريف", "يعد"],
        "procedure": ["إجراء", "تقديم", "إخطار", "طلب", "تحقيق"],
        "jurisdiction": ["تختص", "النيابة", "المحكمة", "الاختصاص"],
        "general": []
    }

    for topic, keywords in topic_map.items():
        if any(word in text for word in keywords):
            return topic

    return "general"

# =========================
# Load JSON
# =========================
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# Rebuild data
# =========================
final_data = []

for item in data:
    base_uid = item.get("uid")
    text = clean_text(item.get("text", ""))

    if not text:
        continue

    chunks = split_legal_text(text)

    for i, chunk in enumerate(chunks, start=1):
        final_data.append({
            "uid": f"{base_uid}_chunk_{i}" if len(chunks) > 1 else base_uid,
            "text": chunk,

            "doc_type": "law",
            "law_name": item.get("law_name"),
            "law_number": item.get("law_number"),
            "law_year": item.get("law_year"),
            "country": item.get("country", "Egypt"),

            "article_number": item.get("article_number"),

            "text_role": "legal_article",
            "topic": detect_topic(chunk),

            "is_citable": True,
            "source_file": item.get("source_file")
        })

# =========================
# Save final JSON
# =========================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print("✅ Final unified legal schema created successfully")
print(f"📄 Total records: {len(final_data)}")
print(f"📍 Saved to: {output_path}")
