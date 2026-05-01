import json
from pathlib import Path

# =========================
# Paths (تعديل المسار ليكون ديناميكي فقط)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Data" / "json_data"
OUTPUT_PATH = BASE_DIR / "Data" / "cleaned_data" / "unified_legal_schema.json"

# =========================
# Utils
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_law_metadata(filename: str):
    """
    استنتاج بيانات القانون من اسم الملف
    تقدري تطوريها بعدين لو حبيتي
    """
    return {
        "law_name": filename.replace(".json", ""),
        "law_number": None,
        "law_year": None,
        "country": "Egypt"
    }

# =========================
# Transformer
# =========================
def transform_law_file(data, source_file):
    unified = []
    meta = infer_law_metadata(source_file)

    for idx, item in enumerate(data, start=1):
        article_number = (
            item.get("رقم_المادة")
            or item.get("article_number")
            or str(idx)
        )

        text = (
            item.get("نص_المادة")
            or item.get("article_text")
            or item.get("text")
            or ""
        )

        unified.append({
            "uid": f"law_{source_file}_{article_number}",
            "text": text,

            "doc_type": "law",
            "law_name": meta["law_name"],
            "law_number": meta["law_number"],
            "law_year": meta["law_year"],
            "country": meta["country"],

            "article_number": article_number,

            "text_role": "legal_article",
            "is_citable": True,

            "source_file": source_file
        })

    return unified

# =========================
# Main
# =========================
def main():
    unified_data = []

    json_files = list(INPUT_DIR.glob("*.json"))
    print(f"📂 Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            data = load_json(json_file)

            if not isinstance(data, list):
                print(f"⚠️ Skipped (not list): {json_file.name}")
                continue

            unified_data.extend(
                transform_law_file(data, json_file.name)
            )

            print(f"✅ Processed: {json_file.name}")

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=2)

    print("\n🎉 DONE")
    print(f"📄 Total unified records: {len(unified_data)}")
    print(f"📍 Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
