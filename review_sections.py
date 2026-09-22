r"""Bölüm etiketlemesini kontrol etmek için rapor üretir.

1) Tüm kararlarda şüpheli durumları sayar (gerekçesi yok, gürültü çok uzun ...).
2) data/n<boyut>/section_review.csv: test setindeki kararlar + her kaynaktan birkaç rastgele karar için
   her bölümün etiketi, uzunluğu, başı ve sonu. 'dogru_etiket' sütununa yanlış etiketlerin doğrusu yazılır.

Kullanım:
    .venv\Scripts\python.exe review_sections.py
"""
import pandas as pd
from transformers import AutoTokenizer

import config
from chunking_methods.section_labels import segment

RANDOM_PER_SOURCE = 3
OUT = config.DATASET_DIR / "section_review.csv"


def flags(sections):
    """Bir kararın bölüm listesindeki şüpheli durumlar."""
    tokens = sections.groupby("etiket").token.sum()
    found = []
    if "GEREKCE" not in tokens:
        found.append("gerekce_yok")
    if "HUKUM" not in tokens:
        found.append("hukum_yok")
    if tokens.get("GURULTU", 0) > 400:
        found.append("gurultu_uzun")
    if tokens.get("USTBILGI", 0) > 500:
        found.append("ustbilgi_uzun")
    if tokens.get("GEREKCE", 0) < 0.2 * tokens.sum():
        found.append("gerekce_kisa")
    return found


def main():
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    docs = pd.read_parquet(config.SAMPLE_PATH)
    tests = pd.read_csv(config.test_set_path("v2"), encoding="utf-8-sig", keep_default_na=False)
    test_ids = tests.drop_duplicates("karar_id").set_index("karar_id").test_id.str[:3]

    rows = []
    for d in docs.itertuples():
        for order, (label, text) in enumerate(segment(d.source, d.text), 1):
            rows.append({"karar_id": d.id, "kaynak": d.source, "esas": d.esas_no, "sira": order,
                         "etiket": label, "metin": text})
    seg = pd.DataFrame(rows)
    seg["token"] = [len(x) for x in tok(seg.metin.tolist(), add_special_tokens=False)["input_ids"]]

    doc_flags = seg.groupby("karar_id").apply(lambda s: ",".join(flags(s)), include_groups=False)
    seg["uyarilar"] = seg.karar_id.map(doc_flags)

    print("Şüpheli durumların kaynağa göre oranı (% karar):")
    per_doc = pd.DataFrame({"kaynak": docs.set_index("id").source, "uyarilar": doc_flags})
    for name in ["gerekce_yok", "hukum_yok", "gurultu_uzun", "ustbilgi_uzun", "gerekce_kisa"]:
        per_doc[name] = per_doc.uyarilar.str.contains(name)
    print((100 * per_doc.groupby("kaynak").mean(numeric_only=True)).round(0).astype(int).to_string())

    random_ids = docs[~docs.id.isin(test_ids.index)].groupby("source").sample(RANDOM_PER_SOURCE, random_state=1).id
    chosen = list(test_ids.index) + list(random_ids)
    review = seg[seg.karar_id.isin(chosen)].copy()
    review["test_id"] = review.karar_id.map(test_ids).fillna("")
    review["basi"] = review.metin.str[:200]
    review["sonu"] = review.metin.str[-120:]
    review["dogru_etiket"] = ""
    review["karar_id"] = pd.Categorical(review.karar_id, categories=chosen, ordered=True)
    review = review.sort_values(["karar_id", "sira"])
    columns = ["test_id", "kaynak", "esas", "karar_id", "uyarilar", "sira", "etiket", "token", "basi", "sonu", "dogru_etiket"]
    review[columns].to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n{len(chosen)} karar, {len(review)} bölüm -> {OUT}")


if __name__ == "__main__":
    main()
