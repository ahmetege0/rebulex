r"""Test setindeki sorguları çalıştırıp aramanın başarısını ölçer.

İki tür ölçü raporlanır:
    tek hedef : yalnızca test setindeki hedef karar (ve ek doğru kararlar) doğru sayılır  -> sira, ilk1_% ...
    geniş     : ayrıca hedefin kardeş kararları ve kanun ortaklığı yüksek kararlar da doğru sayılır
                (relevance.py)                                                          -> sira_genis, ilk1_genis_% ...

Sorgu sorgu sonuçlar data/n<boyut>/eval/<deneme>_<test seti>.csv'ye yazılır; özet bir satır olarak
data/experiments.csv'ye eklenir (deneme koşulları build_index.py'nin kaydından alınır).

Kullanım:
    .venv\Scripts\python.exe evaluate.py                          # config.py'deki deneme ve test seti
    .venv\Scripts\python.exe evaluate.py --run sentence_c1024_o128
    .venv\Scripts\python.exe evaluate.py --test-set v1
    .venv\Scripts\python.exe evaluate.py --test-set dilekce_v1   # dilekçeden karar arama
"""
import argparse
import json
import re
import textwrap
from datetime import datetime

import pandas as pd
from qdrant_client import QdrantClient

import config
from relevance import Relevance
from search import load_model, search

K = 10  # her sorguda ilk kaç karara bakılacağı


def norm(text):
    """Boşlukları tekleştirir; kanıt cümlesi satır sonlarından bağımsız aransın diye."""
    return re.sub(r"\s+", " ", text).strip()


MARKS = {"hedef": "[DOĞRU] ", "kardes": "[KARDEŞ] ", "kanun": "[KANUN] ", None: ""}


def describe(hit, reason):
    p = hit.payload
    return f"{MARKS[reason]}[{hit.score:.3f}] {p['header']} (parça {p['chunk_index'] + 1}/{p['n_chunks']})"


def count_tokens(tokenizer, texts):
    return sum(len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"])


def run_query(model, client, collection, row, relevance):
    hits = search(model, client, collection, row.sorgu, k=K)
    accepted = {row.karar_id, *filter(None, row.ek_dogru_kararlar.split(";"))}
    # her sonuç neden doğru sayılıyor: hedef / kardes / kanun / None
    reasons = ["hedef" if h.payload["decision_id"] in accepted else relevance.match(row.karar_id, h.payload["decision_id"])
               for h in hits]
    found = next((i for i, r in enumerate(reasons) if r == "hedef"), None)
    found_wide = next((i for i, r in enumerate(reasons) if r), None)
    # Okuyan kişi ilk 5 sonucu görüyor: cevabı içeren cümle bu 5 parçanın birinde mi?
    evidence = any(norm(row.kanit) in norm(hit.payload["text"]) for hit in hits[:5])

    result = {
        "test_id": row.test_id,
        "sorgu_tipi": row.sorgu_tipi,
        "kaynak": row.kaynak,
        "sorgu": row.sorgu,
        "hedef_kunye": row.kunye,
        "sira": None if found is None else found + 1,
        "sira_genis": None if found_wide is None else found_wide + 1,
        "genis_neden": "" if found_wide is None else reasons[found_wide],
        "ilgili_ilk5": sum(bool(r) for r in reasons[:5]),  # ilk 5 sonucun kaçı doğru sayılıyor (geniş)
        "kanit_ilk5": evidence,
        # okuma maliyeti: ilk 5 sonucun parçaları toplam kaç token
        "okunan_token_ilk5": count_tokens(model.tokenizer, [hit.payload["text"] for hit in hits[:5]]),
        "kanit": row.kanit,
    }
    for i in range(5):
        result[f"sonuc_{i + 1}"] = describe(hits[i], reasons[i]) if i < len(hits) else ""
    # bölüm bazlı yöntemlerde doğru karar hangi bölüm grubundan bulundu (dava_ozeti / gerekce / karar)
    result["bulunan_bolum"] = "" if found is None else (hits[found].payload.get("bolum") or "")
    result["bulunan_parca_metni"] = "" if found is None else hits[found].payload["text"]
    return result


def summarize(frame):
    return pd.Series({
        "sorgu": len(frame),
        "ilk1_%": 100 * (frame.sira == 1).mean(),
        "ilk5_%": 100 * (frame.sira <= 5).mean(),
        "ilk10_%": 100 * frame.sira.notna().mean(),
        "mrr": (1 / frame.sira).fillna(0).mean(),
        "kanit_ilk5_%": 100 * frame.kanit_ilk5.mean(),
        "okunan_token_ilk5": frame.okunan_token_ilk5.mean(),
        "ilk1_genis_%": 100 * (frame.sira_genis == 1).mean(),
        "ilk5_genis_%": 100 * (frame.sira_genis <= 5).mean(),
        "ilk10_genis_%": 100 * frame.sira_genis.notna().mean(),
        "mrr_genis": (1 / frame.sira_genis).fillna(0).mean(),
        "ilgili_ilk5_%": 100 * frame.ilgili_ilk5.mean() / 5,
    })


def log_experiment(run, test_set, summary):
    """Deneme koşullarını ve sonuçlarını experiments.csv'ye yeni bir satır olarak ekler."""
    info_path = config.index_info_path(run)
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"deneme": run}
    row = {**info, "test_seti": test_set, "test_sorgu": int(summary["sorgu"]),
           **summary.drop("sorgu").round(2).to_dict(), "olcum_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M")}
    path = config.EXPERIMENTS_PATH
    # Dosyanın tamamı okunup yeniden yazılır: sütun eklenirse eski satırlar kaymaz
    old = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def summarize_by(frame, column):
    return pd.DataFrame({name: summarize(group) for name, group in frame.groupby(column)}).T


def evaluate(run=config.RUN_NAME, test_set=config.TEST_SET_VERSION):
    collection = config.collection_name(run)
    client = QdrantClient(path=str(config.QDRANT_PATH))
    if not client.collection_exists(collection):
        client.close()
        raise SystemExit(f"'{collection}' indeksi yok. Önce build_index.py çalıştırın.")
    model = load_model()
    tests = pd.read_csv(config.test_set_path(test_set), encoding="utf-8-sig", keep_default_na=False)
    pool = pd.read_parquet(config.SAMPLE_PATH, columns=["id", "text"])
    relevance = Relevance(dict(zip(pool.id, pool.text)))  # atıf ağırlıkları havuzun tamamına göre

    results = pd.DataFrame([run_query(model, client, collection, row, relevance) for row in tests.itertuples()])
    out = config.eval_path(run, test_set)
    results.to_csv(out, index=False, encoding="utf-8-sig")
    log_experiment(run, test_set, summarize(results))

    print(f"İndeks: {collection} | Test seti: {test_set} ({len(tests)} sorgu)\n")
    print("TOPLAM")
    print(summarize(results).round(2).to_string())
    print("\nSORGU TİPİNE GÖRE")
    print(summarize_by(results, "sorgu_tipi").round(2).to_string())
    print("\nKAYNAĞA GÖRE")
    print(summarize_by(results, "kaynak").round(2).to_string())

    print("\nGENİŞ ÖLÇÜDE DOĞRU SAYILMA NEDENİ (ilk 10'da ilk doğru sonuç):")
    print(results.genis_neden.replace("", "bulunamadı").value_counts().to_string())

    misses = results[results.sira_genis.isna()]
    print(f"\nİLK {K}'DA HİÇBİR DOĞRU SONUÇ OLMAYANLAR — geniş ölçüye göre ({len(misses)})")
    for m in misses.itertuples():
        # sorgular en fazla 200 karakter; dilekçelerin yalnızca başı gösterilir (tamamı CSV'de)
        print(f"  {m.test_id} | {m.hedef_kunye}\n      {textwrap.shorten(m.sorgu, 200)}")

    print(f"\nSorgu sorgu ayrıntılar: {out}")
    print(f"Deneme özeti eklendi: {config.EXPERIMENTS_PATH}")
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Arama başarısını test setiyle ölçer")
    parser.add_argument("--run", default=config.RUN_NAME,
                        help="Hangi denemenin indeksi ölçülecek (örn. per_decision, sentence_c1024_o128)")
    parser.add_argument("--test-set", default=config.TEST_SET_VERSION, help="Test seti sürümü (v1, v2, dilekce_v1)")
    args = parser.parse_args()
    evaluate(args.run, args.test_set)


if __name__ == "__main__":
    main()
