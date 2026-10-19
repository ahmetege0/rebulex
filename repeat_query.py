r"""Aynı sorguyu defalarca arar: sonuçlar her seferinde aynı mı?

Model bir kez yüklenir, sorgu --times kez aranır. Her denemede hedefin sırası, ilk 10 kararın listesi
ve 1. sonucun skoru tutulur. Özet data/experiments.csv'ye bir satır olarak eklenir
(test_seti = tekrar_<test_id>_x<kez>, test_sorgu = deneme sayısı).

Kullanım:
    .venv\Scripts\python.exe repeat_query.py --test-id T20-F --run per_decision --times 100
"""
import argparse
import json
from datetime import datetime

import pandas as pd
from qdrant_client import QdrantClient

import config
from search import load_model, search


def main():
    parser = argparse.ArgumentParser(description="Aynı sorguyu defalarca arar")
    parser.add_argument("--test-id", default="T20-F")
    parser.add_argument("--test-set", default="v2")
    parser.add_argument("--run", default="per_decision")
    parser.add_argument("--times", type=int, default=100)
    args = parser.parse_args()

    tests = pd.read_csv(config.test_set_path(args.test_set), encoding="utf-8-sig", keep_default_na=False)
    row = tests[tests.test_id == args.test_id].iloc[0]
    model = load_model()
    client = QdrantClient(path=str(config.QDRANT_PATH))
    collection = config.collection_name(args.run)

    rankings, ranks, top_scores = [], [], []
    for _ in range(args.times):
        hits = search(model, client, collection, row.sorgu, k=10)
        ids = [h.payload["decision_id"] for h in hits]
        rankings.append(tuple(ids))
        ranks.append(ids.index(row.karar_id) + 1 if row.karar_id in ids else None)
        top_scores.append(hits[0].score)
    client.close()

    ranks = pd.Series(ranks, dtype="float")
    found = int((ranks <= 5).sum())
    different = len(set(rankings))
    score_range = max(top_scores) - min(top_scores)
    print(f"Sorgu: {row.sorgu}")
    print(f"Hedef ilk 5'te: {found}/{args.times}, ilk 5'te değil: {args.times - found}")
    print(f"Hedefin sıraları (sıra: kaç kez): {ranks.value_counts(dropna=False).to_dict()}")
    print(f"Farklı sonuç listesi sayısı: {different}")
    print(f"1. sonucun skoru: en düşük {min(top_scores):.8f}, en yüksek {max(top_scores):.8f}")

    # experiments.csv'ye bir satır: deneme koşulları build_index.py'nin kaydından
    info = json.loads(config.index_info_path(args.run).read_text(encoding="utf-8"))
    log_row = {
        **info,
        "test_seti": f"tekrar_{args.test_id}_x{args.times}",
        "test_sorgu": args.times,
        "ilk1_%": 100 * (ranks == 1).mean(),
        "ilk5_%": 100 * (ranks <= 5).mean(),
        "ilk10_%": 100 * ranks.notna().mean(),
        "mrr": round((1 / ranks).fillna(0).mean(), 2),
        "not": f"aynı sorgu {args.times} kez arandı; farklı sonuç listesi: {different}, "
               f"hedefin sıraları: {ranks.value_counts(dropna=False).to_dict()}, 1. skor oynaması: {score_range:.2e}",
        "olcum_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    old = pd.read_csv(config.EXPERIMENTS_PATH, encoding="utf-8-sig")
    pd.concat([old, pd.DataFrame([log_row])], ignore_index=True).to_csv(
        config.EXPERIMENTS_PATH, index=False, encoding="utf-8-sig")
    print(f"Özet eklendi: {config.EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
