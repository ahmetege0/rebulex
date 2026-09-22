r"""Bir kararın nasıl parçalara bölündüğünü gösterir.

Kullanım:
    .venv\Scripts\python.exe show_chunks.py                                  # Danıştay'dan örnek
    .venv\Scripts\python.exe show_chunks.py yargitay                         # başka bir kaynaktan örnek
    .venv\Scripts\python.exe show_chunks.py emsal sentence_c1024_o128        # başka bir denemeden
"""
import sys

import pandas as pd

import config

source = sys.argv[1] if len(sys.argv) > 1 else "danistay"
run = sys.argv[2] if len(sys.argv) > 2 else config.RUN_NAME
chunks = pd.read_parquet(config.chunks_path(run))
print(f"Deneme: {run}")

print("Kaynak başına karar ve parça sayısı:")
print(chunks.groupby("source").agg(karar=("decision_id", "nunique"), parca=("text", "size")).to_string())

# 3-5 parçalı bir karar: parça sınırlarını ve örtüşmeyi görmek için ideal uzunluk
candidates = chunks[(chunks.source == source) & chunks.n_chunks.between(3, 5)]
if candidates.empty:
    candidates = chunks[chunks.source == source]
decision = chunks[chunks.decision_id == candidates.decision_id.iloc[0]]

print(f"\nÖrnek karar: {decision.header.iloc[0]}  ->  {len(decision)} parça\n")
for row in decision.itertuples():
    print(f"--- Parça {row.chunk_index + 1}/{row.n_chunks}  ({len(row.text)} karakter) ---")
    print(f"  BAŞI: {row.text[:250]} ...")
    print(f"  SONU: ... {row.text[-200:]}")
    print()
