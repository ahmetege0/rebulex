r"""Temiz metinle denemeler: 3 model, her deneme v2 ve dilekce_v1 ile ölçülür.

Önce 2000'lik havuzda 4 parçalama, sonra 10 binlik havuzda yalnızca ilk ikisi (per_decision ve büyük parça;
bu ikisinin 10 binde ham sonuçları var).

Ham denemelerden (run_experiments.py) farkları:
    - metin data/n<boyut>/kararlar_temiz.parquet'ten gelir (cleaning.py; yoksa ilk çalıştırmada üretilir).
      Orijinal kararlar.parquet'e dokunulmaz; her parça decision_id ile orijinal karara bağlı kalır.
    - parça başlığı yalnızca kaynak ve daire: "Yargıtay 13. Ceza Dairesi" (tam künye ~49 token, bu ~15 token)
    - her denemenin kendi Qdrant klasörü var (yerel Qdrant açılırken klasördeki bütün koleksiyonları belleğe alır)
    - arama doğrudan vektörler üzerinden yapılır: her sorguda bütün parçalar skorlanır, kararlar en iyi
      parçalarına göre sıralanır; böylece ilk 100 karar eksiksiz gelir
    - yalnızca geniş ölçü yazılır (relevance.py: hedef + kardeş + kanun ortaklığı), ilk 1/5/10/20/50/100
    - kanıt cümlesi de aynı temizleyiciden geçirilir; kanun ortaklığı orijinal metinden hesaplanır

Dosyalar: data/n<boyut>/temiz/<model>/chunks, embeddings, index_info, eval, qdrant/<deneme>
Özetler data/experiments.csv'ye eklenir; temiz denemelerin önüne bir kez "#####" satırı konur.

Her model ayrı bir süreçte çalışır: bir model hata verse de sıradakiler çalışır. Bitmiş denemeler atlanır;
yarıda kalırsa aynı komutla kaldığı yerden devam eder.

Kullanım:
    .venv\Scripts\python.exe run_clean_experiments.py                                # iki havuz, bütün modeller
    .venv\Scripts\python.exe run_clean_experiments.py --size 2000 --model cosmos_e5  # tek havuz, tek model
    .venv\Scripts\python.exe run_clean_experiments.py --size 2000 --model magibu_embedding --tam-kunye --deneme recursive_c1024_o128
"""
import argparse
import gc
import json
import re
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import config
from build_index import build
from cleaning import Cleaner
from relevance import Relevance
from search import load_model

SIZES = [2000, 10000]  # sırayla çalışır
TEST_SETS = ["v2", "dilekce_v1"]
RANKS = [1, 5, 10, 20, 50, 100]
SEPARATOR = "#" * 30

# config.py'deki yorumla aynı değerler (üçünde de TRUST_REMOTE_CODE = False)
MODELS = {
    "magibu_embedding": dict(MODEL_NAME="magibu/embeddingmagibu-200m", MODEL_MAX_TOKENS=8192,
                             QUERY_PROMPT="task: search result | query: ",
                             DOCUMENT_FORMAT="title: {title} | text: {text}"),
    "cosmos_e5": dict(MODEL_NAME="ytu-ce-cosmos/turkish-e5-large", MODEL_MAX_TOKENS=512,
                      QUERY_PROMPT="Instruct: Given a Turkish search query, retrieve relevant passages written in "
                                   "Turkish that best answer the query\nQuery: ",
                      DOCUMENT_FORMAT="{title}\n{text}"),
    "embeddinggemma": dict(MODEL_NAME="google/embeddinggemma-300m", MODEL_MAX_TOKENS=2048,
                           QUERY_PROMPT="task: search result | query: ",
                           DOCUMENT_FORMAT="title: {title} | text: {text}"),
    # Magibu'nun 2. modeli (EmbeddingGemma 2 tabanlı); önekleri 200m ile aynı. sentence-transformers>=6.1 ve
    # transformers>=5.19 ister (yalnızca Colab'da yükseltildi), Hugging Face'te erişim onayı gerekir
    "magibu2_text": dict(MODEL_NAME="alibayram/embeddingmagibu2-text-v1", MODEL_MAX_TOKENS=8192,
                         QUERY_PROMPT="task: search result | query: ",
                         DOCUMENT_FORMAT="title: {title} | text: {text}"),
}


def experiments(max_tokens, size):
    """(yöntem, parça sınırı, örtüşme); 512 token okuyan modelde (cosmos) büyük parça 512'dir.
    10 binlik havuzda yalnızca ilk ikisi."""
    big = 1024 if max_tokens >= 1024 else 512
    runs = [("per_decision", max_tokens, 0), ("recursive", big, big // 8), ("recursive", 128, 16), ("recursive", 64, 8)]
    return runs if size == config.BASE_SIZE else runs[:2]


def original_path(size):
    return config.dataset_dir(size) / "kararlar.parquet"


def clean_path(size):
    return config.dataset_dir(size) / "kararlar_temiz.parquet"


def make_clean_pool(size):
    pool = pd.read_parquet(original_path(size))
    cleaner = Cleaner()
    before = pool.text.str.split().str.len()
    pool["text"] = [cleaner.clean(source, text) for source, text in zip(pool.source, pool.text)]
    pool["text_len"] = pool.text.str.len()
    pool.to_parquet(clean_path(size), index=False)
    removed = 100 * (1 - pool.text.str.split().str.len().groupby(pool.source).sum() / before.groupby(pool.source).sum())
    print(f"Temiz havuz: {clean_path(size)}\nAtılan kelime (%):\n{removed.round(1).to_string()}")
    print(f"En kısa temiz metin: {pool.text_len.min()} karakter")


def variant(raw_text, full_title):
    """Çıktı klasörü: temiz (temiz metin, kısa başlık), ham_kisa_baslik veya temiz_tam_kunye."""
    if raw_text:
        return "ham_kisa_baslik"
    return "temiz_tam_kunye" if full_title else "temiz"


def use_model(name, size, raw_text=False, full_title=False):
    """Bu süreçteki config değerlerini seçilen modele ve havuza çevirir; config.py dosyası değişmez."""
    for key, value in MODELS[name].items():
        setattr(config, key, value)
    config.MODEL_FOLDER = name
    config.MODEL_SLUG = config.MODEL_NAME.split("/")[-1].replace("-", "_")
    config.MODEL_DIR = config.dataset_dir(size) / variant(raw_text, full_title) / name
    config.SAMPLE_PATH = original_path(size) if raw_text else clean_path(size)
    config.FULL_TITLE = full_title
    for folder in ["chunks", "embeddings", "index_info", "eval", "qdrant"]:
        (config.MODEL_DIR / folder).mkdir(parents=True, exist_ok=True)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


MARKS = {"hedef": "[DOĞRU] ", "kardes": "[KARDEŞ] ", "kanun": "[KANUN] ", None: ""}


def evaluate(model, run, test_set, relevance, cleaner, names):
    tests = pd.read_csv(config.test_set_path(test_set), encoding="utf-8-sig", keep_default_na=False)
    chunks = pd.read_parquet(config.chunks_path(run), columns=["decision_id", "chunk_index", "n_chunks", "text", "tokens"])
    vectors = np.load(config.embeddings_path(run))
    queries = model.encode(tests.sorgu.tolist(), prompt=config.QUERY_PROMPT, normalize_embeddings=True)

    results = []
    for row, query in zip(tests.itertuples(), queries):
        scores = vectors @ query
        # her karar en yüksek skorlu parçasıyla sıralanır
        best = chunks.iloc[np.argsort(-scores)].drop_duplicates("decision_id").head(max(RANKS))
        accepted = {row.karar_id, *filter(None, row.ek_dogru_kararlar.split(";"))}
        reasons = ["hedef" if d in accepted else relevance.match(row.karar_id, d) for d in best.decision_id]
        target = next((i + 1 for i, r in enumerate(reasons) if r == "hedef"), None)
        wide = next((i + 1 for i, r in enumerate(reasons) if r), None)
        # temiz metinde kanıt da temizlenir (cleaner verilmişse); ham metinde olduğu gibi aranır
        kanit = norm(cleaner.clean(row.kaynak, row.kanit) if cleaner else row.kanit)
        top5 = best.head(5)
        result = {
            "test_id": row.test_id, "sorgu_tipi": row.sorgu_tipi, "kaynak": row.kaynak, "sorgu": row.sorgu,
            "hedef_kunye": row.kunye,
            "sira": target,
            "sira_genis": wide,
            "genis_neden": reasons[wide - 1] if wide else "",
            "ilgili_ilk5": sum(bool(r) for r in reasons[:5]),
            "kanit_ilk5": any(kanit in norm(text) for text in top5.text),
            "okunan_token_ilk5": int(top5.tokens.sum()),
            "kanit": kanit,
        }
        for i, (index, hit) in enumerate(top5.iterrows()):
            result[f"sonuc_{i + 1}"] = (f"{MARKS[reasons[i]]}[{scores[index]:.3f}] {names[hit.decision_id]} "
                                        f"(parça {hit.chunk_index + 1}/{hit.n_chunks})")
        result["bulunan_parca_metni"] = best.text.iloc[target - 1] if target else ""
        results.append(result)

    results = pd.DataFrame(results)
    results.to_csv(config.eval_path(run, test_set), index=False, encoding="utf-8-sig")
    summary = {f"ilk{k}_genis_%": 100 * (results.sira_genis <= k).mean() for k in RANKS}
    summary["mrr_genis"] = (1 / results.sira_genis).fillna(0).mean()
    summary["kanit_ilk5_%"] = 100 * results.kanit_ilk5.mean()
    summary["okunan_token_ilk5"] = results.okunan_token_ilk5.mean()
    summary["ilgili_ilk5_%"] = 100 * results.ilgili_ilk5.mean() / 5
    log(run, test_set, len(results), summary)
    print(f"{run} | {test_set}: " + ", ".join(f"{k} {v:.1f}" for k, v in summary.items()))
    print(results.groupby("sorgu_tipi").sira_genis.apply(lambda s: f"ilk5 %{100 * (s <= 5).mean():.0f}, "
                                                                   f"ilk100 %{100 * (s <= 100).mean():.0f}").to_string())


def log(run, test_set, n_queries, summary):
    """experiments.csv'ye bir satır ekler; ilk temiz denemeden önce bir kez ayırıcı satır konur."""
    info = json.loads(config.index_info_path(run).read_text(encoding="utf-8"))
    text = "temiz" if config.SAMPLE_PATH.name == "kararlar_temiz.parquet" else "ham"
    title = "tam künye" if config.FULL_TITLE else "kaynak + daire"
    row = {**info, "test_seti": test_set, "test_sorgu": n_queries, **{k: round(v, 2) for k, v in summary.items()},
           "olcum_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M"), "not": f"{text} veri; başlık: {title}"}
    path = config.EXPERIMENTS_PATH
    # metin olarak okunur ki eski satırlar olduğu gibi kalsın (boş hücre yüzünden 2000 -> 2000.0 olmasın)
    old = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if SEPARATOR not in old.tarih.values:
        old = pd.concat([old, pd.DataFrame([{"tarih": SEPARATOR}])], ignore_index=True)
    new = pd.DataFrame([{k: str(v) for k, v in row.items()}])
    pd.concat([old, new], ignore_index=True).fillna("").to_csv(path, index=False, encoding="utf-8-sig")


def run_model(name, size, raw_text=False, full_title=False, only=None):
    use_model(name, size, raw_text, full_title)
    original = pd.read_parquet(original_path(size), columns=["id", "source", "court", "esas_no", "text"]).fillna("")
    relevance = Relevance(dict(zip(original.id, original.text)))  # kanun ortaklığı orijinal metinden
    names = {r.id: " ".join(p for p in [config.SOURCE_LABELS[r.source], r.court, r.esas_no and f"{r.esas_no} E."] if p)
             for r in original.itertuples()}
    cleaner = None if raw_text else Cleaner()
    runs = [r for r in experiments(config.MODEL_MAX_TOKENS, size) if not only or config.run_name(*r) in only]
    for i, (method, tokens, overlap) in enumerate(runs, 1):
        run = config.run_name(method, tokens, overlap)
        todo = [t for t in TEST_SETS if not config.eval_path(run, t).exists()]
        print(f"\n{'=' * 70}\nn{size} {name} [{i}/{len(runs)}] {run}  ({datetime.now():%H:%M})\n{'=' * 70}")
        if not todo:
            print("Zaten ölçülmüş, atlanıyor.")
            continue
        config.QDRANT_PATH = config.MODEL_DIR / "qdrant" / run  # her denemenin kendi veritabanı
        if config.index_info_path(run).exists():
            print("İndeks zaten kurulu, yalnızca ölçülüyor.")
        else:
            build(method, tokens, overlap)
        # önceki model ekran kartından silinsin; yoksa iki model birden kalır, cosmos'ta 6 GB yetmiyor.
        # empty_cache boşalan belleği karta geri verir; verilmezse sonraki denemede bellek taşıp çok yavaşlıyor
        gc.collect()
        torch.cuda.empty_cache()
        model = load_model()
        for test_set in todo:
            evaluate(model, run, test_set, relevance, cleaner, names)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Temiz 2000'lik havuzda denemeler")
    parser.add_argument("--size", type=int, choices=SIZES, help="Verilmezse iki havuz sırayla")
    parser.add_argument("--model", choices=list(MODELS), help="Verilmezse bütün modeller sırayla")
    # başlığın ve temizliğin etkisini ayırmak için (çıktılar ayrı klasöre gider: ham_kisa_baslik, temiz_tam_kunye)
    parser.add_argument("--ham-metin", action="store_true", help="Temiz yerine orijinal metin (kısa başlıkla)")
    parser.add_argument("--tam-kunye", action="store_true", help="Başlıkta tam künye (temiz metinle)")
    parser.add_argument("--deneme", nargs="+", help="Yalnızca bu denemeler (örn. recursive_c1024_o128)")
    args = parser.parse_args()
    if args.ham_metin and args.tam_kunye:
        parser.error("ham metin + tam künye eski denemelerin aynısı; run_experiments.py ile çalıştırılır")
    sizes = [args.size] if args.size else SIZES
    for size in sizes:
        if not clean_path(size).exists():
            make_clean_pool(size)
    if args.model:
        for size in sizes:
            run_model(args.model, size, args.ham_metin, args.tam_kunye, args.deneme)
        return
    options = (["--ham-metin"] if args.ham_metin else []) + (["--tam-kunye"] if args.tam_kunye else []) \
        + (["--deneme", *args.deneme] if args.deneme else [])
    for size in sizes:
        for name in MODELS:
            # her model ayrı süreçte: per_decision'ın sınırı modülü yüklenirken modelin token sınırından hesaplanır,
            # ekran kartı belleği model aralarında tamamen boşalır, bir modelin hatası diğerlerini durdurmaz
            subprocess.run([sys.executable, __file__, "--size", str(size), "--model", name, *options])
    print(f"\nBitti ({datetime.now():%H:%M}). Sonuçlar: {config.EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
