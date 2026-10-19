r"""Embedding aramasının getirdiği ilk 100 kararı bir reranker'la yeniden sıralar ve ölçer.

1. aşama (gece koşusundakiyle aynı arama): sorgu vektörü × parça vektörleri (.npy). Her karar en iyi parçasıyla
   sıralanır, ilk 100 karar aday olur. İndeks yeniden kurulmaz; yalnızca test sorgularının vektörü hesaplanır.
2. aşama: her (sorgu, adayın en iyi parçası) çifti reranker'a verilir, reranker tek bir puan döndürür.
   Adaylar bu puana göre yeniden sıralanır. Reranker ilk 100'ün dışından karar getiremez: ilk100 değişmez,
   ilk 1/5/10/20/50 değişir.

"yok" adlı reranker 1. aşamanın sırasını olduğu gibi bırakır; aynı kodla ölçülmüş karşılaştırma noktasıdır.

--metin: reranker'ın okuduğu metin.
    parca     adayın en iyi parçası (sınır 1664 token)
    temiz     kararın temiz metni (kararlar_temiz.parquet), sınır 8192 token
    orijinal  kararın orijinal metni (kararlar.parquet), sınır 8192 token
    Tam metinde sorgu + karar sınıra sığmıyorsa karar parçalara bölünür; her parça sorguyla ayrı okunur,
    kararın puanı en yüksek parçanın puanıdır.

Çıktılar. Her (kaynak, reranker, test seti) ayrı dosyaya yazılır, hiçbiri bir diğerinin üzerine yazmaz.
Bitmiş ölçümler atlanır; yarıda kesilirse aynı komutla kaldığı yerden devam eder.
    data/n2000/rerank/<kaynak>/<reranker>/<deneme>_<test seti>.csv   sorgu sorgu: sira_once, sira_sonra ...
    (tam metinde klasör rerank_temiz / rerank_orijinal)
    data/experiments.csv                                              ölçüm başına bir satır (önünde bir kez "#####")

Kullanım:
    .venv\Scripts\python.exe rerank.py                                     # bütün kaynaklar, bütün rerankerlar
    .venv\Scripts\python.exe rerank.py --kaynak temiz/cosmos_e5 --reranker yok modernbert_tr --test-set v2
    .venv\Scripts\python.exe rerank.py --metin temiz --reranker bge_m3
    .venv\Scripts\python.exe rerank.py --bellek-testi        # her reranker'a bir grup en uzun çift; bellek yetiyor mu
"""
import argparse
import gc
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from relevance import Relevance
from run_clean_experiments import MARKS, MODELS, RANKS
from search import load_model

SIZE = 2000
POOL_DIR = config.dataset_dir(SIZE)
TEST_SETS = ["v2", "dilekce_v1"]
SEPARATOR = "#" * 30 + " reranker"

# Aday kaynakları: (data/n2000 altındaki klasör, deneme). Klasörün son parçası embedding modelinin adı.
# data/n2000/magibu_embedding = ham metin + tam künye (eski denemeler)
SOURCES = [
    ("magibu_embedding", "recursive_c1024_o128"),
    ("temiz/cosmos_e5", "recursive_c64_o8"),
    ("temiz/embeddinggemma", "per_decision"),
]
RERANKERS = {
    "yok": None,
    "bge_m3": "BAAI/bge-reranker-v2-m3",
    "modernbert_tr": "ytu-ce-cosmos/modernbert-tr-reranker",
    "qwen3_06b": "Qwen/Qwen3-Reranker-0.6B",
}
CANDIDATES = 100  # reranker'a verilen aday karar sayısı
# reranker'ın okuduğu sorgu + metin. Parçada: dilekçe ~600 + parça 1024; uzunsa parçanın sonu kesilir.
# Tam metinde: üç reranker da 8192 okuyabiliyor; sığmayan karar parçalara bölünür
MAX_TOKENS = {"parca": 1664, "temiz": 8192, "orijinal": 8192}
MARGIN = 128  # tam metni bölerken şablon ve özel tokenlar için ayrılan pay
# ekran kartında aynı anda puanlanan çift sayısı; varsayılan 6 GB'a göre, Colab L4/A100'de ortam değişkeniyle büyütülür
BATCH_SIZE = int(os.environ.get("REBULEX_RERANK_BATCH", 8))


# ---------------------------------------------------------------- 1. aşama: embedding araması
def first_stage(folder, run):
    """Her test seti için aday tablosu: test_id, sira_once (1-100), decision_id, chunk_index, n_chunks, text."""
    model_name = folder.split("/")[-1]
    for key, value in MODELS[model_name].items():
        setattr(config, key, value)
    slug = config.MODEL_NAME.split("/")[-1].replace("-", "_")
    chunks = pd.read_parquet(POOL_DIR / folder / "chunks" / f"{run}.parquet",
                             columns=["decision_id", "chunk_index", "n_chunks", "text"])
    vectors = np.load(POOL_DIR / folder / "embeddings" / f"{slug}_{run}.npy")
    model = load_model()

    candidates = {}
    for test_set in TEST_SETS:
        tests = pd.read_csv(config.test_set_path(test_set), encoding="utf-8-sig", keep_default_na=False)
        queries = model.encode(tests.sorgu.tolist(), prompt=config.QUERY_PROMPT, normalize_embeddings=True)
        rows = []
        for test_id, query in zip(tests.test_id, queries):
            scores = vectors @ query
            # her karar en yüksek skorlu parçasıyla sıralanır (gece koşusundaki aramanın aynısı)
            best = chunks.iloc[np.argsort(-scores)].drop_duplicates("decision_id").head(CANDIDATES)
            rows.append(best.assign(test_id=test_id, sira_once=range(1, len(best) + 1)))
        candidates[test_set] = pd.concat(rows, ignore_index=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return candidates


# ---------------------------------------------------------------- 2. aşama: reranker
class Qwen3Reranker:
    """Qwen3-Reranker bir dil modeli. Ona "Bu belge sorguya cevap veriyor mu? Yalnızca yes ya da no" diye sorulur;
    puan, modelin "yes" deme olasılığıdır. Şablon ve talimat modelin Hugging Face sayfasındaki kullanımdan."""
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    # göreve özel talimat (modelin sayfası İngilizce talimat öneriyor)
    INSTRUCTIONS = {
        "v2": "Given a description of a legal issue or case in Turkish, retrieve Turkish court decisions "
              "that rule on the same legal issue",
        "dilekce_v1": "Given a Turkish legal petition, retrieve Turkish court decisions that rule on the same "
                      "legal dispute",
    }

    def __init__(self, name, max_tokens):
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)

    def predict(self, pairs, batch_size, test_set):
        # uzun çiftler bir arada gruplansın diye uzunluğa göre (en uzun başta) sıralanır; puanlar sonra eski sıraya döner
        order = np.argsort([-len(query) - len(doc) for query, doc in pairs], kind="stable")
        scores = []
        for start in tqdm(range(0, len(pairs), batch_size)):
            texts = [f"<Instruct>: {self.INSTRUCTIONS[test_set]}\n<Query>: {pairs[i][0]}\n<Document>: {pairs[i][1]}"
                     for i in order[start:start + batch_size]]
            # metin sınırı aşarsa sonundan (belgenin sonundan) kesilir; şablonun başı ve sonu her zaman eklenir
            ids = self.tokenizer(texts, truncation=True, add_special_tokens=False,
                                 max_length=self.max_tokens - len(self.prefix) - len(self.suffix))["input_ids"]
            batch = self.tokenizer.pad({"input_ids": [self.prefix + x + self.suffix for x in ids]},
                                       return_tensors="pt").to(self.model.device)
            # yalnızca son tokenın kelime olasılıkları hesaplansın; hepsi hesaplanırsa bellek taşar
            # (32 çift × 1664 token × 151 bin kelime ≈ 15 GB). Dolgu solda olduğu için son token her çiftte gerçek son
            with torch.no_grad():
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, [self.no, self.yes]]
            scores += logits.float().softmax(-1)[:, 1].tolist()
        result = np.empty(len(pairs))
        result[order] = scores
        return result


class CrossEncoderReranker:
    """bge ve ModernBERT: sorgu ile parça tek girdi olarak okunur, model bir puan verir."""

    def __init__(self, name, max_tokens):
        self.model = CrossEncoder(name, max_length=max_tokens, device="cuda",
                                  model_kwargs={"dtype": torch.float16})
        self.tokenizer = self.model.tokenizer

    def predict(self, pairs, batch_size, test_set):
        # CrossEncoder çiftleri kendisi uzunluğa göre sıralar (en uzun başta)
        return self.model.predict(pairs, batch_size=batch_size, show_progress_bar=True)


def load_reranker(name, max_tokens):
    return Qwen3Reranker(name, max_tokens) if name.startswith("Qwen/") else CrossEncoderReranker(name, max_tokens)


def split_long(pairs, tokenizer, max_tokens):
    """Sığmayan kararı parçalara böler, her parça sorguyla ayrı bir çift olur.
    owner[i]: i. çiftin geldiği satır; satırın puanı parçalarının en yüksek puanı olur."""
    query_tokens = {}
    new_pairs, owner = [], []
    for row, (query, doc) in enumerate(pairs):
        if query not in query_tokens:
            query_tokens[query] = len(tokenizer(query, add_special_tokens=False)["input_ids"])
        room = max_tokens - query_tokens[query] - MARGIN
        # her tokenın metindeki karakter konumu; parçalar metinden olduğu gibi kesilir
        offsets = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        starts = [offsets[i][0] for i in range(0, len(offsets), room)] + [len(doc)]
        pieces = [doc[a:b] for a, b in zip(starts, starts[1:])] or [doc]
        new_pairs += [(query, piece) for piece in pieces]
        owner += [row] * len(pieces)
    return new_pairs, owner


# ---------------------------------------------------------------- ölçüm
def ranks(decision_ids, row, relevance):
    """Hedefin sırası (tek hedef) ve ilk doğru sayılan kararın sırası (geniş); ilk 100'de yoksa None."""
    accepted = {row.karar_id, *filter(None, row.ek_dogru_kararlar.split(";"))}
    reasons = ["hedef" if d in accepted else relevance.match(row.karar_id, d) for d in decision_ids]
    target = next((i + 1 for i, r in enumerate(reasons) if r == "hedef"), None)
    wide = next((i + 1 for i, r in enumerate(reasons) if r), None)
    return target, wide, reasons


def measure(candidates, test_set, relevance, names):
    """candidates'ta 'puan' sütunu var (reranker puanı); her sorgunun adayları bu puana göre yeniden sıralanır."""
    tests = pd.read_csv(config.test_set_path(test_set), encoding="utf-8-sig", keep_default_na=False)
    results = []
    for row in tests.itertuples():
        before = candidates[candidates.test_id == row.test_id].sort_values("sira_once")
        after = before.sort_values("puan", ascending=False, kind="stable")
        target_before, wide_before, _ = ranks(before.decision_id, row, relevance)
        target_after, wide_after, reasons = ranks(after.decision_id, row, relevance)
        result = {
            "test_id": row.test_id, "sorgu_tipi": row.sorgu_tipi, "kaynak": row.kaynak, "hedef_kunye": row.kunye,
            "sira_once": target_before, "sira_sonra": target_after,
            "sira_genis_once": wide_before, "sira_genis_sonra": wide_after,
            "genis_neden_sonra": reasons[wide_after - 1] if wide_after else "",
            "hedef_puan": after.puan.iloc[target_after - 1] if target_after else None,
        }
        for i, hit in enumerate(after.head(5).itertuples()):
            result[f"sonuc_{i + 1}"] = (f"{MARKS[reasons[i]]}[{hit.puan:.3f}] {names[hit.decision_id]} "
                                        f"(parça {hit.chunk_index + 1}/{hit.n_chunks})")
        results.append(result)
    return pd.DataFrame(results)


def summarize(results):
    summary = {f"ilk{k}_genis_%": 100 * (results.sira_genis_sonra <= k).mean() for k in RANKS}
    summary["mrr_genis"] = (1 / results.sira_genis_sonra).fillna(0).mean()
    return summary


def log(folder, run, test_set, reranker, metin, n_queries, summary, seconds):
    """experiments.csv'ye bir satır ekler; ilk reranker ölçümünden önce bir kez ayırıcı satır konur."""
    info = json.loads((POOL_DIR / folder / "index_info" / f"{run}.json").read_text(encoding="utf-8"))
    row = {**info, "test_seti": test_set, "test_sorgu": n_queries, "reranker": RERANKERS[reranker] or "yok",
           "aday_sayisi": CANDIDATES, "rerank_sn": round(seconds),
           **{k: round(v, 2) for k, v in summary.items()},
           "olcum_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M"), "not": f"rerank; metin: {metin}; aday: n{SIZE}/{folder}"}
    path = config.EXPERIMENTS_PATH
    old = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if SEPARATOR not in old.tarih.values:
        old = pd.concat([old, pd.DataFrame([{"tarih": SEPARATOR}])], ignore_index=True)
    new = pd.DataFrame([{k: str(v) for k, v in row.items()}])
    pd.concat([old, new], ignore_index=True).fillna("").to_csv(path, index=False, encoding="utf-8-sig")


def out_dir(metin):
    return POOL_DIR / ("rerank" if metin == "parca" else f"rerank_{metin}")


def out_path(metin, folder, reranker, run, test_set):
    return out_dir(metin) / folder / reranker / f"{run}_{test_set}.csv"


def memory_test(batch_size):
    """Her reranker'a bir grup en uzun (8192 tokenlık) çift verir; bellek hatası gece değil şimdi çıksın."""
    text = "Mahkeme kararı metni. " * 3000
    for reranker, name in RERANKERS.items():
        if not name:
            continue
        model = load_reranker(name, MAX_TOKENS["temiz"])
        start = time.time()
        model.predict([("dava dilekçesi " * 300, text)] * batch_size, batch_size, "dilekce_v1")
        print(f"{reranker}: {batch_size} çift × 8192 token sorunsuz ({time.time() - start:.0f} sn), "
              f"en yüksek bellek {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser(description="İlk 100 adayı reranker'la yeniden sıralar ve ölçer")
    parser.add_argument("--kaynak", nargs="+", help="Yalnızca bu klasörler (örn. temiz/cosmos_e5)")
    parser.add_argument("--reranker", nargs="+", choices=list(RERANKERS), default=list(RERANKERS))
    parser.add_argument("--test-set", nargs="+", choices=TEST_SETS, default=TEST_SETS)
    parser.add_argument("--metin", choices=list(MAX_TOKENS), default="parca", help="Reranker'ın okuduğu metin")
    parser.add_argument("--bellek-testi", action="store_true", help="Yalnızca bellek testi")
    args = parser.parse_args()
    if args.bellek_testi:
        memory_test(BATCH_SIZE)
        return
    sources = [s for s in SOURCES if not args.kaynak or s[0] in args.kaynak]
    todo = [(folder, run, reranker, test_set) for folder, run in sources for reranker in args.reranker
            for test_set in args.test_set if not out_path(args.metin, folder, reranker, run, test_set).exists()]
    if not todo:
        print("Bütün ölçümler zaten yapılmış.")
        return

    original = pd.read_parquet(config.dataset_dir(SIZE) / "kararlar.parquet",
                               columns=["id", "source", "court", "esas_no", "text"]).fillna("")
    relevance = Relevance(dict(zip(original.id, original.text)))  # kanun ortaklığı orijinal metinden
    # tam metinde reranker'ın okuyacağı karar metinleri (decision_id → metin)
    if args.metin == "orijinal":
        full_texts = dict(zip(original.id, original.text))
    elif args.metin == "temiz":
        clean = pd.read_parquet(POOL_DIR / "kararlar_temiz.parquet", columns=["id", "text"]).fillna("")
        full_texts = dict(zip(clean.id, clean.text))
    names = {r.id: " ".join(p for p in [config.SOURCE_LABELS[r.source], r.court, r.esas_no and f"{r.esas_no} E."] if p)
             for r in original.itertuples()}

    # 1. aşama: her kaynak için adaylar (embedding modelleri sırayla yüklenip bırakılır)
    candidates = {(folder, run): first_stage(folder, run)
                  for folder, run in sources if any(t[:2] == (folder, run) for t in todo)}

    # 2. aşama: her reranker bir kez yüklenir, bütün kaynaklarda çalışır
    for reranker in args.reranker:
        jobs = [t for t in todo if t[2] == reranker]
        if not jobs:
            continue
        print(f"\n{'=' * 70}\nreranker: {reranker} ({RERANKERS[reranker] or 'sıra değişmez'})\n{'=' * 70}")
        model = load_reranker(RERANKERS[reranker], MAX_TOKENS[args.metin]) if RERANKERS[reranker] else None
        for folder, run, _, test_set in jobs:
            frame = candidates[(folder, run)][test_set].copy()
            start = time.time()
            if model:
                queries = pd.read_csv(config.test_set_path(test_set), encoding="utf-8-sig",
                                      keep_default_na=False).set_index("test_id").sorgu
                texts = frame.text if args.metin == "parca" else frame.decision_id.map(full_texts)
                pairs = list(zip(frame.test_id.map(queries), texts))
                if args.metin == "parca":
                    frame["puan"] = model.predict(pairs, BATCH_SIZE, test_set)
                else:
                    pairs, owner = split_long(pairs, model.tokenizer, MAX_TOKENS[args.metin])
                    scores = model.predict(pairs, BATCH_SIZE, test_set)
                    frame["puan"] = pd.Series(scores).groupby(owner).max().to_numpy()
            else:
                frame["puan"] = -frame.sira_once  # sıra değişmez
            seconds = time.time() - start
            results = measure(frame, test_set, relevance, names)
            path = out_path(args.metin, folder, reranker, run, test_set)
            path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(path, index=False, encoding="utf-8-sig")
            summary = summarize(results)
            log(folder, run, test_set, reranker, args.metin, len(results), summary, seconds)
            print(f"{folder} {run} | {test_set} | {seconds:.0f} sn | "
                  + ", ".join(f"{k} {v:.1f}" for k, v in summary.items()))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\nBitti. Sonuçlar: {out_dir(args.metin)} ve {config.EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
