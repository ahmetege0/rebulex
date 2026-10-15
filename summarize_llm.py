r"""Kararları bir LLM ile yapılandırılmış biçimde özetler; özetleri vektörleştirip mini B aramasıyla dener.

Şablon testi: 5 hedef karar (TEST_IDS) magibu-11b-v0.8 ile özetlenir. Sonra mini B araması:
    - mevcut 2000'lik parça vektörleri (Magibu, recursive_c1024_o128) olduğu gibi kullanılır;
    - 5 özet "title: künye | text: özet" biçiminde aynı embedding modeliyle vektörleştirilir;
    - bu 5 hedefin sorguları (15 v2 + 5 dilekçe) iki kez sıralanır:
        özetsiz: kararın puanı = en iyi parçasının puanı
        özetli:  kararın puanı = max(en iyi parça, özet)   (özeti olmayan kararlarda değişmez)
DİKKAT: Yalnızca hedeflerin özeti olduğu için özetli sıra hedeflere haksız avantaj verir. Bu bir boru hattı
testidir, kazanç ölçümü değildir.

Model kullanımı Hugging Face model kartındaki gibi: pipeline("text-generation") + sohbet mesajları.
Modelin bağlam penceresi 32.768 token (model kartı).

Çıktılar (Colab'da bu klasör Drive'a bağlanır):
    data/n2000/ozet/<test_id>.json   her karar bittiği anda yazılır; varsa atlanır (kaldığı yerden devam)
    data/n2000/ozet/mini_b.csv       sorgu sorgu özetsiz / özetli sıralar

Kullanım:
    python summarize_llm.py
"""
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

import config
from build_index import decision_header
from relevance import Relevance
from rerank import ranks
from run_clean_experiments import MODELS
from search import load_model

SIZE = 2000
POOL_DIR = config.dataset_dir(SIZE)
OUT_DIR = POOL_DIR / "ozet"

LLM = "magibu/magibu-11b-v0.8"
SYSTEM = "Sadece istenen çıktıyı üret."  # Magibu'nun kendi Cetvel ölçümündeki sistem mesajı (model kartı)
MAX_NEW_TOKENS = 2048  # yalnızca güvenlik sınırı; gerçek özet uzunluğu örneklere bakılarak seçilecek
# Üretim ayarları (3. deneme):
# - 1. deneme greedy (do_sample=False): uzun kararlar döngüye girdi.
# - 2. deneme greedy + repetition_penalty=1.1: döngü sürdü; ceza girdideki (kararın) kelimelerine de uygulandığı
#   için harf kayıpları arttı ("omyon" <- otomasyon). Ceza kaldırıldı.
# - 3. deneme: düşük sıcaklıkla örnekleme. Her karardan önce aynı seed verilir: aynı karar aynı ekran kartında
#   yine aynı özeti verir.
TEMPERATURE = 0.3
TOP_P = 0.9
SEED = 42
TEMPLATE = """Aşağıda bir Türk mahkeme kararı var. Bu kararı, hukuki arama sisteminde kullanılmak üzere
aşağıdaki başlıklarla özetle.

Kurallar:
- Sadece kararda yazanları kullan; kararda olmayan bilgi ekleme.
- Kişi ve şirket adlarını yazma.
- GEREKÇE kısmına sadece mahkemenin kendi değerlendirmesini yaz; tarafların iddialarını mahkemenin görüşü gibi yazma.
- Kanun adını ve kısaltmasını (TBK, HMK, TTK, CMK vb.) kararda geçtiği gibi yaz, değiştirme.
- Harç, gider, vekâlet ücreti, faiz, tebliğ ve süre gibi usul ayrıntılarını yazma.
- Kararın hukuken önemli bütün noktalarını yaz; kısaltmaya çalışma ama aynı şeyi tekrar etme.
- Birden çok hukuki mesele varsa her birini ayrı madde yaz.
- Türkçe yaz.

DAVA TÜRÜ / KONU: (tek cümle)
OLAY: (somut olgular ve davanın süreci, birkaç cümle)
HUKUKİ MESELE(LER): (her biri, mahkemenin cevap verdiği bir soru biçiminde)
UYGULANAN KURALLAR: (kanun adı ve madde numarası, kararda geçtiği gibi)
GEREKÇE / BENİMSENEN İLKE: (mahkemenin değerlendirmesi ve vardığı ilke)
SONUÇ: (onama / bozma / kabul / ret / ihlal vb.)

KARAR METNİ:
{metin}"""

TEST_IDS = ["T01", "T12", "T21", "T28", "T29"]  # Yargıtay kısa, Emsal, Danıştay, AYM BB uzun, AYM norm
TEST_SETS = ["v2", "dilekce_v1"]
CHUNKS = ("magibu_embedding", "recursive_c1024_o128")  # mevcut parça vektörleri (ham metin + tam künye)


# ---------------------------------------------------------------- 1. özetleme
def summarize_all(decisions):
    """Her hedef kararı özetler; sonucu hemen <test_id>.json olarak yazar. Bitmiş kararlar atlanır."""
    todo = [r for r in decisions.itertuples() if not (OUT_DIR / f"{r.test_id}.json").exists()]
    if not todo:
        print("Bütün özetler zaten var.")
        return
    # model, model kartındaki gibi pipeline ile yüklenir; üretim, token sayısını görebilmek için doğrudan
    # model.generate ile yapılır
    generator = pipeline("text-generation", model=LLM, device="cuda", dtype=torch.bfloat16)
    tokenizer, model = generator.tokenizer, generator.model
    for r in todo:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": TEMPLATE.format(metin=r.text)}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(model.device)
        input_tokens = inputs["input_ids"].shape[1]
        start = time.time()
        torch.manual_seed(SEED)  # örnekleme her kararda aynı yerden başlar
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                    temperature=TEMPERATURE, top_p=TOP_P)
        seconds = time.time() - start
        new_ids = output[0, input_tokens:].tolist()
        # 2. denemede özel tokenları silmeden metne çevirmek hiçbir fark yaratmadı (0 özel token); düz yönteme dönüldü
        summary = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        result = {
            "test_id": r.test_id, "karar_id": r.id, "kunye": r.kunye,
            "karar_kelime": len(r.text.split()), "giris_token": input_tokens,
            "ozet_kelime": len(summary.split()), "ozet_token": len(new_ids),
            "sinira_carpti": len(new_ids) >= MAX_NEW_TOKENS,
            "sure_sn": round(seconds), "model": LLM, "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": model.generation_config.top_k, "seed": SEED,
            "ozet": summary,
        }
        (OUT_DIR / f"{r.test_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{r.test_id} | giriş {input_tokens} token | özet {len(new_ids)} token, {result['ozet_kelime']} kelime"
              f"{' (SINIRA ÇARPTI)' if result['sinira_carpti'] else ''} | {seconds:.0f} sn")
    del generator
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 2. mini B araması
def mini_search(decisions, original):
    folder, run = CHUNKS
    for key, value in MODELS[folder].items():
        setattr(config, key, value)
    slug = config.MODEL_NAME.split("/")[-1].replace("-", "_")
    chunk_ids = pd.read_parquet(POOL_DIR / folder / "chunks" / f"{run}.parquet", columns=["decision_id"]).decision_id
    vectors = np.load(POOL_DIR / folder / "embeddings" / f"{slug}_{run}.npy")
    model = load_model()

    summaries = [json.loads((OUT_DIR / f"{t}.json").read_text(encoding="utf-8")) for t in decisions.test_id]
    # özet, parçalarla aynı biçimde verilir: "title: {künye} | text: {özet}"
    headers = {r.id: decision_header(r) for r in original.itertuples()}
    inputs = [config.DOCUMENT_FORMAT.format(title=headers[s["karar_id"]], text=s["ozet"]) for s in summaries]
    summary_vectors = model.encode(inputs, normalize_embeddings=True)
    summary_ids = [s["karar_id"] for s in summaries]

    relevance = Relevance(dict(zip(original.id, original.text)))  # kanun ortaklığı orijinal metinden
    results = []
    for test_set in TEST_SETS:
        tests = pd.read_csv(config.test_set_path(test_set), encoding="utf-8-sig", keep_default_na=False)
        tests = tests[tests.test_id.str[:3].isin(TEST_IDS)]
        queries = model.encode(tests.sorgu.tolist(), prompt=config.QUERY_PROMPT, normalize_embeddings=True)
        for row, query in zip(tests.itertuples(), queries):
            chunk_scores = pd.Series(vectors @ query).groupby(chunk_ids.values).max()  # karar başına en iyi parça
            with_summary = chunk_scores.copy()
            summary_scores = pd.Series(summary_vectors @ query, index=summary_ids)
            with_summary[summary_ids] = np.maximum(with_summary[summary_ids], summary_scores)
            before = chunk_scores.sort_values(ascending=False).index[:100]
            after = with_summary.sort_values(ascending=False).index[:100]
            target_before, wide_before, _ = ranks(before, row, relevance)
            target_after, wide_after, _ = ranks(after, row, relevance)
            results.append({
                "test_id": row.test_id, "sorgu_tipi": row.sorgu_tipi,
                "sira_once": target_before, "sira_sonra": target_after,
                "sira_genis_once": wide_before, "sira_genis_sonra": wide_after,
                "hedef_parca_puan": round(chunk_scores[row.karar_id], 3),
                "hedef_ozet_puan": round(summary_scores[row.karar_id], 3),
            })
    results = pd.DataFrame(results)
    results.to_csv(OUT_DIR / "mini_b.csv", index=False, encoding="utf-8-sig")
    print(results.to_string(index=False))
    print("\nNOT: Yalnızca hedeflerin özeti var; özetli sıra hedeflere haksız avantaj verir. "
          "Bu bir boru hattı testi, kazanç ölçümü değil.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet(POOL_DIR / "kararlar.parquet")
    targets = (pd.read_csv(config.test_set_path("v2"), encoding="utf-8-sig", keep_default_na=False)
               .assign(test_id=lambda t: t.test_id.str[:3]).drop_duplicates("test_id"))
    targets = targets[targets.test_id.isin(TEST_IDS)][["test_id", "karar_id", "kunye"]]
    decisions = targets.merge(original[["id", "text"]], left_on="karar_id", right_on="id")
    summarize_all(decisions)
    mini_search(decisions, original)


if __name__ == "__main__":
    main()
