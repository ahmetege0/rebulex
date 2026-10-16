r"""magibu-11b özetlerindeki harf kaybının sebebini arar (4. deneme). Özet üretmez; sadece teşhis yazar.

3. denemede özetlerde parçalar düşüyordu: HUKUKİ -> "HUKİ", SONUÇ -> "SONÇ", otomasyon -> "omasyon".
Üç olası sebep ayrı ayrı denenir:
    1. Gidiş-dönüş: metin token'a çevrilip geri çevrilince aynı metin mi çıkıyor? (bozuksa tokenizer suçlu;
       model kararı da bozuk okuyor demektir). Hızlı (fast) ve yavaş (sentencepiece) tokenizer ayrı denenir.
    2. Üretim ayarları: generation_config'te yasaklı token listesi (suppress_tokens, bad_words_ids) var mı?
    3. Token dökümü: T21 bir kez özetlenir; üretilen her token tek başına metne çevrilir. Model "ot" token'ını
       üretip de metne boş mu çevriliyor (tokenizer), yoksa hiç mi üretmiyor (model / yasak liste)?

Çıktı (Colab'da Drive'a bağlı klasör):
    data/n2000/ozet/tani.txt

Kullanım:
    python diagnose_tokens.py
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline

import config
from summarize_llm import LLM, MAX_NEW_TOKENS, OUT_DIR, POOL_DIR, SEED, SYSTEM, TEMPERATURE, TEMPLATE, TOP_P

# 3. denemede bozulan kelimeler (doğru yazılışları)
WORDS = ["HUKUKİ", "SONUÇ", "BENİMSENEN", "İLKE", "otomasyon", "arızalı", "davacının", "Borçlar", "Emekli",
         "kararlarıyla", "inkar"]
TEST_ID = "T21"  # en çok bozulan özet (otomasyon -> omasyon)
DUMP_TOKENS = 150  # dökümü yazılacak ilk üretilen token sayısı

lines = []


def write(text=""):
    print(text)
    lines.append(str(text))


def roundtrip(tokenizer, name, text):
    """Metni token'a çevirip geri çevirir; farklı çıkan kelimeleri yazar."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    back = tokenizer.decode(ids)
    write(f"[{name}] {len(ids)} token | geri çeviri aynı mı: {back == text}")
    if back != text:
        diff = [(a, b) for a, b in zip(text.split(), back.split()) if a != b]
        write(f"  farklı kelime sayısı: {len(diff)} (kelime sayıları {len(text.split())} / {len(back.split())})")
        for a, b in diff[:15]:
            write(f"  {a!r} -> {b!r}")


def word_pieces(tokenizer, name):
    """Her kelimenin hangi token'lara bölündüğünü ve her token'ın tek başına neye çevrildiğini yazar."""
    write(f"[{name}] kelime -> token (tek tek geri çeviri)")
    for word in WORDS:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        pieces = [f"{i}:{tokenizer.convert_ids_to_tokens(i)!r}->{tokenizer.decode([i])!r}" for i in ids]
        write(f"  {word}: {' | '.join(pieces)}  => {tokenizer.decode(ids)!r}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet(POOL_DIR / "kararlar.parquet")
    target = (pd.read_csv(config.test_set_path("v2"), encoding="utf-8-sig", keep_default_na=False)
              .assign(test_id=lambda t: t.test_id.str[:3]).query("test_id == @TEST_ID").iloc[0])
    text = original.set_index("id").text[target.karar_id]

    write("=" * 30 + " 1. Gidiş-dönüş")
    fast = AutoTokenizer.from_pretrained(LLM)
    write(f"hızlı tokenizer: {type(fast).__name__} | len {len(fast)} | vocab_size {fast.vocab_size}")
    roundtrip(fast, "hızlı, şablon", TEMPLATE)
    roundtrip(fast, f"hızlı, {TEST_ID} kararı", text)
    word_pieces(fast, "hızlı")
    try:
        slow = AutoTokenizer.from_pretrained(LLM, use_fast=False)
        write(f"yavaş tokenizer: {type(slow).__name__} | len {len(slow)}")
        roundtrip(slow, "yavaş, şablon", TEMPLATE)
        roundtrip(slow, f"yavaş, {TEST_ID} kararı", text)
        word_pieces(slow, "yavaş")
    except Exception as e:  # yavaş tokenizer yoksa teşhis yine sürer
        write(f"yavaş tokenizer yüklenemedi: {type(e).__name__}: {e}")

    write("=" * 30 + " 2. Üretim ayarları")
    generator = pipeline("text-generation", model=LLM, device="cuda", dtype=torch.bfloat16)
    tokenizer, model = generator.tokenizer, generator.model
    write(f"model config vocab_size: {model.config.get_text_config().vocab_size} | "
          f"embedding satırı: {model.get_input_embeddings().weight.shape[0]}")
    write(f"generation_config: {model.generation_config.to_dict()}")

    write("=" * 30 + f" 3. Token dökümü ({TEST_ID}, 3. denemenin ayarlarıyla)")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEMPLATE.format(metin=text)}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]
    torch.manual_seed(SEED)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                temperature=TEMPERATURE, top_p=TOP_P)
    new_ids = output[0, input_tokens:].tolist()
    empty = [i for i in new_ids if tokenizer.decode([i]) == ""]
    write(f"üretilen token: {len(new_ids)} | tek başına boş metne çevrilen: {len(empty)} "
          f"(farklı id: {sorted(set(empty))[:30]})")
    write(f"birlikte çeviri:\n{tokenizer.decode(new_ids)}")
    write("ilk token'lar (sıra | id | token | tek başına çeviri):")
    for n, i in enumerate(new_ids[:DUMP_TOKENS]):
        write(f"  {n:3d} | {i:6d} | {tokenizer.convert_ids_to_tokens(i)!r} | {tokenizer.decode([i])!r}")

    (OUT_DIR / "tani.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
