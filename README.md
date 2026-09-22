# REBULEX — Emsal Karar Vektör Uzayı

Türk mahkeme kararlarından (Yargıtay, UYAP Emsal, Danıştay, AYM) örneklem alıp Magibu embedding
modeliyle anlamsal arama yapılabilen bir vektör uzayı oluşturur ve parçalama yöntemlerini test setiyle karşılaştırır.

## Akış

```
fetch_sample.py     bucket'tan karar havuzu      ->  data/n<boyut>/kararlar.parquet
build_index.py      parçalama + embedding        ->  data/n<boyut>/<model>/chunks, embeddings, index_info, qdrant
search.py           sorgu -> en yakın kararlar
evaluate.py         test seti ile ölçüm          ->  data/n<boyut>/<model>/eval/<deneme>_<test seti>.csv + data/experiments.csv
run_experiments.py  deneme listesini sırayla kurar ve ölçer
make_test_set.py    test setlerini üretir        ->  data/test_sets/test_set_v1.csv, v2.csv, dilekce_v1.csv
review_sections.py  bölüm etiketlemesi raporu    ->  data/n<boyut>/section_review.csv
```

## Klasör düzeni

```
data/
├── experiments.csv        bütün denemelerin tek tablosu (karar_sayisi: havuz, model: embedding modeli)
├── test_sets/             test setleri (tüm havuzlar ve modeller için ortak)
│   ├── dilekceler/        T01.txt ... T30.txt: hedef kararları arayan dilekçeler (dilekce_v1'in kaynağı)
│   └── hedef_kararlar.txt 30 hedef kararın tam metni (parquet'ten kopya, okumak için)
├── n2000/                 2000 kararlık havuz
│   ├── kararlar.parquet   kararlar  (kararlar.txt: okunabilir hali)
│   ├── magibu_embedding/  bu modelin dosyaları (her modelin kendi klasörü var: cosmos_e5, bge_m3 ...)
│   │   ├── chunks/        <deneme>.parquet         parçalar (modelin tokenizer'ına göre bölünür)
│   │   ├── embeddings/    <model>_<deneme>.npy     vektörler
│   │   ├── index_info/    <deneme>.json            kurulum bilgisi (experiments.csv'ye yazılır)
│   │   ├── eval/          <deneme>_<test seti>.csv sorgu sorgu sonuçlar
│   │   └── qdrant/        bu modelin bu havuzdaki vektör veritabanı
│   └── _eski/             ilk denemelerden kalan, artık kullanılmayan dosyalar
└── n10000/                10 bin kararlık havuz (aynı yapı)
```

Hangi havuzla çalışılacağını `config.py`'deki `DATASET_SIZE` belirler (2000 veya 10000).
10 binlik havuz 2000'lik havuzu kapsar; test setinin hedef kararları ikisinde de bulunur.

Hangi embedding modeliyle çalışılacağını `config.py`'deki model satırları belirler (`MODEL_NAME`, `MODEL_FOLDER`,
`MODEL_MAX_TOKENS`, `QUERY_PROMPT`, `DOCUMENT_FORMAT`, `TRUST_REMOTE_CODE`). Denenen modellerin doğru değerleri
aynı yerde yorum olarak yazılı; model değiştirirken altı satır birlikte değiştirilir.

| Dosya | Görevi |
|---|---|
| `config.py` | Tüm ayarlar: havuz boyutu, kaynak payları, model, varsayılan parçalama yöntemi, dosya yolları |
| `chunking_methods/` | Parçalama yöntemleri; her dosyada aynı imzalı `chunk()` fonksiyonu |
| `chunking_methods/section_labels.py` | Kararları bölümlere ayırır (USTBILGI, OLAY, GEREKCE, HUKUM ...) |

## Kurulum

```bash
python -m venv .venv --system-site-packages
.venv\Scripts\python.exe -m pip install -r requirements.txt
hf auth login
```

Bucket private olduğu için magibu org'una okuma yetkisi olan bir HF token gerekir.

## Çalıştırma

```bash
.venv\Scripts\python.exe fetch_sample.py
.venv\Scripts\python.exe run_experiments.py
.venv\Scripts\python.exe search.py "kira bedelinin ödenmemesi nedeniyle tahliye" --run recursive_c1024_o128
```

Arama seçenekleri: `--k 10` (sonuç sayısı), `--source yargitay` (kaynak filtresi), `--min-year 2020`,
`--chars 0` (eşleşen parçanın tamamı). Sorgu vermeden çalıştırılırsa etkileşimli moda geçer.

## Örneklem

Kaynak payları: Yargıtay %40, Emsal %25, Danıştay %20, AYM Bireysel Başvuru %10, AYM Norm Denetimi %5.
Her kaynaktan rastgele parquet blokları seçilir ve her bloktan en fazla 100 karar alınır. Dosyalar
tarih ve daireye göre gruplu olduğu için bu yöntem tek konuya yığılmayı önler; dosyaların tamamı
indirilmez, sadece seçilen bloklar okunur.

## Parçalama denemesi yapmak

`run_experiments.py` içindeki listeyi sırayla kurar ve ölçer (2000'lik havuzda tam tablo, büyük
havuzlarda temsilci denemeler). Tek bir deneme için:

```bash
.venv\Scripts\python.exe build_index.py --method sentence --chunk-tokens 1024 --overlap 128
.venv\Scripts\python.exe evaluate.py --run sentence_c1024_o128 --test-set v2
```

Deneme adı yöntemden türetilir: `per_decision`, `sentence_c1024_o128` gibi. Her deneme kendi dosyalarına
ve Qdrant koleksiyonuna yazılır; sonuç `data/experiments.csv`'ye yeni satır olarak eklenir.

Hangi test setiyle ölçüleceğini `config.py`'deki `TEST_SET_VERSION` belirler. Dilekçeden karar aramayı ölçmek için
`dilekce_v1` yapılıp `run_experiments.py` çalıştırılır; kurulu indeksler yeniden kurulmaz, yalnızca ölçülür.

Yeni yöntem eklemek için `chunking_methods/` içine `chunk(text, tokenizer, max_tokens, overlap_tokens, source=None)`
fonksiyonu olan bir dosya yazılır ve adı `--method`'a verilir.
