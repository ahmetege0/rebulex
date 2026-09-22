"""Projenin tüm ayarları tek yerde."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
# Veri klasörü; Colab'da başka bir yer gösterilebilir: os.environ["REBULEX_DATA"] = "/content/data"
DATA_DIR = Path(os.environ.get("REBULEX_DATA", ROOT / "data"))

# --- Karar havuzu ---
# Hangi havuzla çalışılacağı: 2000 veya 10000. Her havuzun dosyaları data/n<boyut>/ altında durur.
DATASET_SIZE = 10000
BASE_SIZE = 2000  # daha büyük havuzlar bu havuzu kapsar (test setinin hedef kararları onda)

BUCKET_DATA = "buckets/magibu/turkish-court-decisions-bucket/data"
SOURCE_SHARES = {  # kaynakların havuzdaki payı
    "yargitay": 0.40,
    "emsal": 0.25,
    "danistay": 0.20,
    "aym_bb": 0.10,
    "aym_norm": 0.05,
}
ROWS_PER_GROUP = 100  # her parquet bloğundan en fazla bu kadar karar alınır -> çeşitlilik
MIN_TEXT_LEN = 300  # bundan kısa metinler (boş/bozuk kayıtlar) atlanır
SEED = 42

SOURCE_LABELS = {
    "yargitay": "Yargıtay",
    "emsal": "UYAP Emsal",
    "danistay": "Danıştay",
    "aym_bb": "Anayasa Mahkemesi (Bireysel Başvuru)",
    "aym_norm": "Anayasa Mahkemesi (Norm Denetimi)",
}


def quotas(size):
    """Havuz boyutuna göre kaynak başına karar sayısı: 2000 -> yargitay 800, emsal 500, ..."""
    return {source: round(share * size) for source, share in SOURCE_SHARES.items()}


def dataset_dir(size=DATASET_SIZE):
    return DATA_DIR / f"n{size}"


DATASET_DIR = dataset_dir()
SAMPLE_PATH = DATASET_DIR / "kararlar.parquet"

# --- Embedding modeli ---
# Model değiştirmek için aşağıdaki altı satır birlikte değiştirilir. Her modelin dosyaları kendi klasöründe
# durur: data/n<boyut>/<MODEL_FOLDER>/ (chunks, embeddings, index_info, eval, qdrant).
#
# MODEL_FOLDER      MODEL_NAME                           MAX_TOKENS  QUERY_PROMPT  |  DOCUMENT_FORMAT
# magibu_embedding  magibu/embeddingmagibu-200m          8192  "task: search result | query: "  |  "title: {title} | text: {text}"
# embeddinggemma    google/embeddinggemma-300m           2048  "task: search result | query: "  |  "title: {title} | text: {text}"
#                   (Hugging Face'te lisans onayı ve token gerekir)
# cosmos_e5         ytu-ce-cosmos/turkish-e5-large       512   "Instruct: Given a Turkish search query, retrieve relevant passages written in Turkish that best answer the query\nQuery: "  |  "{title}\n{text}"
# mursit_large      newmindai/Mursit-Large-TR-Retrieval  2048  ""  |  "{title}\n{text}"
# bge_m3            BAAI/bge-m3                          8192  ""  |  "{title}\n{text}"
# qwen3_06b         Qwen/Qwen3-Embedding-0.6B            8192  "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"  |  "{title}\n{text}"
#                   (model 32K okuyabilir; bellek için 8192 ile sınırlı)
# gte_multi         Alibaba-NLP/gte-multilingual-base    8192  ""  |  "{title}\n{text}"   (TRUST_REMOTE_CODE = True)
MODEL_NAME = "magibu/embeddingmagibu-200m"
MODEL_FOLDER = "magibu_embedding"
MODEL_MAX_TOKENS = 8192  # modelin tek seferde okuyabildiği en fazla token
QUERY_PROMPT = "task: search result | query: "  # aranan metnin başına eklenir
DOCUMENT_FORMAT = "title: {title} | text: {text}"  # parçanın modele veriliş biçimi; title = karar künyesi
TRUST_REMOTE_CODE = False  # modelin kendi kodunu çalıştırmasına izin (yalnızca gte_multi için True)

MODEL_DIR = DATASET_DIR / MODEL_FOLDER  # bu modelin bu havuzdaki bütün dosyaları
QDRANT_PATH = MODEL_DIR / "qdrant"

# Varsayılan deneme (build_index.py / search.py / evaluate.py komut satırından değiştirilebilir)
# Parçalama yöntemi = chunking_methods klasöründeki dosya adı:
# "per_decision", "fixed_tokens", "recursive", "sentence", "sections_structural", "sections_fixed", "sections_paragraph"
CHUNK_METHOD = "recursive"
CHUNK_TOKENS = 1024                # bir parçanın en fazla token sayısı (per_decision'da kullanılmaz)
CHUNK_OVERLAP = CHUNK_TOKENS // 8  # örtüşme: 512 -> 64, 1024 -> 128 (per_decision'da kullanılmaz)


def batch_size(method, chunk_tokens):
    """Uzun metinler ekran kartında çok bellek ister; uzadıkça aynı anda daha az metin işlenir."""
    if method == "per_decision":
        return 2
    return 8 if chunk_tokens > 1024 else 16


def run_name(method=CHUNK_METHOD, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """Bir denemenin adı; dosya ve koleksiyon adlarında kullanılır, denemeler birbirinin üzerine yazmaz."""
    if method == "per_decision":
        return method
    return f"{method}_c{chunk_tokens}_o{overlap}"


RUN_NAME = run_name()
MODEL_SLUG = MODEL_NAME.split("/")[-1].replace("-", "_")


# --- Denemeye göre dosya ve koleksiyon adları (seçili havuzun, seçili modelin klasöründe) ---
def collection_name(run):
    return f"kararlar_{MODEL_SLUG}_{run}"


def chunks_path(run):
    return MODEL_DIR / "chunks" / f"{run}.parquet"


def index_info_path(run):
    return MODEL_DIR / "index_info" / f"{run}.json"


def embeddings_path(run):
    return MODEL_DIR / "embeddings" / f"{MODEL_SLUG}_{run}.npy"


def eval_path(run, test_set):
    return MODEL_DIR / "eval" / f"{run}_{test_set}.csv"


EXPERIMENTS_PATH = DATA_DIR / "experiments.csv"  # tüm havuzların ve modellerin denemeleri tek tabloda

# --- Test seti --- (make_test_set.py üretir; tüm havuzlar için ortak)
TEST_SET_VERSION = "dilekce_v1"  # v1: hukukçu + avukat dili (60), v2: + farklı olay (90), dilekce_v1: dilekçe -> karar (30)
TEST_SETS_DIR = DATA_DIR / "test_sets"


def test_set_path(version):
    return TEST_SETS_DIR / f"test_set_{version}.csv"


for folder in ["chunks", "embeddings", "index_info", "eval"]:
    (MODEL_DIR / folder).mkdir(parents=True, exist_ok=True)
TEST_SETS_DIR.mkdir(parents=True, exist_ok=True)
