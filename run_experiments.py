r"""Parçalama denemelerini sırayla kurar ve ölçer; her sonuç data/experiments.csv'ye bir satır ekler.

İndeksi zaten kurulmuş denemeler yeniden kurulmaz. Ölçümü zaten yapılmış denemeler atlanır;
yarıda kesilirse aynı komutla kaldığı yerden devam eder. --reevaluate ile ölçümler yeniden yapılır
(örneğin ölçü yöntemi değiştiğinde; indeksler yine yeniden kurulmaz).

Kullanım:
    .venv\Scripts\python.exe run_experiments.py                           # config.py'deki test seti
    .venv\Scripts\python.exe run_experiments.py --test-sets v2 dilekce_v1  # indeks bir kez kurulur, ikisi ölçülür
    .venv\Scripts\python.exe run_experiments.py --reevaluate
"""
import argparse

import config
from build_index import build
from evaluate import evaluate

# Havuz boyutuna göre deneme listesi: (yöntem, parça sınırı, örtüşme)
# 2000'lik havuzda tüm tablo ölçüldü; büyük havuzlarda yalnızca temsilci denemeler çalıştırılır.
REPRESENTATIVE = [
    ("per_decision", config.MODEL_MAX_TOKENS, 0),
    ("sentence", 1024, 128),
    ("recursive", 1024, 128),
    ("recursive", 2048, 256),
    ("sections_structural", 1024, 128),
]
FULL_GRID = [
    ("fixed_tokens", 512, 64),
    ("fixed_tokens", 1024, 128),
    ("fixed_tokens", 2048, 256),
    ("fixed_tokens", 1024, 0),
    ("fixed_tokens", 1024, 256),
    ("sentence", 512, 64),
    ("sentence", 1024, 128),
    ("sentence", 2048, 256),
    ("sentence", 1024, 0),
    ("sentence", 1024, 256),
    ("recursive", 512, 64),
    ("recursive", 1024, 128),
    ("recursive", 2048, 256),
    ("recursive", 1024, 0),
    ("recursive", 1024, 256),
    # bölüm bazlı: B1 yapıya göre, B2 sabit token, B3 paragraf
    ("sections_structural", 1024, 128),
    ("sections_fixed", 1024, 128),
    ("sections_paragraph", 1024, 128),
]
# Yalnızca 512 token okuyabilen modeller (cosmos_e5) için aynı yöntemler 512'lik parçalarla
SHORT_CONTEXT = [
    ("per_decision", config.MODEL_MAX_TOKENS, 0),
    ("sentence", 512, 64),
    ("recursive", 512, 64),
    ("sections_structural", 512, 64),
]
if config.DATASET_SIZE == config.BASE_SIZE:
    EXPERIMENTS = FULL_GRID
elif config.MODEL_MAX_TOKENS < 1024:
    EXPERIMENTS = SHORT_CONTEXT
else:
    EXPERIMENTS = REPRESENTATIVE


def main():
    parser = argparse.ArgumentParser(description="Parçalama denemelerini kurar ve ölçer")
    parser.add_argument("--reevaluate", action="store_true", help="Ölçülmüş denemeleri de yeniden ölç")
    parser.add_argument("--test-sets", nargs="+", default=[config.TEST_SET_VERSION],
                        help="Ölçülecek test setleri (örn. v2 dilekce_v1); varsayılan config.py'deki")
    args = parser.parse_args()

    for i, (method, chunk_tokens, overlap) in enumerate(EXPERIMENTS, 1):
        run = config.run_name(method, chunk_tokens, overlap)
        todo = [t for t in args.test_sets if args.reevaluate or not config.eval_path(run, t).exists()]
        if not todo:
            print(f"[{i}/{len(EXPERIMENTS)}] {run}: zaten ölçülmüş, atlanıyor")
            continue
        print(f"\n{'=' * 70}\n[{i}/{len(EXPERIMENTS)}] {run}  (test setleri: {', '.join(todo)})\n{'=' * 70}")
        if config.index_info_path(run).exists():
            print("İndeks zaten kurulu, yalnızca ölçülüyor.")
        else:
            build(method, chunk_tokens, overlap)
        for test_set in todo:
            evaluate(run, test_set)
    print(f"\nBitti. Karşılaştırma tablosu: {config.EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
