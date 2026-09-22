r"""Bucket'tan kaynak başına kotalarla rastgele karar örneklemi çeker -> data/n<boyut>/kararlar.parquet

Havuz boyutu config.DATASET_SIZE ile seçilir. BASE_SIZE (2000) havuzundan büyük havuzlar onu kapsar:
2000 karar aynen alınır, kaynak kotalarının eksiği yeni kararlarla doldurulur. Böylece test setinin
hedef kararları büyük havuzda da bulunur.

Dosyaların tamamı indirilmez: her kaynaktan rastgele birkaç parquet bloğu (row group) okunur
ve her bloktan rastgele kararlar seçilir.

Kullanım:
    .venv\Scripts\python.exe fetch_sample.py
"""
import math
import random

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

import config

COLUMNS = ["id", "source", "court", "esas_no", "karar_no", "karar_tarihi", "year", "text", "text_len"]


def list_row_groups(fs, source):
    groups = []
    for path in sorted(fs.ls(f"{config.BUCKET_DATA}/{source}", detail=False)):
        with fs.open(path, "rb") as f:
            n = pq.ParquetFile(f).metadata.num_row_groups
        groups += [(path, i) for i in range(n)]
    return groups


def sample_source(fs, rng, source, quota, exclude):
    """Kaynaktan quota kadar karar seçer; exclude içindeki kararlar (mevcut havuz) seçilmez."""
    groups = list_row_groups(fs, source)
    n_groups = min(len(groups), math.ceil(quota / config.ROWS_PER_GROUP))
    per_group = math.ceil(quota / n_groups)
    frames = []
    for i, (path, rg) in enumerate(rng.sample(groups, n_groups), 1):
        with fs.open(path, "rb") as f:
            df = pq.ParquetFile(f).read_row_group(rg, columns=COLUMNS).to_pandas()
        df = df[(df.text_len >= config.MIN_TEXT_LEN) & ~df.id.isin(exclude)]
        frames.append(df.sample(n=min(per_group, len(df)), random_state=rng.randrange(2**32)))
        print(f"  [{i}/{n_groups}] {path.rsplit('/', 1)[-1]} blok {rg}: {len(frames[-1])} karar")
    out = pd.concat(frames).drop_duplicates("id")
    return out.sample(n=min(quota, len(out)), random_state=config.SEED)


def main():
    size = config.DATASET_SIZE
    base = pd.DataFrame(columns=COLUMNS)
    if size > config.BASE_SIZE:
        base = pd.read_parquet(config.dataset_dir(config.BASE_SIZE) / "kararlar.parquet")
        print(f"{config.BASE_SIZE} kararlık havuz alındı ({len(base)} karar), üzerine eklenecek.")
    # 2000'lik havuz ilk çekimdeki tohumla aynı kalır; büyük havuzlar farklı bloklardan örneklensin diye başka tohum
    rng = random.Random(config.SEED if size == config.BASE_SIZE else config.SEED + size)
    fs = HfFileSystem()

    parts = [base]
    for source, quota in config.quotas(size).items():
        need = quota - int((base.source == source).sum())
        if need <= 0:
            continue
        print(f"{source}: {need} yeni karar örnekleniyor")
        parts.append(sample_source(fs, rng, source, need, set(base.id)))
    df = pd.concat(parts, ignore_index=True)
    assert df.id.is_unique
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SAMPLE_PATH, index=False)

    print(f"\n{len(df)} karar -> {config.SAMPLE_PATH}")
    summary = df.groupby("source").agg(
        karar=("id", "size"),
        daire_sayisi=("court", "nunique"),
        yil_min=("year", "min"),
        yil_max=("year", "max"),
        ort_karakter=("text_len", "mean"),
    )
    print(summary.round(0).to_string())


if __name__ == "__main__":
    main()
