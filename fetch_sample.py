r"""Bucket'tan kaynak başına kotalarla karar örneklemi çeker -> data/n<boyut>/kararlar.parquet

Büyük havuz bir öncekini kapsar: 10 bin 2000'i, 100 bin 10 bini aynen alır, kaynak kotalarının eksiği yeni
kararlarla doldurulur. Böylece test setinin hedef kararları büyük havuzda da bulunur.

Doldurma:
    2000 / 10 bin: her kaynaktan rastgele birkaç parquet bloğu (row group) okunur, her bloktan rastgele kararlar.
    100 bin (ürün havuzu): her kaynakta en yeni kararlar (karar_tarihi), günümüzden geçmişe doğru.
Dosyaların tamamı indirilmez, yalnızca gereken bloklar okunur.

Kullanım:
    .venv\Scripts\python.exe fetch_sample.py                 # config.DATASET_SIZE
    .venv\Scripts\python.exe fetch_sample.py --size 100000
"""
import argparse
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


def newest_source(fs, source, quota, exclude):
    """Kaynaktan en yeni quota kadar kararı seçer; exclude içindeki kararlar seçilmez.
    Bloklar en yeni tarihlerinden başlayarak okunur; bir bloğun en yeni kararı, o ana kadar seçilenlerin
    en eskisinden de eskiyse kalan bloklarda daha yeni karar yoktur ve okuma durur."""
    groups = []  # (bloğun en yeni karar tarihi, dosya, blok)
    for path in sorted(fs.ls(f"{config.BUCKET_DATA}/{source}", detail=False)):
        with fs.open(path, "rb") as f:
            meta = pq.ParquetFile(f).metadata
        column = meta.schema.to_arrow_schema().get_field_index("karar_tarihi")
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(column).statistics
            groups.append((stats.max if stats and stats.has_min_max else "", path, rg))
    out = pd.DataFrame(columns=COLUMNS)
    for i, (newest, path, rg) in enumerate(sorted(groups, reverse=True), 1):
        if len(out) >= quota and newest < out.karar_tarihi.iloc[quota - 1]:
            break
        with fs.open(path, "rb") as f:
            df = pq.ParquetFile(f).read_row_group(rg, columns=COLUMNS).to_pandas()
        df = df[(df.text_len >= config.MIN_TEXT_LEN) & df.karar_tarihi.notna() & ~df.id.isin(exclude)]
        out = pd.concat([out, df]).drop_duplicates("id").sort_values("karar_tarihi", ascending=False)
        print(f"  [{i}/{len(groups)}] {path.rsplit('/', 1)[-1]} blok {rg}: {len(df)} karar, en yeni {newest}")
    out = out.head(quota)
    print(f"  {len(out)} karar, tarih aralığı {out.karar_tarihi.min()} - {out.karar_tarihi.max()}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Bucket'tan karar havuzu çeker")
    parser.add_argument("--size", type=int, default=config.DATASET_SIZE, choices=[2000, 10000, 100000])
    size = parser.parse_args().size
    base_size = {10000: 2000, 100000: 10000}.get(size)  # kapsanan önceki havuz
    base = pd.DataFrame(columns=COLUMNS)
    if base_size:
        base = pd.read_parquet(config.dataset_dir(base_size) / "kararlar.parquet")
        print(f"{base_size} kararlık havuz alındı ({len(base)} karar), üzerine eklenecek.")
    # 2000'lik havuz ilk çekimdeki tohumla aynı kalır; büyük havuzlar farklı bloklardan örneklensin diye başka tohum
    rng = random.Random(config.SEED if size == config.BASE_SIZE else config.SEED + size)
    fs = HfFileSystem()

    parts = [base]
    for source, quota in config.quotas(size).items():
        need = quota - int((base.source == source).sum())
        if need <= 0:
            continue
        print(f"{source}: {need} yeni karar seçiliyor")
        if size == 100000:
            parts.append(newest_source(fs, source, need, set(base.id)))
        else:
            parts.append(sample_source(fs, rng, source, need, set(base.id)))
    df = pd.concat(parts, ignore_index=True)
    assert df.id.is_unique
    path = config.dataset_dir(size) / "kararlar.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    print(f"\n{len(df)} karar -> {path}")
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
