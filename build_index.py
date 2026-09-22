r"""Örneklem kararları parçalar, embedding çıkarır ve Qdrant vektör veritabanına yazar.

Kullanım:
    .venv\Scripts\python.exe build_index.py                                     # config.py'deki deneme
    .venv\Scripts\python.exe build_index.py --method sentence --chunk-tokens 512
    .venv\Scripts\python.exe build_index.py --method recursive --chunk-tokens 1024 --overlap 0
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient, models

import config
from chunking_methods import get_chunker
from search import load_model


def _clean(value):
    """Eksik alanlar (NaN) None olur; aksi halde başlıkta 'nan' yazar ve JSON'a yazılamaz."""
    return None if pd.isna(value) else value


def decision_header(row):
    court, esas, karar = _clean(row.court), _clean(row.esas_no), _clean(row.karar_no)
    parts = [config.SOURCE_LABELS[row.source], court,
             esas and f"{esas} E.", karar and f"{karar} K.", _clean(row.karar_tarihi)]
    return " ".join(p for p in parts if p)


SECTION_TITLES = {"dava_ozeti": "Dava özeti", "gerekce": "Gerekçe"}  # bölüm bazlı yöntemlerde başlığa eklenir


def build_chunks(df, tokenizer, method, chunk_tokens, overlap):
    chunker = get_chunker(method)
    records = []
    for row in df.itertuples(index=False):
        header = decision_header(row)
        parts = chunker(row.text, tokenizer, chunk_tokens, overlap, source=row.source)
        for i, part in enumerate(parts):
            part, section = part if isinstance(part, tuple) else (part, None)
            records.append({
                "decision_id": row.id,
                "source": row.source,
                "court": _clean(row.court),
                "esas_no": _clean(row.esas_no),
                "karar_no": _clean(row.karar_no),
                "karar_tarihi": _clean(row.karar_tarihi),
                "year": int(row.year),
                "chunk_index": i,
                "n_chunks": len(parts),
                "header": header,
                "bolum": section,
                "text": part,
            })
    return records


def build(method=config.CHUNK_METHOD, chunk_tokens=config.CHUNK_TOKENS, overlap=None):
    """Bir denemenin indeksini kurar ve deneme adını döndürür. overlap verilmezse parça boyutunun 1/8'i."""
    if overlap is None:
        overlap = chunk_tokens // 8
    run = config.run_name(method, chunk_tokens, overlap)
    df = pd.read_parquet(config.SAMPLE_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    print(f"Deneme: {run}")
    print(f"Model: {config.MODEL_NAME} ({device}), boyut: {model.get_embedding_dimension()}")

    records = build_chunks(df, model.tokenizer, method, chunk_tokens, overlap)
    chunks = pd.DataFrame(records)
    chunks["tokens"] = [len(ids) for ids in model.tokenizer(chunks.text.tolist(), add_special_tokens=False)["input_ids"]]
    chunks.to_parquet(config.chunks_path(run), index=False)
    print(f"{len(df)} karar -> {len(chunks)} parça")
    print(chunks.groupby("source").agg(karar=("decision_id", "nunique"), parca=("text", "size"),
                                       ort_token=("tokens", "mean")).round(0).to_string())

    # Parça modele config.DOCUMENT_FORMAT biçiminde verilir; Magibu/EmbeddingGemma: "title: {başlık} | text: {metin}"
    # Bölüm bazlı yöntemlerde başlığa bölüm adı eklenir: "... 2022/3049 E. ... — Gerekçe"
    def title(r):
        name = SECTION_TITLES.get(r["bolum"])
        return f"{r['header']} — {name}" if name else r["header"]
    inputs = [config.DOCUMENT_FORMAT.format(title=title(r), text=r["text"]) for r in records]
    start = time.time()
    embeddings = model.encode(inputs, batch_size=config.batch_size(method, chunk_tokens),
                              normalize_embeddings=True, show_progress_bar=True,
                              convert_to_numpy=True).astype(np.float32)
    embed_seconds = time.time() - start
    print(f"Embedding süresi: {embed_seconds:.0f} sn, matris: {embeddings.shape}")
    np.save(config.embeddings_path(run), embeddings)

    # evaluate.py bu bilgileri experiments.csv'ye yazar
    per_decision = method == "per_decision"
    info = {
        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "deneme": run,
        "model": config.MODEL_NAME,
        "yontem": method,
        "parca_siniri": config.MODEL_MAX_TOKENS if per_decision else chunk_tokens,
        "ortusme": 0 if per_decision else overlap,
        "karar_sayisi": len(df),
        "parca_sayisi": len(chunks),
        "bolunen_karar": int((chunks.groupby("decision_id").size() > 1).sum()),
        "ort_parca_token": round(chunks.tokens.mean()),
        "en_uzun_parca_token": int(chunks.tokens.max()),
        "embedding_sn": round(embed_seconds),
    }
    config.index_info_path(run).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    collection = config.collection_name(run)
    client = QdrantClient(path=str(config.QDRANT_PATH))
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE),
    )
    client.upload_collection(collection, vectors=embeddings, payload=records, ids=list(range(len(records))))
    print(f"Qdrant koleksiyonu '{collection}': {client.count(collection).count} vektör")
    client.close()  # veritabanı kilidi bırakılsın; aynı çalıştırmada evaluate açabilsin
    return run


def main():
    parser = argparse.ArgumentParser(description="Parçalama + embedding + Qdrant indeksi")
    parser.add_argument("--method", default=config.CHUNK_METHOD)
    parser.add_argument("--chunk-tokens", type=int, default=config.CHUNK_TOKENS)
    parser.add_argument("--overlap", type=int, help="Verilmezse parça boyutunun 1/8'i")
    args = parser.parse_args()
    build(args.method, args.chunk_tokens, args.overlap)


if __name__ == "__main__":
    main()
