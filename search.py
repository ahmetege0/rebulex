"""Vektör uzayında emsal karar arar.

Kullanım:
    python search.py "kira bedelinin ödenmemesi nedeniyle tahliye"
    python search.py "işe iade" --k 10 --source yargitay --min-year 2018
    python search.py            # etkileşimli mod
"""
import argparse
import textwrap

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

import config


def build_filter(source, min_year):
    must = []
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    if min_year:
        must.append(models.FieldCondition(key="year", range=models.Range(gte=min_year)))
    return models.Filter(must=must) if must else None


def load_model(device=None):
    """config.py'deki modeli yükler; okuyabileceği en fazla uzunluk MODEL_MAX_TOKENS ile sınırlanır."""
    model = SentenceTransformer(config.MODEL_NAME, device=device, trust_remote_code=config.TRUST_REMOTE_CODE)
    model.max_seq_length = config.MODEL_MAX_TOKENS
    return model


def search(model, client, collection, query, k=5, source=None, min_year=None):
    # sorgunun başına modelin beklediği önek eklenir (config.QUERY_PROMPT)
    vector = model.encode(query, prompt=config.QUERY_PROMPT, normalize_embeddings=True)
    hits = client.query_points(collection, query=vector.tolist(), limit=k * 5,
                               query_filter=build_filter(source, min_year), with_payload=True).points
    # Aynı karardan birden çok parça gelirse yalnızca en yüksek skorlusu kalır
    best = {}
    for hit in hits:
        best.setdefault(hit.payload["decision_id"], hit)
    return list(best.values())[:k]


def print_results(hits, max_chars):
    if not hits:
        print("Sonuç bulunamadı.")
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        print(f"\n{rank}. [{hit.score:.3f}] {p['header']}  (parça {p['chunk_index'] + 1}/{p['n_chunks']})")
        text = p["text"] if max_chars == 0 else textwrap.shorten(p["text"], max_chars)
        print(textwrap.indent(text, "   "))


def main():
    parser = argparse.ArgumentParser(description="Emsal karar arama")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--source", choices=list(config.SOURCE_SHARES))
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--run", default=config.RUN_NAME,
                        help="Hangi denemenin indeksinde aranacağı (örn. per_decision, sentence_c1024_o128)")
    parser.add_argument("--chars", type=int, default=1500,
                        help="Her sonuçta gösterilecek karakter sayısı (0 = eşleşen parçanın tamamı)")
    args = parser.parse_args()

    client = QdrantClient(path=str(config.QDRANT_PATH))
    collection = config.collection_name(args.run)
    if not client.collection_exists(collection):
        raise SystemExit(f"'{collection}' indeksi yok. Önce config.py'yi bu denemeye göre ayarlayıp "
                         f"build_index.py çalıştırın.")
    print(f"İndeks: {collection}")
    model = load_model()

    if args.query:
        print_results(search(model, client, collection, args.query, args.k, args.source, args.min_year), args.chars)
        return
    while query := input("\nSorgu (çıkmak için boş bırak): ").strip():
        print_results(search(model, client, collection, query, args.k, args.source, args.min_year), args.chars)


if __name__ == "__main__":
    main()
