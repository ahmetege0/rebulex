"""Cümle duyarlı parçalama: cümleler sırayla bir parçaya doldurulur, sınır aşılacaksa parça kapanır.

Yeni parçanın başına önceki parçanın son cümleleri (en fazla overlap_tokens) kopyalanır.
"""
import re

from chunking_methods.common import pack_pieces

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def split_sentences(text):
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _pieces(sentences, tokenizer, max_tokens, overlap_tokens):
    """(metin, token_sayısı) çiftleri; sınırı tek başına aşan cümleler token pencerelerine bölünür."""
    lengths = [len(ids) for ids in tokenizer(sentences, add_special_tokens=False)["input_ids"]]
    for sent, n in zip(sentences, lengths):
        if n <= max_tokens:
            yield sent, n
            continue
        ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
        for start in range(0, len(ids), max_tokens - overlap_tokens):
            window = ids[start:start + max_tokens]
            yield tokenizer.decode(window), len(window)


def chunk(text, tokenizer, max_tokens, overlap_tokens, source=None):
    sentences = split_sentences(text)
    if not sentences:
        return []
    pieces = _pieces(sentences, tokenizer, max_tokens, overlap_tokens)
    return pack_pieces(pieces, max_tokens, overlap_tokens, joiner=" ")
