"""Özyinelemeli parçalama: metni önce en büyük doğal sınırdan böler, sığmayan bölümü bir alt sınırdan böler.

Sınır sırası: paragraf (boş satır) -> satır sonu -> cümle sonu -> kelime -> token.
max_tokens'a sığan bölümlere dokunulmaz. Elde edilen birimler sırayla max_tokens'a kadar
birleştirilir; parçalar arasına örtüşme eklenir (sentence yöntemindeki gibi).
"""
import re

from chunking_methods.common import pack_pieces

SEPARATORS = [r"\n\s*\n", r"\n", r"(?<=[.!?;:])\s+", r"\s+"]  # büyükten küçüğe


def _count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _split(text, tokenizer, max_tokens, level=0):
    """Metni, her biri max_tokens'a sığan (metin, token_sayısı) birimlerine ayırır."""
    n = _count(tokenizer, text)
    if n <= max_tokens:
        return [(text, n)]
    if level == len(SEPARATORS):  # tek kelime bile sığmıyorsa token pencerelerine böl
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        windows = [ids[i:i + max_tokens] for i in range(0, len(ids), max_tokens)]
        return [(tokenizer.decode(w), len(w)) for w in windows]
    # Ayırıcı yakalanıp önceki bölümün sonuna eklenir; böylece boş satırlar ve satır sonları kaybolmaz
    parts = re.split(f"({SEPARATORS[level]})", text)
    sections = [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
    units = []
    for section in sections:
        if section.strip():
            units.extend(_split(section, tokenizer, max_tokens, level + 1))
    return units


def chunk(text, tokenizer, max_tokens, overlap_tokens, source=None):
    units = _split(text, tokenizer, max_tokens)
    return [c.strip() for c in pack_pieces(units, max_tokens, overlap_tokens, joiner="")]
