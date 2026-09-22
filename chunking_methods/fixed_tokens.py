"""Sabit token parçalama: metne bakmadan her max_tokens tokende bir keser.

Pencereler overlap_tokens kadar üst üste biner (her pencere max_tokens - overlap_tokens kadar ileri kayar).
Yeni token getirmeyen, yani tamamen önceki pencerenin içinde kalan son pencere oluşturulmaz.
"""


def chunk(text, tokenizer, max_tokens, overlap_tokens, source=None):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [text]
    step = max_tokens - overlap_tokens
    starts = range(0, len(ids) - overlap_tokens, step)
    return [tokenizer.decode(ids[start:start + max_tokens]) for start in starts]
