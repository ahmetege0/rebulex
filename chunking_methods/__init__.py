"""Parçalama yöntemleri.

Her yöntem dosyasında aynı imzalı tek bir fonksiyon bulunur:
    chunk(text, tokenizer, max_tokens, overlap_tokens, source=None) -> [parça, ...]
Parça bir metindir; bölüm bazlı yöntemlerde (metin, grup_adı) çiftidir.
source (kararın kaynağı: yargitay, emsal, ...) yalnızca bölüm bazlı yöntemlerde kullanılır.
Yöntem, dosya adıyla seçilir (config.py'deki CHUNK_METHOD veya build_index.py --method).
"""
from importlib import import_module


def get_chunker(method):
    return import_module(f"chunking_methods.{method}").chunk
