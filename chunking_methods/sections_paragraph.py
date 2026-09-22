"""B3 - Bölüm bazlı parçalama: büyük gruplar yalnızca paragraflardan bölünür (alt başlıklara bakılmaz)."""
from chunking_methods.sections import chunk_by_sections, split_paragraph


def chunk(text, tokenizer, max_tokens, overlap_tokens, source=None):
    return chunk_by_sections(text, tokenizer, max_tokens, overlap_tokens, source, split_paragraph)
