"""B1 - Bölüm bazlı parçalama: büyük gruplar alt başlık -> paragraf -> cümle -> token sırasıyla bölünür."""
from chunking_methods.sections import chunk_by_sections, split_structural


def chunk(text, tokenizer, max_tokens, overlap_tokens, source=None):
    return chunk_by_sections(text, tokenizer, max_tokens, overlap_tokens, source, split_structural)
