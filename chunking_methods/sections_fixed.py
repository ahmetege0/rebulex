"""B2 - Bölüm bazlı parçalama: büyük gruplar sabit token pencerelerine bölünür."""
from chunking_methods.sections import chunk_by_sections, split_fixed


def chunk(text, tokenizer, max_tokens, overlap_tokens, source=None):
    return chunk_by_sections(text, tokenizer, max_tokens, overlap_tokens, source, split_fixed)
