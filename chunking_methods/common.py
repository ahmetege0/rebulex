"""Birden fazla parçalama yönteminin ortak kullandığı yardımcılar."""


def pack_pieces(pieces, max_tokens, overlap_tokens, joiner):
    """(metin, token_sayısı) birimlerini sırayla parçalara doldurur.

    Sıradaki birim max_tokens'ı aşacaksa parça kapanır. Yeni parçanın başına önceki
    parçanın son birimleri (toplamı en fazla overlap_tokens) kopyalanır.
    """
    chunks = []
    current, current_len = [], 0
    for piece, n in pieces:
        if current and current_len + n > max_tokens:
            chunks.append(joiner.join(p for p, _ in current))
            carry, carry_len = [], 0
            for p, m in reversed(current):
                if carry_len + m > overlap_tokens:
                    break
                carry.insert(0, (p, m))
                carry_len += m
            if carry_len + n > max_tokens:
                carry, carry_len = [], 0
            current, current_len = carry, carry_len
        current.append((piece, n))
        current_len += n
    if current:
        chunks.append(joiner.join(p for p, _ in current))
    return chunks
