"""Karar bazlı parçalama: her karar tek parça olur.

Modelin okuyabileceğinden (config.MODEL_MAX_TOKENS) uzun kararlar, sonları kesilip
kaybolmasın diye bu uzunlukta ardışık parçalara bölünür (2000 kararın ~%3'ü).
"""
import config

TITLE_RESERVE = 64  # modele verilirken eklenen künye ve biçim (config.DOCUMENT_FORMAT) için ayrılan pay
LIMIT = config.MODEL_MAX_TOKENS - TITLE_RESERVE


def chunk(text, tokenizer, max_tokens=None, overlap_tokens=None, source=None):
    """max_tokens, overlap_tokens ve source bu yöntemde kullanılmaz; diğer yöntemlerle aynı imza için var."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= LIMIT:
        return [text]
    return [tokenizer.decode(ids[start:start + LIMIT]) for start in range(0, len(ids), LIMIT)]
