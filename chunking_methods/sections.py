"""Bölüm bazlı parçalamanın ortak mantığı.

B1/B2/B3 yöntemleri (sections_structural / sections_fixed / sections_paragraph) yalnızca
max_tokens'ı aşan grupların nasıl bölündüğünde ayrılır.

1. Kararın tamamı max_tokens'a sığıyorsa tek parça kalır ("karar").
2. Bölümler etiketlenir (section_labels.py) ve iki gruba toplanır:
       dava_ozeti = KONU + OLAY + IDDIA (+ AYM Norm'da MEVZUAT: orada davanın konusudur)
       gerekce    = GEREKCE + KARSIOY + HUKUM
   Diğer MEVZUAT ve GURULTU bölümleri kendinden önceki bölümün grubuna katılır.
   USTBILGI kısaysa atılır (künye parça başlığında zaten var); uzunsa içerik karışmış olabilir, dava özetine girer.
3. MIN_GROUP_TOKENS'tan kısa grup diğer gruba eklenir.
4. max_tokens'ı aşan grup, yönteme göre verilen split fonksiyonuyla bölünür.

Her parça (metin, grup_adı) olarak döner.
"""
import re

from chunking_methods.common import pack_pieces
from chunking_methods.fixed_tokens import chunk as fixed_token_chunk
from chunking_methods.section_labels import norm, segment_spans

MIN_GROUP_TOKENS = 100
MAX_HEADER_TOKENS = 300
GROUP_OF = {"KONU": "dava_ozeti", "OLAY": "dava_ozeti", "IDDIA": "dava_ozeti",
            "GEREKCE": "gerekce", "KARSIOY": "gerekce", "HUKUM": "gerekce"}
FOLLOWERS = {"MEVZUAT", "GURULTU"}  # kendinden önceki bölümün grubuna katılır
AYM = {"aym_bb", "aym_norm"}

UP, LOW = "A-ZÇĞİÖŞÜ", "a-zçğıöşü"
# AYM: numaralı paragraf ("13. Başvurucu") veya harfli / Roma rakamlı alt başlık ("B. Değerlendirme", "IV- ESASIN")
# başlık kelimesi en az 3 harfli olmalı: "(... B. No: 2012/636)" atıfları başlık sayılmasın
AYM_BREAK = re.compile(rf"(?<= )(?:(\d{{1,3}})\. (?=[{UP}])|(?:[A-H]|[IVX]{{1,4}})[.-] (?=[{UP}][{LOW}]{{2}}|[{UP}]{{3}}))")
AYM_HEADING = re.compile(rf"^(?:[A-H]|[IVX]{{1,4}})[.-] (?:[{UP}][{LOW}]{{2}}|[{UP}]{{3}})")
# Diğer kaynaklar: "HUKUKİ DEĞERLENDİRME :" gibi büyük harfli etiket ya da tamamı büyük harfli kısa satır
PLAIN_HEADING = re.compile(rf"^(?:[{UP}][{UP} ]{{4,60}}(?: ?:|$)|[IVX]{{1,4}}[.-] ?[{UP}])")
# cümle sonu; ama kısaltma ve sayılardan ("E.", "s.", "58.") sonra bölünmez
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
ABBREVIATION_END = re.compile(rf"(?:^|[\s(])(?:\d{{1,4}}\.|(?:[{UP}{LOW}]{{1,3}}\.){{1,3}})$")  # "58.", "s.", "T.C."
JOINER_TOKENS = 1  # birimler "\n" ile birleştirilir; her birim için sınıra 1 token pay eklenir


def _count(tokenizer, texts):
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


def _pack(units, max_tokens, overlap_tokens):
    """Birimleri sınıra kadar doldurur; birleştirmede eklenen satır sonlarını da sayar."""
    units = [(text, n + JOINER_TOKENS) for text, n in units]
    return pack_pieces(units, max_tokens, overlap_tokens, joiner="\n")


# ---------------------------------------------------------------- paragraflar ve bloklar
def _aym_paragraphs(text):
    """AYM: boş satırlar PDF satır kırığıdır; paragraflar numaralarından ve alt başlıklardan bulunur."""
    body = norm(text)
    cuts, expected = [], None
    for m in AYM_BREAK.finditer(body):
        if not m.group(1):  # alt başlık
            cuts.append(m.start())
            continue
        n = int(m.group(1))
        # sıradaki numara ya da (ilk kez) ardından n+1 numaralı paragraf da gelen bir numara
        if n == expected or (expected is None and re.search(rf" {n + 1}\. [{UP}]", body[m.end():])):
            cuts.append(m.start())
            expected = n + 1
    bounds = [0] + cuts + [len(body)]
    return [body[a:b].strip() for a, b in zip(bounds, bounds[1:]) if body[a:b].strip()]


def _paragraphs(text, source):
    if source in AYM:
        return _aym_paragraphs(text)
    return [norm(p) for p in re.split(r"\n\s*\n", text) if p.strip()]


def _is_heading(paragraph, source):
    if source in AYM:
        return len(paragraph) < 250 and bool(AYM_HEADING.match(paragraph))
    return bool(PLAIN_HEADING.match(paragraph))


def _blocks(paragraphs, source):
    """Paragrafları alt başlık bloklarına toplar; her başlık yeni bir blok başlatır."""
    blocks = []
    for p in paragraphs:
        if not blocks or _is_heading(p, source):
            blocks.append([p])
        else:
            blocks[-1].append(p)
    return blocks


# ---------------------------------------------------------------- birimler
def _sentences(text):
    parts = [p.strip() for p in SENTENCE_END.split(text) if p.strip()]
    merged = []
    for p in parts:
        if merged and (ABBREVIATION_END.search(merged[-1]) or p[0].islower()):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def _units(paragraphs, tokenizer, max_tokens):
    """Paragrafları (metin, token) birimlerine çevirir: sığmayan paragraf cümlelere,
    sığmayan cümle token pencerelerine bölünür."""
    units = []
    for para, n in zip(paragraphs, _count(tokenizer, paragraphs)):
        if n <= max_tokens:
            units.append((para, n))
            continue
        sentences = _sentences(para)
        for sent, m in zip(sentences, _count(tokenizer, sentences)):
            if m <= max_tokens:
                units.append((sent, m))
                continue
            ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), max_tokens):
                window = ids[i:i + max_tokens]
                units.append((tokenizer.decode(window), len(window)))
    return units


# ---------------------------------------------------------------- büyük grubu bölme yöntemleri
def split_fixed(text, source, tokenizer, max_tokens, overlap_tokens):
    """B2: sabit token pencereleri."""
    return fixed_token_chunk(text, tokenizer, max_tokens, overlap_tokens)


def split_paragraph(text, source, tokenizer, max_tokens, overlap_tokens):
    """B3: paragraflar sınıra kadar doldurulur (alt başlıklara bakılmaz)."""
    units = _units(_paragraphs(text, source), tokenizer, max_tokens)
    return _pack(units, max_tokens, overlap_tokens)


def split_structural(text, source, tokenizer, max_tokens, overlap_tokens):
    """B1: sığan alt başlık blokları bütün kalır; sığmayan blok paragraflara iner. Birimler sınıra kadar doldurulur."""
    units = []
    for block in _blocks(_paragraphs(text, source), source):
        joined = "\n".join(block)
        n = _count(tokenizer, [joined])[0]
        if n <= max_tokens:
            units.append((joined, n))
        else:
            units.extend(_units(block, tokenizer, max_tokens))
    return _pack(units, max_tokens, overlap_tokens)


# ---------------------------------------------------------------- ana fonksiyon
def _groups(text, source, tokenizer):
    """Bölümleri {"dava_ozeti": metin, "gerekce": metin} gruplarına toplar."""
    parts = {"dava_ozeti": [], "gerekce": []}
    current, pending = None, []
    for label, start, end in segment_spans(source, text):
        piece = text[start:end]
        if label == "USTBILGI":
            if _count(tokenizer, [piece])[0] <= MAX_HEADER_TOKENS:
                continue
            group = "dava_ozeti"
        elif label == "MEVZUAT" and source == "aym_norm":
            group = "dava_ozeti"
        elif label in FOLLOWERS:
            if current is None:  # önünde grup yoksa, sonraki grubu bekler
                pending.append(piece)
                continue
            group = current
        else:
            group = GROUP_OF[label]
        parts[group].extend(pending + [piece])
        pending, current = [], group
    parts[current or "gerekce"].extend(pending)
    return {g: "\n\n".join(p) for g, p in parts.items() if p}


def chunk_by_sections(text, tokenizer, max_tokens, overlap_tokens, source, split):
    if _count(tokenizer, [text])[0] <= max_tokens:
        return [(text, "karar")]
    groups = _groups(text, source, tokenizer) or {"gerekce": text}
    sizes = {g: _count(tokenizer, [t])[0] for g, t in groups.items()}

    # kısa grup diğer gruba eklenir (sıra korunur: dava özeti önce)
    if len(groups) == 2 and min(sizes.values()) < MIN_GROUP_TOKENS:
        keep = max(sizes, key=sizes.get)
        groups = {keep: groups["dava_ozeti"] + "\n\n" + groups["gerekce"]}
        sizes = {keep: sizes["dava_ozeti"] + sizes["gerekce"]}

    chunks = []
    for group, group_text in groups.items():
        if sizes[group] <= max_tokens:
            chunks.append((group_text, group))
        else:
            chunks.extend((c, group) for c in split(group_text, source, tokenizer, max_tokens, overlap_tokens))
    return chunks
