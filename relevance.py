"""Test setindeki hedef karar yerine başka bir kararın da 'doğru' sayılıp sayılamayacağına karar verir.

Test setinde her sorgunun tek bir hedef kararı var. Havuz büyüdükçe aynı meseleyi işleyen başka kararlar
da üst sıralara çıkıyor ve tek hedefli ölçüm bunları yanlış sayıyor. Embedding modelinden bağımsız iki ölçü:

1. Kardeş karar: metinler neredeyse aynı (aynı dairenin seri kararları).
       metin ortaklığı = ortak 5 kelimelik öbek sayısı / kısa olan kararın öbek sayısı   >= SIBLING_THRESHOLD

2. Kanun ortaklığı: hedef kararın dayandığı kanun maddeleri adayda da geçiyor.
       her (kanun, madde) atıfının ağırlığı, havuzda ne kadar nadir olduğuna göre (IDF):
           ağırlık = min( log(N / atıfın geçtiği karar sayısı),  log(N / MIN_DOC_FREQ) )
       kanun ortaklığı = ortak atıfların ağırlığı / hedefin bütün atıflarının ağırlığı    >= LAW_THRESHOLD
   Her kararda geçen usul maddeleri (İYUK m.49 gibi) düşük, konuya özgü maddeler yüksek ağırlık alır.
   Tavan, sadece birkaç kararda geçen (çoğu okuma hatası ya da karara özgü) atıfların baskın olmasını önler.
   Ayrıca ortak atıflardan en az biri özgül olmalı (havuzdaki kararların en fazla %1'inde geçen); yoksa yalnızca
   genel usul maddelerine (İYUK m.45, m.49 ...) atıf yapan kararlar birbirine "%100 ortak" görünür.
   Hedef kararda hiç atıf yoksa bu ölçü karar veremez.
"""
import math
import re
from collections import Counter

SIBLING_THRESHOLD = 0.70
LAW_THRESHOLD = 0.60
MIN_DOC_FREQ = 5  # ağırlık tavanı: log(N / 5)
SPECIFIC_MAX_SHARE = 0.01  # havuzdaki kararların en fazla %1'inde geçen atıf "özgül" sayılır
SHINGLE_WORDS = 5

# kısaltma -> kanun numarası
ABBREVIATIONS = {
    "TCK": "5237", "CMK": "5271", "CMUK": "1412", "HMK": "6100", "HUMK": "1086", "TBK": "6098", "BK": "818",
    "TMK": "4721", "MK": "743", "İİK": "2004", "IIK": "2004", "İYUK": "2577", "IYUK": "2577", "TTK": "6102",
    "VUK": "213", "AY": "ANAYASA", "Anayasa": "ANAYASA",
}
ARTICLE = r"(?<!\d)(\d{1,3})(?!\d)"  # madde numarası başka bir sayının parçası olmamalı ("5219" -> 521 değil)
ORDINAL = r"(?:\.|['’]?\s?(?:inci|ıncı|nci|ncı|üncü|uncu|nüncü|ncu))?"
# "5237 sayılı TCK'nın 142/1-e", "6100 sayılı Hukuk Muhakemeleri Kanunu'nun 362 nci maddesi"
NUMBERED = re.compile(rf"\b(\d{{3,5}}) sayılı [^.;:\d]{{0,110}}?{ARTICLE}(?:/\d+)?{ORDINAL}\s?"
                      rf"(?:madde|md|-|/|\s[a-z]\s|fıkra|bend|\))", re.I)
# "TCK'nun 142/1-e", "HMK 303", "Anayasa'nın 35. maddesi"
ABBREVIATED = re.compile(rf"\b({'|'.join(ABBREVIATIONS)})(?:['’]?\s?n?[ıiuü]n|\.?nun|\.?nın)?\s?"
                         rf"(?:mad(?:de)?\.?\s?)?{ARTICLE}{ORDINAL}")
# listenin devamı: "141/1, 142/2-h" veya "141 ve 142"
LIST_ITEM = re.compile(rf"(?:/[\w\-]+)*\.?\s?(?:,|ve|ile)\s?{ARTICLE}")


def extract_citations(text):
    """Metindeki (kanun, madde) atıflarının kümesi; fıkra ve bent göz ardı edilir."""
    text = re.sub(r"\s+", " ", text)
    found = {(law, int(article)) for law, article in NUMBERED.findall(text)}
    for m in ABBREVIATED.finditer(text):
        law = ABBREVIATIONS[m[1]]
        found.add((law, int(m[2])))
        pos = m.end()
        while item := LIST_ITEM.match(text, pos):
            found.add((law, int(item[1])))
            pos = item.end()
    return found


def shingles(text):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


class Relevance:
    """Havuzdaki kararların atıflarını ve atıf ağırlıklarını bir kez hesaplar."""

    def __init__(self, texts):
        """texts: {karar_id: karar metni} — havuzun tamamı (ağırlıklar havuza göre hesaplanır)."""
        self.texts = texts
        self.citations = {decision: extract_citations(text) for decision, text in texts.items()}
        n = len(self.citations)
        doc_freq = Counter(c for cites in self.citations.values() for c in cites)
        cap = math.log(n / MIN_DOC_FREQ)
        self.weight = {c: min(math.log(n / k), cap) for c, k in doc_freq.items()}
        self.specific = {c for c, k in doc_freq.items() if k <= SPECIFIC_MAX_SHARE * n}
        self._shingles = {}

    def law_overlap(self, target, candidate):
        """Hedefin atıf ağırlığının adayda da bulunan payı (0-1); hedefte atıf yoksa None."""
        cites = self.citations[target]
        if not cites:
            return None
        common = cites & self.citations[candidate]
        return sum(self.weight[c] for c in common) / sum(self.weight[c] for c in cites)

    def text_overlap(self, target, candidate):
        """Ortak 5 kelimelik öbeklerin, kısa olan kararın öbeklerine oranı (0-1)."""
        a, b = self._get_shingles(target), self._get_shingles(candidate)
        return len(a & b) / max(1, min(len(a), len(b)))

    def match(self, target, candidate):
        """Aday, hedefin yerine doğru sayılır mı? 'kardes', 'kanun' veya None."""
        if self.text_overlap(target, candidate) >= SIBLING_THRESHOLD:
            return "kardes"
        overlap = self.law_overlap(target, candidate)
        shares_specific = bool(self.citations[target] & self.citations[candidate] & self.specific)
        if overlap is not None and overlap >= LAW_THRESHOLD and shares_specific:
            return "kanun"
        return None

    def _get_shingles(self, decision):
        if decision not in self._shingles:
            self._shingles[decision] = shingles(self.texts[decision])
        return self._shingles[decision]
