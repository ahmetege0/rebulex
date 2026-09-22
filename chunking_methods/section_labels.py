"""Kararları başlıklarına göre ortak bölüm etiketlerine ayırır.

Etiketler:
    USTBILGI  künye, taraflar            MEVZUAT  ilgili kanun / Anayasa metinleri
    KONU      davanın / başvurunun konusu GEREKCE  mahkemenin değerlendirmesi
    OLAY      olaylar, alt derece süreci  HUKUM    hüküm fıkrası
    IDDIA     tarafların iddia/savunması  KARSIOY  karşı oy, farklı gerekçe
    GURULTU   başvuru süreci, ilk inceleme, tetkik hâkimi düşüncesi, imza satırları

Başlıklar, metindeki boşluklar tek boşluğa indirildikten sonra aranır (AYM metinlerinde
PDF kaynaklı satır kırıkları başlıkları böldüğü için). Kalıplar büyük harf duyarlıdır.
"""
import re

R = r"[IVX]{1,4}\s?[.-]\s?"  # Roma rakamlı başlık öneki: "IV- ", "II. "

# kaynak -> [(etiket, regex)]; eşleşmenin BAŞINDAN itibaren yeni bölüm başlar
HEADINGS = {
    "aym_bb": [
        ("KONU", R + r"BAŞVURUNUN (KONUSU|ÖZETİ)"),
        ("GURULTU", R + r"BAŞVURU SÜRECİ"),
        ("OLAY", R + r"OLAY(LAR)? VE OLGULAR"),
        ("MEVZUAT", R + r"İLGİLİ HUKUK"),
        ("MEVZUAT", r"\b[A-Z]\. İlgili (Hukuk|Mevzuat)"),
        ("GEREKCE", R + r"(İNCELEME VE GEREKÇE|DEĞERLENDİRME)"),
        ("IDDIA", r"\b([A-Z]|\d)\. Başvurucunun (İddiaları|Görüşleri)"),
        ("GEREKCE", r"\b([A-Z]|\d)\. Değerlendirme\b"),
        ("HUKUM", R + r"(HÜKÜM|GİDERİM)"),
    ],
    "aym_norm": [
        ("USTBILGI", r"(İPTAL DAVASINI AÇAN|İTİRAZ YOLUNA BAŞVURAN|İtirazda bulunan)"),
        ("USTBILGI", r"İSTEMDE BULUNAN"),
        ("KONU", r"(DAVANIN|İTİRAZIN|İtirazın|İSTEMİN) (KONUSU|konusu)"),
        ("OLAY", r"(OLAY|Olay) ?:|" + R + r"OLAY\b"),
        ("GEREKCE", r"(İNCELEME|İnceleme|Esasın incelenmesi) ?:"),
        ("HUKUM", r"\b(SONUÇ|Sonuç) ?:"),
        ("MEVZUAT", R + r"(İPTALİ|İTİRAZ KONUSU|DAVA KONUSU|İLGİLİ)[A-ZÇĞİÖŞÜ ]{0,30}(KANUN|YASA|KURAL|HÜKÜM|HUKUK)"),
        ("MEVZUAT", R + r"(DAYANILAN|İLGİLİ) ANAYASA KURALLARI"),
        ("GURULTU", R + r"İLK İNCELEME"),
        ("GEREKCE", R + r"ESASIN İNCELENMESİ"),
        ("OLAY", r"\b[A-Z]\. Anlam ve Kapsam"),
        ("IDDIA", r"\b[A-Z]\. (İptal Talebinin|İtirazın|Başvuru Kararının|İptal Taleplerinin) Gerekçe"),
        ("IDDIA", R + r"(İPTAL|İTİRAZ) (İSTEMİNİN|DAVASININ|NEDENLERİ|İSTEMİ)"),
        ("GEREKCE", r"\b([A-Z]\.|[IVX]+\s?[.-]) ?(Anayasa.ya Aykırılık Sorunu|ANAYASA.YA AYKIRILIK SORUNU)"),
        ("GURULTU", R + r"(İPTALİN DİĞER|YÜRÜRLÜĞÜN DURDURULMASI|UYGULANACAK KURAL|SINIRLAMA SORUNU)"),
        ("HUKUM", R + r"(HÜKÜM|SONUÇ)\b"),
    ],
    "danistay": [
        ("USTBILGI", r"(TEMYİZ EDEN|KARŞI TARAF|DAVACI|DAVALI|İSTEMDE BULUNAN)[A-ZÇĞİÖŞÜ() ]{0,20}:"),
        ("KONU", r"(İSTEMİN|DAVANIN) (KONUSU|ÖZETİ) ?:"),
        ("OLAY", r"(YARGILAMA SÜRECİ|MADDİ OLAY) ?:"),
        ("OLAY", r"İlk Derece Mahkemesi kararının özeti ?:"),
        ("IDDIA", r"(TEMYİZ EDENİN İDDİALARI|KARŞI TARAFIN SAVUNMASI|SAVUNMANIN ÖZETİ|DAVACININ İDDİALARI|DAVALININ SAVUNMASI) ?:"),
        ("GURULTU", r"(DANIŞTAY )?(TETKİK H[ÂA]K[İI]M[İI]|SAVCISI)[^:]{0,40}DÜŞÜNCESİ ?:"),
        ("GURULTU", r"TÜRK MİLLETİ ADINA"),
        ("GEREKCE", r"(HUKUKİ DEĞERLENDİRME|İNCELEME VE GEREKÇE|ESAS YÖNÜNDEN) ?:?"),
        ("MEVZUAT", r"İLGİLİ MEVZUAT ?:?"),
        ("HUKUM", r"(KARAR SONUCU|HÜKÜM) ?:"),
    ],
    "emsal": [
        ("IDDIA", r"(Davacı|DAVACI)( taraf)?( vekili)?( tarafından)?[^.]{0,40}(dilekçe|DİLEKÇE)"),
        ("IDDIA", r"(İDDİA|TALEP|CEVAP|SAVUNMA|İSTİNAF (BAŞVURU )?(SEBEPLERİ|NEDENLERİ|İSTEMİ))[A-ZÇĞİÖŞÜ ]{0,20}:"),
        ("IDDIA", r"(Davalı|DAVALI)( taraf)?( vekili)?[^.]{0,40}(cevap|CEVAP)"),
        ("OLAY", r"İLK DERECE MAHKEMESİ(NİN)? KARARI|İLK DERECE MAHKEMESİNCE"),
        ("MEVZUAT", r"İLGİLİ (MEVZUAT|HUKUK) ?:"),
        # çok kelimeli gerekçe başlıklarında iki nokta isteğe bağlı
        ("GEREKCE", r"(DELİLLER(İN)? VE GEREKÇE|DEĞERLENDİRME VE GEREKÇE|DELİLLERİN DEĞERLENDİRİLMESİ( VE GEREKÇE)?"
                    r"|İNCELEME VE GEREKÇE|İSTİNAF SEBEPLERİNİN DEĞERLENDİRİLMESİ)( VE SONUÇ)? ?:?"),
        # tek kelimeli başlıklar cümle içinde de büyük harfle geçebildiği için iki nokta şart
        ("GEREKCE", r"\b(GEREKÇE|DEĞERLENDİRME)( VE SONUÇ)? ?:"),
        ("USTBILGI", r"İNCELENEN KARARIN"),
        ("GEREKCE", r"\bDava[,;:] [^.]{5,250}?(ilişkin|dair)"),
        ("GEREKCE", r"(Tüm dosya kapsamı|Dosya kapsamı(nın)? (ve|incelen)|Toplanan (tüm )?deliller|Tarafların iddia ve savunmaları|Mahkememizce (yapılan|toplanan))"),
        ("GEREKCE", r"\bDELİLLER ?:"),
        ("HUKUM", r"\bH ?Ü ?K ?Ü ?M ?(:|-|(?=Yukarıda|Açıklanan|Gerekçesi|1 ?-))"),
    ],
    "yargitay": [
        ("IDDIA", R + r"(DAVA|CEVAP|İTİRAZ)\b"),
        ("IDDIA", r"TEBLİĞNAME (GÖRÜŞÜ|İÇERİĞİ)"),
        ("IDDIA", R + r"TEMYİZ SEBEPLERİ|\b[A-Z]\. Temyiz Sebepleri"),
        ("OLAY", R + r"(İLK DERECE MAHKEMESİ KARARI|İSTİNAF|OLAY VE OLGULAR|İLK DERECE)"),
        ("OLAY", R + r"TEMYİZ\b"),
        ("GEREKCE", R + r"(GEREKÇE|DEĞERLENDİRME|HUKUKİ NİTELENDİRME)"),
        ("GEREKCE", r"\b([A-Z]|\d)\. (Gerekçe|Uyuşmazlık Konusu|Değerlendirme)\b"),
        ("MEVZUAT", R + r"İLGİLİ HUKUK|\b([A-Z]|\d)\. İlgili Hukuk\b"),
        ("HUKUM", R + r"(KARAR|SONUÇ|HÜKÜM)\b"),
        ("HUKUM", r"\bSONUÇ ?:"),
    ],
}
# eşleşmenin SONUNDAN itibaren yeni bölüm başlatan kalıplar
AFTER_MARKS = {
    "danistay": [("GEREKCE", r"(gereği|Gereği|GEREĞİ) (görüşüldü|düşünüldü|GÖRÜŞÜLDÜ|DÜŞÜNÜLDÜ) ?:")],
    # emsalde "GEREĞİ DÜŞÜNÜLDÜ:" mahkemenin değerlendirmesinin başladığı yerdir
    "emsal": [("GEREKCE", r"GEREĞİ DÜŞÜNÜLDÜ ?:?")],
    # eski AYM kararlarında esas inceleme, İLK İNCELEME başlığının altında ilk usul cümlesinden sonra başlar
    "aym_norm": [("GEREKCE", R + r"İLK İNCELEME.{0,1500}?(karar verilmiştir\.|görüşülüp düşünüldü ?:)")],
}
# başlıksız gerekçe: IDDIA başladıktan sonra, ayrıca gerekçe başlığı yoksa ilk görülen bu ifadeler gerekçeyi başlatır
REASONING_STARTS = {
    "emsal": re.compile(r"\bDava [^.]{3,120}? davasıdır|İncelenen dosya kapsamı|İddia, savunma|Somut olayda"),
}
HEAD_ONLY = {"USTBILGI": 3000}  # üst bilgi kalıpları sadece metnin ilk 3000 karakterinde aranır

END_OF_VERDICT = re.compile(r"(karar verildi|usulen anlatıldı|okunup anlatıldı|KARAR VERİLDİ)\.?")
DISSENT = re.compile(r"(KARŞI ?OY|FARKLI GEREKÇE|AZLIK OYU|MUHALEFET)")
# başlıklı hüküm yoksa: son "... nedenlerle" ifadesi hükmün başıdır
VERDICT_INTRO = re.compile(r"(Açıklanan|Belirtilen|Yukarıda açıklanan|Yukarıda belirtilen|Bu) (nedenlerle|gerekçelerle)")
YARGITAY_FIELD = re.compile(r"^[A-ZÇĞİÖŞÜ .()]{3,45}:")
VERDICT_WORD = re.compile(r"(BOZULMASINA|ONANMASINA|DÜŞMESİNE|REDDİNE|KALDIRILMASINA|DÜZELTİLEREK|İADESİNE)")


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def _collapse(text):
    """norm() ile aynı metni üretir; ayrıca her karakterin orijinal metindeki konumunu döndürür."""
    parts, index = [], []
    for m in re.finditer(r"(\s+)|\S+", text):
        if m.group(1):
            if parts:  # baştaki boşluk atılır
                parts.append(" ")
                index.append(m.start())
        else:
            parts.append(m.group())
            index.extend(range(m.start(), m.end()))
    if parts and parts[-1] == " ":  # sondaki boşluk atılır
        parts.pop()
        index.pop()
    return "".join(parts), index


def _yargitay_prefix(text):
    """Yargıtay: ilk satır ve 'MAHKEMESİ : ...' gibi alan satırları üst bilgidir; kalan satırlar gövdedir.

    Gövde satırlarını ve gövdenin orijinal metinde başladığı konumu döndürür.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    n_head = 1
    while n_head < len(lines):
        line = lines[n_head].replace('"İçtihat Metni"', "").strip()
        if not (YARGITAY_FIELD.match(line) or line == ""):
            break
        n_head += 1
    offset, seen = 0, 0
    for raw in text.splitlines(keepends=True):
        if seen == n_head:
            break
        offset += len(raw)
        seen += bool(raw.strip())
    return lines[n_head:], offset


def segment(source, text):
    """Kararı [(etiket, metin), ...] listesine böler (metinler boşlukları tekleştirilmiş haldedir)."""
    return [(label, norm(text[start:end])) for label, start, end in segment_spans(source, text)]


def segment_spans(source, text):
    """Kararı [(etiket, başlangıç, bitiş), ...] listesine böler; konumlar orijinal metne göredir."""
    if source == "yargitay":
        rest, offset = _yargitay_prefix(text)
        body, index = _collapse(text[offset:])
        index = [i + offset for i in index]
        segments = [("USTBILGI", 0, offset)]
    else:
        body, index = _collapse(text)
        segments = []

    marks = []  # (konum, etiket, ifade_sonu_mu)
    for label, pattern in HEADINGS[source]:
        for m in re.finditer(pattern, body):
            if label in HEAD_ONLY and m.start() > HEAD_ONLY[label]:
                continue
            marks.append((m.start(), label, False))
    for label, pattern in AFTER_MARKS.get(source, []):
        for m in re.finditer(pattern, body):
            marks.append((m.end(), label, True))
    marks.sort()
    # aynı yerde birden fazla eşleşme varsa ilkini tut; ama başlık eşleşmesi, ifade sonu eşleşmesinden önceliklidir
    # ("GEREĞİ DÜŞÜNÜLDÜ: Davacı vekili dilekçesinde..." -> IDDIA kazanır)
    deduped = []
    for pos, label, after in marks:
        if deduped and pos - deduped[-1][0] < 3:
            if deduped[-1][2] and not after:
                deduped[-1] = (pos, label, after)
            continue
        deduped.append((pos, label, after))
    marks = [(pos, label) for pos, label, _ in deduped]

    if source in REASONING_STARTS:
        claims = [pos for pos, label in marks if label == "IDDIA"]
        verdicts = [pos for pos, label in marks if label == "HUKUM"]
        if claims and not any(label == "GEREKCE" and pos > claims[0] for pos, label in marks):
            m = REASONING_STARTS[source].search(body, claims[0])
            if m and (not verdicts or m.start() < verdicts[0]):
                marks.append((m.start(), "GEREKCE"))
                marks.sort()

    has_verdict = any(label == "HUKUM" for _, label in marks)
    if source == "yargitay" and not has_verdict:
        # başlıksız eski format: hüküm kelimesi geçen son paragraf hükümdür
        for j in range(len(rest) - 1, -1, -1):
            if VERDICT_WORD.search(rest[j]):
                marks.append((len(norm(" ".join(rest[:j]))) + (1 if j else 0), "HUKUM"))
                break
        marks.sort()
    elif source != "yargitay" and not has_verdict:
        intros = list(VERDICT_INTRO.finditer(body))
        if intros:
            marks.append((intros[-1].start(), "HUKUM"))
            marks.sort()

    # hükümden sonraki "karar verildi" -> imzalar (gürültü); sonrasındaki karşı oy başlığı -> KARSIOY
    verdict_positions = [pos for pos, label in marks if label == "HUKUM"]
    if verdict_positions:
        m = END_OF_VERDICT.search(body, verdict_positions[-1])
        if m and m.end() < len(body) - 5:
            marks = [(pos, label) for pos, label in marks if pos < m.end()]
            marks.append((m.end(), "GURULTU"))
            dissent = DISSENT.search(body, m.end())
            if dissent:
                marks.append((dissent.start(), "KARSIOY"))

    first_label = "GEREKCE" if source == "yargitay" else "USTBILGI"  # yargıtay gövdesi başlıksızsa gerekçedir
    if not marks or marks[0][0] > 0:
        marks.insert(0, (0, first_label))
    # arka arkaya gelen aynı etiketler tek bölüm olur
    marks = [mark for k, mark in enumerate(marks) if k == 0 or mark[1] != marks[k - 1][1]]
    for k, (pos, label) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(body)
        piece = body[pos:end]
        start, stop = pos + len(piece) - len(piece.lstrip()), end - (len(piece) - len(piece.rstrip()))
        if start < stop:
            segments.append((label, index[start], index[stop - 1] + 1))
    return segments
