r"""Karar metinlerinden konuyla ilgisi olmayan kısımları atar: şablon cümleler, künye, imza blokları,
isimler, şehir adları, tarihler, karar numaraları, atıflar.

Her kaynağın kendi metodu var (aym_bb, aym_norm, danistay, emsal, yargitay). Ortak işler yardımcı metotlarda.

    from cleaning import Cleaner
    temiz = Cleaner().clean("aym_bb", metin)

AYM BB'de atılanlar (3 şablon: uzun eski, uzun GİDERİM'li, kısa ÖZET/DEĞERLENDİRME):
    künye (I. başlığından öncesi), paragraf numaraları, BAŞVURU SÜRECİ bölümü, üç giriş kalıbı ("... olaylar özetle
    şöyledir:", "Mahkemenin ... toplantısında ... gereği düşünüldü:", standart Tahir Canan paragrafı), hükmün usule
    ilişkin maddeleri (yargılama gideri, faiz, Adalet Bakanlığına gönderme, adli yardım, birleştirme), hükmün kapanış
    tarihi, imza bloğu ve ekli tablolar. GİDERİM, "Uluslararası Hukuk" ve başvurunun özeti olduğu gibi kalır.
AYM Norm'da atılanlar:
    künye ve davayı açan (konu etiketinden öncesi), İLK İNCELEME ve ESASIN İNCELENMESİ açılış cümleleri,
    YÜRÜRLÜĞÜN DURDURULMASI TALEBİ ve YÜRÜRLÜĞE GİRECEĞİ GÜN SORUNU bölümleri, hükmün kapanış cümlesi,
    imza ve üye listeleri. Eski (1962-2011) ve yeni şablonun kalıpları ayrı listelerde, ikisi de uygulanır.
Her ikisinde tutulanlar: başlıklar, hüküm sonuçları, incelenen kanun metinleri, para tutarları, karşı oy metinleri.

Çıktıda boşluklar tekleştirilir; paragraflar ve başlıklar boş satırla ayrılır.

Kullanım (2000'lik havuzun AYM kararlarını temizler, özet ve örnekleri yazar):
    .venv\Scripts\python.exe cleaning.py
"""
import re

UP, LOW = "A-ZÇĞİÖŞÜ", "a-zçğıöşü"
ROMAN = r"\b[IVX]{1,4}[.-] "  # "II. ", "IV- "
ROMAN_HEADING = rf"{ROMAN}(?=[{UP}]{{3}})"  # "II. BAŞVURU SÜRECİ"
LETTER_HEADING = rf"\b[A-H]\. (?=[{UP}][{LOW}]{{2}})"  # "B. Uluslararası Hukuk" ("B. No:" başlık sayılmaz)
PARAGRAPH = "¶"  # paragraf sınırı; en sonda boş satıra çevrilir

# 81 il; "Ordu" ordu kelimesiyle karıştığı için listede yok. İlçeler atılmıyor.
CITIES = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Aksaray", "Amasya", "Ankara", "Antalya", "Ardahan", "Artvin",
    "Aydın", "Balıkesir", "Bartın", "Batman", "Bayburt", "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa",
    "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Düzce", "Edirne", "Elazığ", "Erzincan", "Erzurum",
    "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", "Hatay", "Iğdır", "Isparta", "İstanbul", "İzmir",
    "Kahramanmaraş", "Karabük", "Karaman", "Kars", "Kastamonu", "Kayseri", "Kırıkkale", "Kırklareli", "Kırşehir",
    "Kilis", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir",
    "Niğde", "Osmaniye", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Şanlıurfa", "Şırnak", "Tekirdağ",
    "Tokat", "Trabzon", "Tunceli", "Uşak", "Van", "Yalova", "Yozgat", "Zonguldak",
]
CITY = re.compile(rf"\b(?:{'|'.join(CITIES)})(?:['’][{LOW}]+)?\b(?: [İi]li\b)?")  # "Kocaeli ili" birlikte gider

# "Hasan Tahsin GÖKCAN", "M. Emin KUZ": büyük harfli soyadlı isimler.
# Soyad sanılan büyük harfli kelimeler: hüküm kelimeleri (GÖNDERİLMESİNE, OLDUĞUNA: -NA/-NE ile biter) ve kısaltmalar.
# Kısaltmalar soyad sanılmasın ("Davanın HMK'nun ..."): soyadında en az bir ünlü olmalı (HMK, TBK, KDV elenir);
# ünlü içeren kısaltmalar ayrıca listede.
ACRONYMS = ("AİHM|AİHS|GGM|İİK|IIK|AAÜT|AYM|YÖK|ÖTV|TOKİ|UYAP|ABD|AB|İYUK|ÖSYM|TÜİK|EPDK|RTÜK|MASAK|TÜBİTAK|SPK"
            "|BDDK|TMSF|TCMB|TBMM|HSYK|TRT|TSE|PTT|MEB|BAM|KHK|HUMK|TL|USD|EUR")
# Hüküm kelimeleri de soyad sanılmasın: "Davanın KABULÜ", "talebin ESASTAN REDDİNE"
RULING_WORDS = (r"KABUL\w*|RED\w*|İPTAL\w*|TAHSİL\w*|ONAN\w*|BOZUL\w*|KALDIR\w*|İADE\w*|TESPİT\w*|ESASTAN|USULDEN"
                r"|KISMEN|OYBİRLİĞİ\w*|OYÇOKLUĞU\w*|BIRAK\w*|GÖNDER\w*|DEVAM\w*|AÇILMAMIŞ|SAYILMA\w*|DÜŞME\w*")
NAME = re.compile(rf"\b(?:[{UP}]\. )?[{UP}][{LOW}]+(?: [{UP}][{LOW}]+)* (?!(?:{ACRONYMS}|{RULING_WORDS})\b)"
                  rf"(?=[{UP}]*[AEIİOÖUÜ])[{UP}]{{2,}}(?<!NA)(?<!NE)\b")
INITIALS = re.compile(rf"\b(?:[{UP}]\.){{2,4}}")  # "İ.M.G."

# Tarih, ardından gelen tarih kelimesiyle birlikte atılır; yoksa metinde boşta "günlü", "tarihinde" kalır:
# "17/7/2004 tarihli ve", "8/3/1950 günlü,", "20/10/2004 tarihinde", "12/3/2015'ten", "2 Temmuz 1974 Salı günü"
MONTHS = "Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık"
_DAY_MONTH_YEAR = (rf"(?:\b\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}\b"  # 18/6/2020, 28.06.2014
                   rf"|(?:\b\d{{1,2}}[ .])?(?:{MONTHS})[ .]\d{{4}}\b(?: [{UP}][{LOW}]+(?= gün))?)")  # 2 Temmuz 1974 Salı, Kasım 1956
_DATE_WORD = r"(?:tarihli|günlü)\b(?: ve\b|,)?|(?:tarih(?:inde|ine|inden|i|te|ten)|gün(?:ü|ünde|üne|ünden|kü))\b"
DATE = re.compile(rf"{_DAY_MONTH_YEAR}(?:['’][{LOW}]+)?(?: (?:{_DATE_WORD}))?")
# Değer + etiket: bir tarih, sayı ya da "..." yer tutucusu, ardından gelen etiket kelimeleriyle birlikte atılır; yoksa
# metinde boşta "tarih ve sayılı", "ili, Mahallesi, ada ve parsel sayılı" kalır.
# "27/01/2010 tarih ve 27475 sayılı", "... tarih ve E:..., K:... sayılı", "... ada ve ... parsel sayılı", "... ili,"
# yer tutucu ".." / "..." / "…" yazılır; değer ile etiket arasında boşluk olmayabilir: "...tarih", "K:...sayılı"
# "++": nokta dizisi bir kez bütün olarak alınır, geri dönülüp başka türlü bölünmez; yoksa 100 noktalık bir dizide
# (karardaki doldurma çizgileri) regex dakikalarca takılır. Silinen metin değişmez.
_VALUE = (r"(?:(?:\.{2,}|…)++|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d[\d./-]*\b"
          r"|\bE ?[.:] ?(?:\.{2,}|…|\d+/\d+),? ?K ?[.:]? ?(?:\.{2,}|…|\d+/\d+))")
_LABEL = (r"(?:tarih(?:li)?|günlü|tarih(?:inde|ine|inden|i|te|ten)|gün(?:ü|ünde|kü)|sayılı|numaralı|nolu|no'lu|plakalı"
          r"|ada|parsel|pafta|ili|İli|ilçesi|Mahallesi|mahallesi|Köyü|köyü|Sokak|Caddesi|Bulvarı)")
VALUE_LABEL = re.compile(rf"{_VALUE}(?:['’][{LOW}]+)? ?{_LABEL}\b(?: {_LABEL}\b)*(?:,? ve,?|,)?")
CASE_NUMBER = re.compile(r"(?:\b(?:E|K)[.:] ?)?\b\d+/\d+(?:-\d+)?\b")  # "2017/37601", "E.2008/686", "2011/1-156"
LAW_NUMBER = re.compile(r"\b\d+ sayılı ")  # "5237 sayılı Türk Ceza Kanunu" -> "Türk Ceza Kanunu"
# "36. maddesinde", "125'inci maddesi", "13., 35. ve 46. maddeleri", "(1) numaralı fıkrası", "3/18-son cümle"
_ARTICLE = rf"(?:\b\d+(?:/\d+)*(?:-[{LOW}]+)?(?:\.|['’][{LOW}]+)?|\(\d+\) numaralı)"
ARTICLE_NUMBER = re.compile(rf"(?:{_ARTICLE}(?:, | ve | ile ))*{_ARTICLE} (?=(?:madde|fıkra|bent|bend|cümle))")
SECTION_SIGN = re.compile(r"§§? ?\d+(?: ?[-–,] ?\d+)*")  # "§ 16", "§§ 92-96"
# parantez içi atıf: "(Tahir Canan, B. No: 2012/969, 18/9/2013, § 16)", "(AYM, E.2019/12, K.2020/5, 4/3/2020)";
# bir kat iç parantez olabilir: "(3) [GK]"
CITATION = re.compile(r"\((?:[^()]|\([^()]*\))*?(?:B\. ?No|§|\[GK\]|\bE\. ?\d+/\d+)(?:[^()]|\([^()]*\))*\)")
BARE_CITATION = re.compile(r"\[GK\]|\bB\. ?No ?:")  # parantez dışında kalan atıf parçaları

# Cümle başı: ". : ;" ve boşluktan ya da paragraf işaretinden sonrası. "[^.]*?..." ile başlayan kalıplar buna sabitlenir;
# yoksa metnin her harfinden başlanarak denenir ve çok yavaşlar.
_SENTENCE = r"(?:(?<=[.:;] )|(?<=¶)|^)"

# AYM BB şablon cümleleri (dava ve olay hakkında bilgi taşımaz). Başvurunun özetindeki süreç cümlelerine
# ("nihai kararı ... öğrendikten sonra ...", "Komisyon; ... karar vermiştir") dokunulmaz: kısa şablonda özet bilgi taşıyor.
AYM_BB_TEMPLATES = [
    rf"{_SENTENCE}[^.:¶]*?olaylar özetle şöyledir:",
    r"(?:Anayasa )?Mahkemenin \S+ tarihinde yapmış olduğu toplantıda[^:]*?gereği düşünüldü:",
    r"Anayasa Mahkemesi, olayların başvurucu tarafından yapılan hukuki nitelendirmesi ile bağlı olmayıp"
    r"(?:[^()]|\((?:[^()]|\([^()]*\))*\))*?\.",
    r"Açıklanan gerekçelerle;",
]
# hükmün usule ilişkin maddeleri; sonuç maddeleri (KABUL EDİLEMEZ, İHLAL EDİLDİĞİNE, tazminat ...) tutulur
AYM_BB_PROCEDURAL_ITEMS = re.compile(
    r"[Yy]argılama gider|(?:FAİZ|[Ff]aiz) UYGULANMASINA|Adalet Bakanlığına GÖNDERİLMESİNE|[Aa]dli yardım talep"
    r"|BİRLEŞTİRİLMESİNE")
# Sonuç işaretleri. Harfsiz hükümde sonuç ve gider aynı maddede olabilir ("Başvurunun ... KABUL EDİLEMEZ OLDUĞUNA,
# yargılama giderlerinin başvurucu üzerinde bırakılmasına,"): o zaman yalnızca giderin geçtiği virgüllü parça atılır.
AYM_BB_OUTCOME = re.compile(r"KABUL EDİL|İHLAL EDİL|REDDİNE|[Tt]azminat|TAZMİNAT|yeniden yargılama|DÜŞMESİNE")
# Hükmün kapanışı: "18/6/2020 tarihinde OYBİRLİĞİYLE karar verildi.", eski kararlarda "29/11/1977 gününde oybirliği ile
# karar verildi." Yalnızca tarih kısmı atılır; "OYBİRLİĞİYLE karar verildi." kalır ki hüküm cümlesi yarım kalmasın.
# Tarih açıkça aranır: "\S+" olsaydı boşluksuz yazımda önceki kelimeyi de alırdı ("ONANMASINA,05/12/2012 gününde").
VERDICT_CLOSING = re.compile(rf"{_DAY_MONTH_YEAR} (?:tarihinde|gününde),? (?P<kapanis>(?:(?:OY ?BİRLİĞİYLE|OY ?ÇOKLUĞUYLA"
                             r"|[Oo]y ?birliği(?:yle| ile)|[Oo]y ?çokluğu(?:yla| ile)) )?karar verildi\.)")
DISSENT = re.compile(r"\b(?:KARŞI ?OY|FARKLI GEREKÇE|MUHALEFET ŞERHİ)\w*")
SIGNATURE = re.compile(r"\b(?:Başkan|Başkanvekili|Üye|Üyeler|Raportör(?: Yrd\.)?|Başvurucu|Başvurucular|Vekili|Vekilleri)"
                       r"(?: ?:| (?=[A-ZÇĞİÖŞÜ][a-zçğıöşü]+ [A-ZÇĞİÖŞÜ]{2,}))")

# AYM Norm şablon cümleleri. İki liste de her karara uygulanır: eski kalıplar yeni kararlarda, yeni kalıplar eski
# kararlarda geçmediği için kendi dönemlerinin dışında hiçbir şeye dokunmaz (2012-2024 geçiş kararlarında ikisi de olabilir).
# Eski kararlarda paragraf numarası olmadığı için paragraf işareti de yok; kalıplar en fazla 800 karakter uzanır ki
# bir kalıp bulunamadığında araya giren içerik silinmesin.
# ilk inceleme cümlesinin bitişleri: "... incelenmesine OYBİRLİĞİYLE karar verilmiştir.", "... sorunu görüşülmüştür.",
# "... üzerinde durulmuştur.", "... okunup incelendikten sonra gereği görüşülüp düşünüldü:"
_FIRST_REVIEW_END = (r"(?:karar veril(?:miştir|di)|karara bağlanmıştır|görüşülmüştür|üzerinde dur(?:ul)?muştur"
                     r"|görüşülüp düşünüldü)[.:]")
AYM_NORM_NEW_TEMPLATES = [  # 2012 sonrası
    # "Anayasa Mahkemesi İçtüzüğü hükümleri uyarınca <15 üye>'nın katılımlarıyla ... ilk inceleme toplantısında ..."
    rf"Anayasa Mahkemesi İçtüzüğü hükümleri uyarınca[^¶]{{0,800}}?ilk inceleme toplantısında[^¶]{{0,800}}?{_FIRST_REVIEW_END}",
    r"Anayasa.nın 152\.[^¶]{0,300}?bir davaya bakmakta olan mahkeme[^¶]{0,600}?başvurmaya yetkilidir\.",
    # "Dava dilekçesi ve ekleri, Raportör <ad> tarafından hazırlanan işin esasına ilişkin rapor, ... düşünüldü:"
    r"(?:Dava dilekçe(?:si|leri)|Başvuru karar(?:ı|ları))[^¶]{0,300}?işin esasına ilişkin rapor[^¶]{0,600}?"
    r"görüşülüp düşünüldü ?:",
]
AYM_NORM_OLD_TEMPLATES = [  # 1962-2011
    # "Anayasa Mahkemesi İçtüzüğü'nün 8. maddesi uyarınca <üyeler>'ün katılmalarıyla ... gününde yapılan ilk inceleme
    # toplantısında ... karar verilmiştir." ("İçtüzüğünün 15 inci maddesi gereğince ..." yazılışı da var)
    r"(?:Anayasa Mahkemesi )?[İi]çtüzüğ\w*['’]?\w* \d+[\w.'’ ]{0,6}maddesi (?:uyarınca|gereğince)[^¶]{0,800}?"
    rf"ilk inceleme[^¶]{{0,800}}?{_FIRST_REVIEW_END}",
    r"İşin esasına ilişkin rapor[^¶]{0,600}?görüşülüp düşünüldü ?[:.]",
    r"\b(?:İşin )?[Gg]ereği görüşülüp düşünüldü ?[:.]",
]
# davanın konusu etiketi; ondan öncesi künye ve davayı açan (mahkeme ya da milletvekilleri)
# eski kararlarda yazılış çok değişir: "İtirazın Konusu", "iptal davasının konusu", "İptal dâvasının konusu",
# "İTİRAZLARIN KONUSU", "İstemin Konusu", "A) Davanın Konusu"
AYM_NORM_SUBJECT = re.compile(r"(?:İTİRAZ(?:LAR)?IN|İPTAL DAVASININ|İSTEMİN|DAVANIN) KONUSU"
                              r"|(?:[İi]tiraz(?:lar)?ın|[İi]tiraz|[İi]ptal d[aâ]vasının|[İi]stemin|D[aâ]vanın|D[aâ]va)"
                              r" [Kk]onusu ?:")
# konu etiketi olmayan kararlarda (iade, ilk inceleme kararları) yalnızca künye alanları atılır:
# "ANAYASA MAHKEMESİ KARARI Esas Sayısı : ... Karar Sayısı : ... Karar Günü : ... R.G. Tarih-Sayı : ..."
AYM_NORM_HEADER = re.compile(r"^ANAYASA MAHKEMESİ KARARI ?(?:(?:Esas|Karar) (?:[Ss]ayısı|No\.?|[Gg]ünü|[Tt]arihi) ?: ?\S+ ?)+"
                             r"(?:(?:R\.G\. ?Tarih ?[-–] ?Sayı|Resmi Gazete tarih/sayı) ?: ?"
                             r"(?:R\.G\.'de yayımlanmamıştır\.|\S+))?")
AYM_NORM_VERDICT = re.compile(rf"{ROMAN}(?:HÜKÜM|SONUÇ)\b|\bHÜKÜM\b(?! ?[{UP}]{{3}})|\b(?:SONUÇ|Sonuç) ?:")
# Hükümde karşı oy verenlerin isimleri: "Yıldız SEFERİNOĞLU, Selahaddin MENTEŞ ile Kenan YAŞAR'ın karşıoyları ve",
# "Üyelerden Rifat Göksu ve İbrahim Senil'in muhalefetleriyle ve". Oy bilgisi (OYÇOKLUĞUYLA) kalır.
# İsim kelimesi -NA/-NE ile bitmez: "İPTALİNE, Yıldız ..." gibi hüküm kelimeleri isim listesine katılmasın.
_NAME_WORD = rf"[{UP}][\w.]*(?<!NA)(?<!NE)"
_NAMES = rf"{_NAME_WORD}(?: {_NAME_WORD})*"
AYM_NORM_VOTES = re.compile(rf"{_NAMES}(?:(?:, | ve | ile ){_NAMES})*['’]\w+ (?:karşı ?oy\w*|muhalefet\w*)(?: ile)?")
# İmza listesi: "Başkan Kadir ÖZKAYA Başkanvekili Hasan Tahsin GÖKCAN Üye ..."; eski kararlarda soyadlar büyük harfli
# değil ("Üye Muhittin Gürün"). Ünvanın ardından en fazla 4 büyük harfle başlayan kelime (bir sonraki ünvan hariç).
_TITLE = r"(?:Başkan|Başkanvekili|Üye)"
NORM_SIGNATURE = re.compile(rf"\b{_TITLE}(?: (?!{_TITLE}\b|KARŞI|MUHALEFET)[{UP}][\w.’']*){{1,4}}")
NORM_SIGNATURE_LIST = re.compile(rf"(?:{NORM_SIGNATURE.pattern} ?){{2,}}")  # arka arkaya en az iki imza
# karşı oy yazılarının başında künye tekrarı: "KARŞIOY YAZISI Esas Sayısı: 1977/109 Karar Sayısı: 1977/131"
NORM_DISSENT_HEADER = re.compile(r"Esas (?:Sayısı|No\.?) ?: ?\S* Karar (?:Sayısı|No\.?) ?: ?\S*")

# Danıştay metinleri anonimleştirilmiş: isimler, mahkemeler, tarihler "..." / "…" ile gizli ("...'IN", "…-TL").
PLACEHOLDER = re.compile(rf"(?:\.{{3,}}|…)(?:['’][{UP}{LOW}]+|-?TL)?")
# yer tutucular silinince kalan künye artıkları: "... tarih ve E:..., K:... sayılı" -> "tarih ve E:, K: sayılı"
REFERENCE_LEFTOVER = re.compile(r"\b(?:(?:tarih|tarihli|günlü),? ve,? )?(?:E ?: ?, ?K ?: ?)?sayılı\b(?= (?:karar|işlem|ilam))"
                                r"|\b(?:tarih|tarihli|günlü),? ve,? (?=E ?:)|\bE ?: ?, ?K\b ?:?")  # "E:..., K:..." ve "E:..., K..."
# kararın asıl metni bu etiketlerden biriyle başlar; öncesi künye ve taraf satırları (TEMYİZ EDEN, KARŞI TARAF, vekiller)
DANISTAY_START = re.compile(r"İSTEMİN[ _](?:KONUSU|ÖZETİ)|İstemin Özeti|DAVANIN[ _]KONUSU|Dava Konusu İstem"
                            r"|YARGILAMA[ _]SÜRECİ")  # bazı kararlarda etiket alt çizgili: "İSTEMİN_ÖZETİ:"
# başlık etiketleri: önlerine paragraf işareti konur (Danıştay kararlarında paragraf numarası yok)
DANISTAY_HEADINGS = re.compile(
    r" (?=İSTEMİN (?:KONUSU|ÖZETİ)|İstemin Özeti|DAVANIN KONUSU|Dava Konusu İstem|YARGILAMA SÜRECİ|Dava konusu istem ?:"
    r"|İlk Derece Mahkemesi kararının özeti|Bölge İdare Mahkemesi kararının özeti|Daire kararının özeti"
    r"|TEMYİZ EDEN(?:İN|LERİN) İDDİALARI|Temyiz İsteminde Bulunan [^:]{0,30}İddialarının Özeti"
    r"|KARŞI TARAFIN (?:SAVUNMASI|CEVABI)|Davalı İdarenin Savunmasının Özeti|SAVUNMANIN ÖZETİ"
    r"|HUKUKİ DEĞERLENDİRME|İNCELEME VE GEREKÇE|İLGİLİ MEVZUAT|MADDİ OLAY|KARAR SONUCU|\(X\) KARŞI OY|KARŞI OY ?:)")
# Tetkik hâkiminin ve Danıştay savcısının düşünceleri, İSTEMİN KONUSU, 49. madde cümlesi ve onama gerekçe cümleleri
# ("Temyizen incelenen karar usul ve hukuka uygun olup ...") kalır.
DANISTAY_TEMPLATES = [
    # "TÜRK MİLLETİ ADINA Karar veren Danıştay ... Dairesince, Tetkik Hâkiminin açıklamaları dinlendikten ve dosyadaki
    # belgeler incelendikten sonra gereği görüşüldü:" cümlesinin yalnızca sabit parçaları; araya giren ifadeler
    # ("adli yardım talebi hakkında karar verilmesine yer olmadığına", "Anayasa'ya aykırılık iddiası ciddi görülmeyerek")
    # kalır.
    r"TÜRK MİLLETİ ADINA",
    r"(?:Karar|Hüküm) veren Danıştay(?: \S+){1,3}? (?:Dairesi|Daireleri Kurulu)\S*,? ?",
    r"Tetkik H[âa]kim(?:i|inin) açıklamaları dinlendikten (?:ve dosyadaki (?:bilgi ve )?belgeler incelendikten )?sonra,? ?",
    r"(?:işin )?gereği görüşüldü ?:",
    rf"{_SENTENCE}[^.¶]{{0,60}}?(?:[Ss]avunma|[Cc]evap) verilmemiştir\.",
    r"Açıklanan (?:nedenlerle|gerekçelerle)[;,]?",
]
# Karar sonucunun usule ilişkin parçaları; yeni formatta numaralı maddeler, eski formatta tek cümlenin virgüllü kısımları.
# Her kalıp yalnızca bir madde ya da virgüllü kısım başında başlar (önünde ". , ; : ¶" var); hem hızlı olur hem de
# cümle ortasından başlayıp önceki içeriği silmez.
# Eski formatta sonuç ve gider "ve" ile bağlı olabilir: "... anılan kararın ONANMASINA ve temyiz giderlerinin istemde
# bulunan üzerinde bırakılmasına,". Bu yüzden kısım "ve"den sonra da başlayabilir ve sonuç kelimesi içeren bir kısmı
# hiçbir kalıp silmez: yalnızca "temyiz giderlerinin ... bırakılmasına" gider, ONANMASINA kalır.
_CLAUSE = r"(?:(?<=[.,;:] )|(?<=¶)|(?<= ve ))(?:\d+[.-] )?"
_NO_OUTCOME = (r"(?:(?!ONANMASINA|onanmasına|BOZULMASINA|bozulmasına|REDDİNE|reddine|reddi\b|İPTALİNE|iptaline"
               r"|KABULÜNE|kabulüne|kabulü\b)[^,;¶])*?")
DANISTAY_PROCEDURAL = [
    rf"{_CLAUSE}{_NO_OUTCOME}gider\w* {_NO_OUTCOME}(?:üzerinde bırakılmasına|alınarak \S+ verilmesine)[,;]?",
    rf"{_CLAUSE}{_NO_OUTCOME}(?:posta (?:gideri|ücreti)|harcının){_NO_OUTCOME}iadesine[,;]?",
    rf"{_CLAUSE}{_NO_OUTCOME}Avukatlık Asgari Ücret Tarifesi{_NO_OUTCOME}verilmesine[,;]?",
    rf"{_CLAUSE}{_NO_OUTCOME}yolu açık olmak üzere,?",  # "Bu kararın tebliğ tarihini izleyen ... temyiz yolu açık olmak üzere"
    # "kesin olarak" yalnızca kapanışta: "oybirliğiyle kesin olarak karar verildi", "Kesin olarak, oybirliği ile karar
    # verildi" (tarih bu noktada zaten silinmiş olur). Gerekçedeki "... dönem sonunda kesin olarak belirlenebileceği" kalır.
    r"\b[Kk]esin olarak,?\s+(?=(?:(?:oy ?birliği|oy ?çokluğu)\S*(?: ile)?\s+)?karar verildi)",
]

# Emsal (UYAP: Asliye Ticaret ve Bölge Adliye Mahkemesi hukuk daireleri). Metinlerde gerçek satır sonları var; giriş
# kısmı satır satır atılır: yukarıdan başlayıp aşağıdaki satırlar sürdükçe hepsi künye / taraf / heyet sayılır.
_FIELD = (r"T\.C\.|ESAS NO|KARAR NO|Esas No|Karar No|DOSYA NO|HAKİM|KATİP|BAŞKAN|ÜYE|Başkan|Üye|Katip|Hakim"
          r"|DAVACI\w*|DAVALI\w*|VEKİL\w*|Vekil\w*|İHBAR OLUNAN\w*|MÜDAHİL\w*|FER'?İ MÜDAHİL\w*|BORÇLU\w*|ALACAKLI\w*"
          r"|DAVA TARİHİ|KARAR TARİHİ|[A-ZÇĞİÖŞÜ. ]*YAZ\w* TARİHİ|İNCELENEN KARARIN|MAHKEMESİ|TARİHİ|NUMARASI"
          r"|İSTİNAF KARARININ|İSTİNAF KARAR TARİHİ|DAVA|DAVANIN KONUSU")
EMSAL_HEADER_LINE = re.compile(
    rf"(?:{_FIELD})\s*:.{{0,120}}"                      # künye ve taraf alanları: "ESAS NO : ...", "DAVALI : ... A.Ş."
    rf"|[{UP}ÎÂÛ0-9 .,()/'’…:-]*"                        # tamamı büyük harf: "T.C.", "ANKARA", "GEREKÇELİ KARAR"
    r"|.{0,60}\b(?:MAHKEMESİ|DAİRESİ)\b.{0,60}"          # "T.C. İstanbul Anadolu 3. ASLİYE TİCARET MAHKEMESİ"
    r"|(?:\d+[-.]\s*)?(?:Av\.\s*)?[\s.…\-\[\]/0-9]*(?:\w+\s*)?(?:UETS)?"  # taraf devamı: "2- ... - ...", "Av. ... UETS"
    r"|.*Esas-Karar No.*")                               # "T.C. ... MAHKEMESİ Esas-Karar No: 2022/806 Esas - 2023/359"
EMSAL_CASE_TYPE = re.compile(r"^\s*(?:DAVA|DAVANIN KONUSU)\s*:\s*(.{3,120}?)\s*$", re.M)  # "DAVA : Alacak (...)"
# giriş kalıbı: "Mahkememizde görülmekte olan ... davasının yapılan açık yargılaması sonunda, GEREĞİ DÜŞÜNÜLDÜ:";
# BAM: "Yukarıda yazılı ilk derece mahkemesi kararına karşı ... inceleme / müzakere sonucunda GEREĞİ DÜŞÜNÜLDÜ:"
EMSAL_OPENING = re.compile(r"^(?:Mahkememiz\w* (?:görülmekte|açılan)[^¶]{0,300}?(?:sonunda|sonucunda)\s*[,;:]?)?"
                           r"[\s¶]*(?:GEREĞİ DÜŞÜNÜLDÜ\s*[:;]?)?")
EMSAL_OPENING_ANYWHERE = re.compile(r"(?:(?<=¶)|^)[^¶]{0,300}?(?:yargılama|inceleme|müzakere)\w* (?:sonunda|sonucunda)"
                                    r"\s*[,;:]?[\s¶]*GEREĞİ DÜŞÜNÜLDÜ\s*[:;]?|GEREĞİ DÜŞÜNÜLDÜ\s*[:;]?")
EMSAL_PLACEHOLDER = re.compile(r"-{3,}|\*{3,}")  # Emsal'in anonimleştirmesi: "-----", "***"
# BAM giriş kalıbı: "Taraf vekilleri tarafından süresinde verilen dilekçeler ile ... HMK'nın 355. maddesi gereğince ...
# sınırlı olarak yapılan inceleme sonunda, gereği düşünüldü;"
EMSAL_BAM_OPENING = re.compile(r"^[^¶]{0,900}?[Gg]ereği (?:düşünüldü|görüşüldü)\s*[:;]?")
EMSAL_SPACED = re.compile(rf"\b(?:[{UP}] ){{3,}}[{UP}]\b")  # "H Ü K Ü M" -> "HÜKÜM"
EMSAL_VERDICT = re.compile(r"(?:^|¶)\s*(?:[IVX]+\.\s*)?(?:HÜKÜM|KARAR)\b\s*:?")
# hükmün usule ilişkin maddeleri: harç, vekâlet ücreti, yargılama gideri, gider avansı, tebligat
EMSAL_PROCEDURAL_ITEMS = re.compile(r"[Hh]arç|[Vv]ek[âa]let ücret|[Yy]argılama gider|[Gg]ider avans|gelir kayd|[Tt]ebligat"
                                    r"|[Pp]osta|Avukatlık Asgari|AAÜT|ücreti vek[âa]let|[Mm]asraf")
# kapanış: "Dair, ... istinaf yolu açık olmak üzere ... okunup usulen anlatıldı."; yalnızca "(oybirliği ile) karar verildi." kalır
# sonuç maddesi işaretleri: bunlardan biri geçen maddeye usul kuralı dokunmaz
# ("... 4.055,41-TL masraf olmak üzere toplam ... alacağın tahsili bakımından takibin devamına")
EMSAL_OUTCOME = re.compile(r"KABUL|REDD|İPTAL|TAHSİL|SAYILMASINA|DEVAMINA|TESPİT|ONAN|KALDIRIL|DÜŞMESİNE"
                           r"|takibin devamına|[Dd]avanın (?:kabul|red|kısmen)")
EMSAL_CLOSING = re.compile(r"\bDair\b|yolu (?:açık|kabil) olmak üzere|usulen anlatıldı|okunup anlatıldı")
EMSAL_DECIDED = re.compile(r"(?:(?:oy ?birliği(?:yle| ile)|oy ?çokluğu(?:yla| ile)) )?karar verildi", re.I)

# Yargıtay: yalnızca kesin kalıplar atılır. MAHKEMESİ / SUÇ / HÜKÜM / TEBLİĞNAME GÖRÜŞÜ / DAVA TÜRÜ alanları, ceza ön
# inceleme paragrafı, hukuk açılışındaki dava konusu, standart gerekçe cümleleri ve gönderme / tevdi maddeleri kalır.
YARGITAY_TEMPLATES = [
    r"^[^¶]*?\d+/\d+ E\. ?, ?\d+/\d+ K\.",  # künye satırı: "3. Ceza Dairesi 2013/30306 E. , 2014/23007 K."
    r'"?İçtihat Metni"?',
    r"(?:(?<=¶)|^)(?:SAYISI|TARİHİ|NUMARASI) ?:[^¶]*",  # "SAYISI : 2016/187 E., 2016/329 K."
    r"Y ?A ?R ?G ?I ?T ?A ?Y K ?A ?R ?A ?R ?I",
    r"(?:Yerel|Mahall[iî]) [Mm]ahkemece verilen hüküm(?:ler)? temyiz edilmekle(?:,? evrak okunarak)?;?",
    r"Temyiz isteğinin reddi nedenleri bulunmadığından işin esasına geçildi\.",
    r"kesinlik, süre, temyiz şartı ve diğer usul eksiklikleri yönünden yapılan ön inceleme sonucunda,? temyiz "
    r"dilekçesinin kabulüne karar verildikten (?:sonra )?ve (?:Tetkik H[âa]kimi [^¶]{0,40}?tarafından hazırlanan "
    r"rapor dinlendikten sonra )?dosyadaki belgeler incelenip",
    r"(?:Türk Milleti adına )?[Gg]ereği (?:görüşülüp )?düşünüldü ?[:;]?",
    r"Açıklanan sebeple(?:rle)?;",
    rf"{_CLAUSE}{_NO_OUTCOME}[Hh]arc\w*{_NO_OUTCOME}(?:alınmasına|iadesine)[,;]?",  # harç maddeleri
]


class Cleaner:
    def clean(self, source, text):
        """Kaynağın temizleme metodunu çağırır; metodu olmayan kaynakların metni olduğu gibi döner."""
        method = getattr(self, source, None)
        return method(text) if method else text

    # ------------------------------------------------------------------ AYM Bireysel Başvuru
    def aym_bb(self, text):
        text = re.sub(r"\s+", " ", text).strip()  # eski metinlerde cümle ortasında bile boş satır var
        first = re.search(rf"\bI[.-] (?=[{UP}]{{3}})", text)  # künye: "I. BAŞVURUNUN KONUSU"ndan öncesi
        if first:
            text = text[first.start():]
        text = self.mark_paragraphs(text)

        body, verdict, tail = self.split_verdict(text, re.compile(ROMAN + "HÜKÜM"))
        # GİDERİM ve "Uluslararası Hukuk" bölümleri kalır (gerekçe içeriyor; içlerindeki atıflar ortak adımda gider)
        body = self.drop_section(body, ROMAN + "BAŞVURU SÜRECİ")
        for pattern in AYM_BB_TEMPLATES:
            body = re.sub(pattern, "", body)
        verdict = self.clean_verdict(verdict)
        tail = self.keep_dissents(tail, SIGNATURE)
        return self.finish(body + PARAGRAPH + verdict + PARAGRAPH + tail)

    # ------------------------------------------------------------------ AYM Norm Denetimi
    def aym_norm(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        subject = AYM_NORM_SUBJECT.search(text, 0, 3000)  # künye ve davayı açan: konu etiketinden öncesi
        text = text[subject.start():] if subject else AYM_NORM_HEADER.sub("", text)
        text = self.mark_paragraphs(text)

        # YÜRÜRLÜĞÜN DURDURULMASI ve YÜRÜRLÜĞE GİRECEĞİ GÜN bölümleri kalır (talebin ve ertelemenin gerekçesi)
        body, verdict, tail = self.split_verdict(text, AYM_NORM_VERDICT)
        for pattern in AYM_NORM_NEW_TEMPLATES + AYM_NORM_OLD_TEMPLATES:
            body = re.sub(pattern, "", body)
        tail = self.keep_dissents(tail, NORM_SIGNATURE)
        text = body + PARAGRAPH + verdict + PARAGRAPH + tail
        # Eski kararların bir kısmında hüküm başlığı yok (hüküm gövdenin sonunda kalır); bu yüzden kapanışın tarihi,
        # karşı oy veren isimleri ve imza listeleri metnin tamamında aranır.
        text = VERDICT_CLOSING.sub(r"\g<kapanis>", text)
        for pattern in [AYM_NORM_VOTES, NORM_SIGNATURE_LIST, NORM_DISSENT_HEADER]:
            text = pattern.sub("", text)
        return self.finish(text)

    # ------------------------------------------------------------------ Danıştay
    def danistay(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        text = VALUE_LABEL.sub("", text)  # yer tutucular etiketleriyle birlikte gitsin diye önce: "... tarih ve ... sayılı"
        text = PLACEHOLDER.sub("", text)
        text = REFERENCE_LEFTOVER.sub("", text)
        start = DANISTAY_START.search(text, 0, 3000)  # künye ve taraf satırları: ilk içerik etiketinden öncesi
        if not start:  # etiketsiz kararlar (Kurul kararları gibi): yalnızca künye, "Karar No : ..."ya kadar
            start = re.search(r"Karar No ?: ?\S+ ?", text[:800])
            text = text[start.end():] if start else text
        else:
            text = text[start.start():]
        text = DANISTAY_HEADINGS.sub(PARAGRAPH, text)
        for pattern in DANISTAY_TEMPLATES + DANISTAY_PROCEDURAL:
            text = re.sub(pattern, "", text)
        text = VERDICT_CLOSING.sub(r"\g<kapanis>", text)  # tarih gider, "oybirliği ile karar verildi." kalır
        return self.finish(text)

    # ------------------------------------------------------------------ Emsal
    def emsal(self, text):
        """Yalnızca giriş (künye, taraflar, heyet, açılış kalıbı) ve çıkış (hükmün usul maddeleri, kanun yolu, imzalar)
        atılır; dava türü satırı ("DAVA : Alacak (...)") kararın konusunu söylediği için metnin başında kalır."""
        text = EMSAL_SPACED.sub(lambda m: m[0].replace(" ", ""), text)
        case_type = EMSAL_CASE_TYPE.search(text)
        lines = text.splitlines()
        first = next((i for i, line in enumerate(lines) if not EMSAL_HEADER_LINE.fullmatch(line.strip())), len(lines))
        # e-imza notları; yalnızca kısa satırlar: uzun satır, e-imzanın konu olduğu bir içerik paragrafı olabilir
        lines = [line for line in lines[first:] if len(line) > 150 or not re.search(r"e-?[İi]mza|[Ee]lektronik imza|✍", line)]
        text = PARAGRAPH.join(re.sub(r"\s+", " ", line).strip() for line in lines if line.strip())
        text = EMSAL_OPENING.sub("", text, count=1)
        if re.match(r"[^¶]{0,900}?istinaf", text, re.I):  # BAM kararı: istinaf incelemesinin giriş cümlesi
            text = EMSAL_BAM_OPENING.sub("", text, count=1)
        text = EMSAL_OPENING_ANYWHERE.sub("", text, count=1)  # giriş kalıbı künyenin hemen başında değilse
        text = EMSAL_PLACEHOLDER.sub("", text)

        verdict = list(EMSAL_VERDICT.finditer(text))
        if verdict:
            body, verdict = text[:verdict[-1].start()], text[verdict[-1].start():]
            verdict = self.clean_emsal_verdict(verdict)
            text = body + PARAGRAPH + verdict
        if case_type:
            text = f"DAVA: {case_type[1]}" + PARAGRAPH + text
        return self.finish(text)

    @staticmethod
    def clean_emsal_verdict(verdict):
        """Hükmün usul maddelerini atar; kapanış paragrafından yalnızca "... karar verildi." kalır, sonrası (imzalar) gider."""
        kept = []
        # maddeler: ayrı satırlar, aynı satırda "1- ... 2- ...", ya da numarasız "..., Alınması gereken harç ..."
        items = re.split(rf"{PARAGRAPH}| (?=\d{{1,2}}[-.] ?[{UP}])|(?<=[,;]) (?=[{UP}])", verdict)
        for i, item in enumerate(items):
            if EMSAL_CLOSING.search(item) and not EMSAL_OUTCOME.search(item):
                # kapanış cümlesi virgülle bölünmüş olabilir: "karar verildi" sonraki parçalarda aranır
                decided = EMSAL_DECIDED.search(" ".join(items[i:]))
                if decided:
                    kept.append(decided[0] + ".")
                break  # kapanıştan sonrası: katip, hâkim, imza satırları
            if re.fullmatch(r"\s*(?:Zabıt )?(?:Katip|Hakim|Başkan|Üye)\w*[\s.…:-]*", item):  # imza satırları
                continue
            if not EMSAL_PROCEDURAL_ITEMS.search(item) or EMSAL_OUTCOME.search(item):
                kept.append(item)
            elif re.search(r"karar verildi\b", item):  # kapanış bir usul maddesiyle aynı maddede: "karar verildi" kalır
                kept.append(EMSAL_DECIDED.search(item)[0] + ".")
        return " ".join(kept)

    # ------------------------------------------------------------------ Yargıtay
    def yargitay(self, text):
        lines = [l for l in text.splitlines() if not re.search(r"e-?[İi]mza|[Ee]lektronik imza|✍", l) or len(l) > 150]
        text = PARAGRAPH.join(re.sub(r"\s+", " ", l).strip() for l in lines if l.strip())
        text = re.sub(r"\bK A R A R\b", "KARAR", text)
        for pattern in YARGITAY_TEMPLATES:
            text = re.sub(pattern, "", text)
        text = VERDICT_CLOSING.sub(r"\g<kapanis>", text)  # tarih gider, "oybirliğiyle karar verildi." kalır
        return self.finish(text)

    # ------------------------------------------------------------------ ortak adımlar
    def finish(self, text):
        """Bütün kaynaklarda en son yapılanlar: atıflar, sayılar, isimler, şehirler, boşluk düzeltme."""
        text = self.drop_citations(text)
        text = VALUE_LABEL.sub("", text)  # önce değer + etiket ("27/01/2010 tarih ve 27475 sayılı"), sonra kalan sayılar
        text = self.drop_numbers(text)
        text = self.drop_names(text)
        text = CITY.sub("", text)
        return self.tidy(text)

    def split_verdict(self, text, heading_pattern):
        """(hükümden önceki metin, hüküm, hükümden sonrası: imzalar, karşı oylar, ekli tablolar)"""
        heading = heading_pattern.search(text)
        if not heading:
            return text, "", ""
        end = re.search(r"karar verildi\.", text[heading.end():])
        end = heading.end() + end.end() if end else len(text)
        return text[:heading.start()], text[heading.start():end], text[end:]

    def clean_verdict(self, verdict):
        """Hükmün usule ilişkin maddelerini atar; sonuç maddeleri ve "... karar verildi." kapanışı kalır.

        Kapanış çoğunlukla usule ilişkin son maddenin sonunda durur ("... GÖNDERİLMESİNE 9/4/2026 tarihinde
        OYBİRLİĞİYLE karar verildi."); bu yüzden önce ayrılır, maddeler süzüldükten sonra tarihsiz olarak geri eklenir.
        """
        verdict = re.sub(r"Açıklanan gerekçelerle;", "", verdict)
        closing = VERDICT_CLOSING.search(verdict)
        verdict = VERDICT_CLOSING.sub("", verdict)
        # "A. ...", "B. ...": hüküm maddeleri; maddenin önünde boşluk ya da paragraf işareti olabilir
        kept = []
        for item in re.split(rf"(?=[ {PARAGRAPH}][{UP}]\. )", verdict):
            if not AYM_BB_PROCEDURAL_ITEMS.search(item):
                kept.append(item)
            elif AYM_BB_OUTCOME.search(item):  # sonuç da var: yalnızca usul parçaları gider
                clauses = re.split(r"(?<=[,;]) ", item)
                kept.append(" ".join(c for c in clauses if not AYM_BB_PROCEDURAL_ITEMS.search(c)))
        verdict = "".join(kept).rstrip(" ,;")
        return verdict + " " + closing["kapanis"] if closing else verdict

    def keep_dissents(self, tail, signature_pattern):
        """İmza bloğu ve ekli tablolar atılır; karşı oy / farklı gerekçe metinleri imzalarına kadar tutulur."""
        kept, end = [], 0
        for m in DISSENT.finditer(tail):
            if m.start() < end:  # bir önceki karşı oyun metni içinde geçen "KARŞIOY" kelimesi
                continue
            dissent = tail[m.start():]
            signature = signature_pattern.search(dissent, m.end() - m.start())
            dissent = dissent[:signature.start()] if signature else dissent
            kept.append(dissent)
            end = m.start() + len(dissent)
        return " ".join(kept)

    # ------------------------------------------------------------------ yardımcılar
    @staticmethod
    def mark_paragraphs(text):
        """Paragraf numaralarını (1. 2. 3. ...) atıp yerine paragraf işareti koyar; başlıkların önüne de koyar.

        Numaralar sırayla aranır (beklenen numara ya da bir fazlası): böylece "Ankara 13. İdare Mahkemesi" gibi
        sayılar ancak tam sırası gelmişse karışabilir; karışsa bile sıra bir sonraki paragrafta düzelir.
        """
        out, pos, expected = [], 0, 1
        for m in re.finditer(rf"(?<= )(\d{{1,3}})\. (?=[{UP}\"“(\d])", text):
            if int(m[1]) in (expected, expected + 1):
                out.append(text[pos:m.start()] + PARAGRAPH)
                pos, expected = m.end(), int(m[1]) + 1
        text = "".join(out) + text[pos:]
        return re.sub(rf" (?={ROMAN_HEADING}|{LETTER_HEADING})", PARAGRAPH, text)

    @staticmethod
    def drop_section(text, heading, subsection=False):
        """Başlıktan bir sonraki aynı ya da üst düzey başlığa kadar olan bölümü atar."""
        until = ROMAN_HEADING + (f"|{LETTER_HEADING}" if subsection else "")
        return re.sub(rf"{heading}.*?(?={PARAGRAPH}?(?:{until})|$)", "", text)  # paragraf işareti kalır

    @staticmethod
    def drop_citations(text):
        text = CITATION.sub("", text)
        return BARE_CITATION.sub("", text)

    @staticmethod
    def drop_numbers(text):
        """Tarihler, karar numaraları, kanun ve madde numaraları, paragraf atıfları; para tutarları kalır."""
        for pattern in [DATE, CASE_NUMBER, LAW_NUMBER, ARTICLE_NUMBER, SECTION_SIGN]:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def drop_names(text):
        return INITIALS.sub("", NAME.sub("", text))

    @staticmethod
    def tidy(text):
        """Silinen parçaların bıraktığı boşlukları ve noktalama artıklarını toparlar."""
        text = re.sub(r"\(\s*[,;]?\s*\)", "", text)  # boş kalan parantezler
        text = re.sub(r" +", " ", text)
        text = re.sub(r" ([,.;:)])", r"\1", text)
        text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)  # ", ," -> ","
        text = re.sub(r"\( ", "(", text)
        paragraphs = [p.strip(" ,;") for p in text.split(PARAGRAPH)]
        return "\n\n".join(p for p in paragraphs if p)


# --- İNCELEME RAPORU: işi bitince bu bölüm silinebilir (aşağıdaki her şey) ---------------------------------
if __name__ == "__main__":
    import pandas as pd

    import config

    SOURCES = ["yargitay"]  # raporu yazılacak kaynaklar (şu an yalnızca gözden geçirilen mahkeme)
    SAMPLE = 10  # her kaynak için rapora önce/sonra hâliyle yazılacak karar sayısı
    pool = pd.read_parquet(config.dataset_dir(2000) / "kararlar.parquet")
    cleaner = Cleaner()
    for source in SOURCES:
        decisions = pool[pool.source == source].copy()
        decisions["temiz"] = [cleaner.clean(source, t) for t in decisions.text]
        decisions["once_kelime"] = decisions.text.str.split().str.len()
        decisions["sonra_kelime"] = decisions.temiz.str.split().str.len()
        decisions["atilan_yuzde"] = 100 * (1 - decisions.sonra_kelime / decisions.once_kelime)

        summary = (f"{source}, 2000'lik havuz: {len(decisions)} karar\n"
                   f"Toplam kelime: {decisions.once_kelime.sum():,} -> {decisions.sonra_kelime.sum():,} "
                   f"(%{100 * (1 - decisions.sonra_kelime.sum() / decisions.once_kelime.sum()):.1f} atıldı)\n"
                   f"Karar başına atılan: ortanca %{decisions.atilan_yuzde.median():.1f}, "
                   f"en az %{decisions.atilan_yuzde.min():.1f}, en çok %{decisions.atilan_yuzde.max():.1f}")
        print(summary)

        # en çok kesilen karar en başta, ardından rastgele örnekler
        most_cut = decisions.loc[[decisions.atilan_yuzde.idxmax()]]
        sample = decisions.drop(most_cut.index).sample(SAMPLE, random_state=config.SEED).sort_values("year")
        lines = [f"# {source} temizleme: önce / sonra", "", "```", summary, "```", ""]
        for r in pd.concat([most_cut, sample]).itertuples():
            note = "EN ÇOK KESİLEN — " if r.Index in most_cut.index else ""
            lines += [f"## {note}{r.id} ({r.year}) — %{r.atilan_yuzde:.1f} atıldı "
                      f"({r.once_kelime} -> {r.sonra_kelime} kelime)",
                      "", "### Önce", "", "```text", re.sub(r"\n\s*\n", "\n", r.text), "```", "",
                      "### Sonra", "", "```text", r.temiz, "```", ""]
        out = config.dataset_dir(2000) / f"temizlik_{source}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Örnekler: {out}\n")
