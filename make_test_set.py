r"""Test setlerini oluşturur ve doğrular.

Her karar için üç tip sorgu ve bir dilekçe vardır:
    H  hukukçu dili       : hukuki mesele, kararın ifadeleri kullanılmadan sorulur
    A  avukat dili        : olay, kararın somut detaylarıyla anlatılır (kolay)
    F  farklı olay        : aynı hukuki mesele, başka bir olay üzerinden sorulur (zor, gerçekçi)
    D  dilekçe            : bu kararı arayan bir dilekçe; olayın detayları değiştirilmiş, kanun adı
                            ve numarası yok (data/test_sets/dilekceler/T01.txt ... T30.txt)

Sürümler:
    v1         = H + A          (60 sorgu)
    v2         = H + A + F      (90 sorgu)
    dilekce_v1 = D              (30 dilekçe)

Kontroller: her kanıt cümlesi kararın metninde birebir geçmeli; sorgu tiplerinin kararla
kelime örtüşmesi, alakasız kararlarla olan örtüşmeyle (taban seviye) birlikte raporlanır.

Kullanım:
    .venv\Scripts\python.exe make_test_set.py
"""
import re

import pandas as pd

import config

VERSIONS = {"v1": ["H", "A"], "v2": ["H", "A", "F"], "dilekce_v1": ["D"]}
TYPE_NAMES = {"H": "hukukcu_dili", "A": "avukat_dili", "F": "farkli_olay", "D": "dilekce"}
PETITIONS_DIR = config.TEST_SETS_DIR / "dilekceler"  # T01.txt ... T30.txt, ITEMS sırasıyla

ITEMS = [
    # ---------------- Yargıtay ----------------
    dict(kaynak="yargitay", esas="2013/26570",
         H="Ruhsatsız olduğu için kapatılan işyerinin çalışmaya devam etmesinde, sorumlu yönetici sıfatının olay günü sürüp sürmediği araştırılmadan ceza verilmesi hukuka uygun mu?",
         A="Müvekkilim LPG istasyonunun mesul müdürüydü ama sözleşmesi 5 aylıktı. Mühürlenen istasyon çalışmaya devam edince mühür bozmadan ceza aldı, temyizde bozma çıkar mı?",
         F="Belediye dükkanımı ruhsatsız diye mühürledi. O tarihte işletmenin sorumlusu artık ben değildim ama faaliyet sürdüğü için hakkımda dava açıldı. Suçun bana ait olduğu nasıl ispatlanmalı?",
         kanit="atılı suçu işlediğine dair sübut delillerinin nelerden ibaret olduğu karar yerinde tartışılmadan eksik incelemeyle yazılı şekilde hüküm kurulması"),
    dict(kaynak="yargitay", esas="2008/10402",
         H="Parasal talebi tamamen karşılıksız bırakılan davada, avukatla temsil edilen karşı tarafa sabit tutarlı avukatlık ücreti yerine dava değerine orantılı ücret verilmesi gerekir mi?",
         A="Adi ortaklıktan doğan alacak davası reddedildi ama mahkeme bizim lehimize sadece maktu vekalet ücreti verdi. Reddedilen tutar üzerinden nispi ücret istememiz gerekmez mi?",
         F="Bana karşı açılan tazminat davası tamamen reddedildi ama mahkeme avukatım için çok düşük sabit bir ücret belirledi. Talep edilen tutara göre hesaplanmış ücret istemem mümkün mü?",
         kanit="reddedilen kısım üzerinden nispi oranda vekalet ücretine hükmedilmesi gerekirken yazılı şekilde maktu vekalet ücrete hükmedilmiş olması usul ve yasaya aykırı olup bozma nedenidir"),
    dict(kaynak="yargitay", esas="2014/19674",
         H="Satış belgesinin düzenlenmesinden sonra yapılan tahsilatlar hangi borca sayılır ve bunların farklı bir alacağa ilişkin olduğunu ileri süren taraf neyi ispatlamak zorundadır?",
         A="Faturayı çeklerle ödedik ama karşı taraf çeklerin başka bir borç için verildiğini söyleyip icra takibi başlattı. İspat yükü kimde?",
         F="Tedarikçimize aldığımız malların parasını birkaç havaleyle gönderdik, şimdi bu paraların eski bir borca sayıldığını söyleyip bizden tekrar ödeme istiyorlar. Bunu kim kanıtlamalı?",
         kanit="Fatura tarihinden sonra yapılan ödemelerin fatura borcuna mahsuben yapıldığının kabulü gerekir. Bu ödemelerin başka bir borca mahsuben yapıldığını iddia eden alacaklı iddiasını kanıtlamalıdır."),
    dict(kaynak="yargitay", esas="2014/7946",
         H="İstimlak nedeniyle açılmış ve uzun süren bedel davalarında mülk sahibine gecikme için faiz ödenmesi gerekir mi, faiz hangi tarihten başlar?",
         A="Tarlamdan elektrik direği geçti, kamulaştırma bedeli davası yıllardır sürüyor. Geç ödenen bedel için faiz alabilir miyim?",
         F="Devlet yol yapımı için arsamı aldı, bedel davası uzun sürdü ve para geç ödendi. Bekleme süresi için ek ödeme talep edebilir miyim?",
         kanit="30.04.2013 tarihinden önce açılmış ve henüz kesinleşmemiş kamulaştırma bedelinin tespiti ve tescili davalarında öngörülen dört aylık yargılama süresinin makul süre kabul edilerek"),
    dict(kaynak="yargitay", esas="2024/7312",
         H="Sigorta uyuşmazlıklarında hakem kurulunun itiraz üzerine verdiği karara karşı Yargıtay yoluna başvurulabilmesi için aranan parasal eşik nedir?",
         A="Sigorta tahkimde itiraz hakem heyeti karşı tarafı haklı buldu. 184 bin TL'lik alacak için Yargıtay'a gidebilir miyiz, yoksa karar kesin mi?",
         F="Kasko şirketiyle aramızdaki anlaşmazlıkta hakem kararı aleyhimize çıktı. Tutar çok yüksek değil, bu karara karşı üst mahkemeye başvurabilir miyim?",
         kanit="Sigorta Tahkim Komisyonuna yapılan iki yüz otuz sekiz bin yedi yüz otuz Türk Lirasının üzerindeki uyuşmazlıklar hakkında itiraz üzerine verilen hakem kararları için temyize gidilebilir."),
    dict(kaynak="yargitay", esas="2011/8828",
         H="Cadde kenarında bırakılmış ve hiçbir yere bağlanmamış iki tekerli aracın çalınması hangi ağırlaştırılmış hırsızlık hâline girer?",
         A="Çocuk sokakta kilitsiz duran bir motosikleti aldı. Bu basit hırsızlık mı yoksa nitelikli hırsızlık mı sayılır?",
         F="Apartman önünde kilitsiz bırakılan elektrikli scooter'ı alan kişi hangi suçtan sorumlu olur?",
         kanit="Sanığın, müştekinin sokağa park ettiği ve sabit bir yere kilitlemediği motosikletini alması şeklindeki eyleminin TCK'nun 142/1-e maddesine uyduğu gözetilmeden"),
    dict(kaynak="yargitay", esas="2023/9851",
         H="Yüksek mahkemenin geçiş hükmünü iptal etmesinden sonra, temyiz incelemesi süren ceza dosyalarında sadeleştirilmiş yargılama yolunun uygulanabilirliği ele alınmalı mıdır?",
         A="Tehdit suçundan 2020'de mahkum oldum, dosya hâlâ Yargıtay'da. Basit yargılama usulünden yararlanmam mümkün mü?",
         F="Birkaç yıl önce yaralama suçundan ceza aldım, dosyam hâlâ temyizde bekliyor. Sonradan getirilen daha hafif yargılama yolu benim dosyama da uygulanabilir mi?",
         kanit="Anayasa'nın 38 inci maddesi ile 5237 sayılı Türk Ceza Kanunu'nun 7 ve 5271 sayılı Kanun'un 251 ve devamı maddeleri gereğince sanığın hukuki durumunun değerlendirilmesinde zorunluluk bulunması"),
    dict(kaynak="yargitay", esas="2008/11586",
         H="Muhatap adreste bulunamadığında yapılan bildirimde, haber verilen kişinin imzası tutanakta yoksa bildirim geçerli sayılır mı ve yokluğunda verilen hükme etkisi nedir?",
         A="Duruşma tebligatı kapıya yapıştırılmış, komşuya haber verildiği yazıyor ama komşunun imzası yok. Haberim olmadan davayı kaybettim, ne yapabilirim?",
         F="Ödeme emri kapıya bırakılmış, kime haber verildiği belli değil. İtiraz süresini kaçırdım, bu tebligat geçerli mi?",
         kanit="Ancak komşu ... imzası alınmamıştır. Bu durumda tebligat işleminin kanun ve tüzük hükmüne uygun yapılmadığı anlaşılmaktadır."),
    dict(kaynak="yargitay", esas="2011/2683",
         H="Evlilik birliği sürerken eşlerden biri, taşınmaza yaptığı katkı nedeniyle mülkiyetin kendisine geçirilmesini veya para ödenmesini isteyebilir mi?",
         A="Eşimle boşanma davamız reddedildi, hâlâ evliyiz. Evin tapusu için katkı payı davası açabilir miyim?",
         F="Eşimle hâlâ evliyiz ama ayrı yaşıyoruz. Birlikte aldığımız arabanın parasının bir kısmını ben ödedim, payımı şimdiden dava ederek alabilir miyim?",
         kanit="taraflar arasındaki mal rejimi henüz sona ermediğinden mal rejiminin tasfiyesine ilişkin eldeki davanın dinlenme olanağı kalmadığından"),
    dict(kaynak="yargitay", esas="2015/10662",
         H="Eski ve yeni ceza kanunlarına göre bulunan cezalar aynı olduğunda, yalnızca yeni kanunda yer alan hak kısıtlamaları hangi kanunun sanık yararına olduğunu belirler mi?",
         A="Eski TCK döneminde işlenen çek sahteciliğinde iki kanuna göre ceza eşit çıkıyorsa, yeni TCK'daki hak yoksunlukları yüzünden hangi kanun uygulanmalı?",
         F="Suç eski kanun döneminde işlendi, karar yeni kanun döneminde verildi ve iki kanuna göre hapis süresi aynı çıkıyor. Seçme hakkının kısıtlanması gibi ek sonuçlar hangi kanunun uygulanacağını etkiler mi?",
         kanit="5237 sayılı TCK'nun 53. maddesinde düzenlenen hapis cezasına mahkumiyetinin kanuni sonucu olarak uygulanan hak yoksunluğuna ilişkin güvenlik tedbirlerinin 765 sayılı Kanunda bulunmaması nedeniyle, 5237 sayılı TCK ile yapılacak uygulamanın aleyhe olduğu"),
    # ---------------- UYAP Emsal ----------------
    dict(kaynak="emsal", esas="2022/2761",
         H="Dava sürerken mahkemelerin iş alanını değiştiren yeni bir yasa çıkarsa ve geçiş düzenlemesi yoksa, bu değişiklik derdest dosyalara da uygulanır mı?",
         A="Davamız açıldıktan sonra kanun değişti ve görevli mahkeme değişti. Asliye hukuktaki dosyamız ticaret mahkemesine mi gönderilmeli?",
         F="Kira davamız sulh hukukta görülürken kanun değişti ve bu tür davalar başka mahkemeye verildi. Bizim dosyamız da oraya gönderilmek zorunda mı?",
         kanit="usul kuralları ve bu kapsamda yer alan görev kuralları kamu düzenine ilişkin olup, aksine düzenleme yoksa derhal uygulanacağından"),
    dict(kaynak="emsal", esas="2021/845",
         H="Ticari alım satımda alıcı ürünleri teslim almadığını savunuyor ama kendi muhasebe kayıtlarını mahkemeye sunmuyorsa, vergi bildirimlerindeki kayıtlar teslimi kanıtlar mı?",
         A="Müşterimize dolar bazlı faturayla yağ sattık. Ödemedi, icra takibine itiraz edip malı teslim almadığını söylüyor, faturaya da itiraz etmemişti. Davayı kazanır mıyız?",
         F="Toptancıya kumaş gönderdik, parasını ödemiyor ve malları hiç almadığını söylüyor. Kendi muhasebe kayıtlarını da göstermiyor, teslimi nasıl ispatlarız?",
         kanit="davalı tarafça defter ibraz edilmediği, faturaya itiraz edilmediği, davalı tarafın malların teslim edilmediğine ilişkin itirazının da BS formlarının incelenmesi sonucu yerinde olmadığı"),
    dict(kaynak="emsal", esas="2022/181",
         H="Rızası dışında elinden çıkan bir ödeme aracının hükümsüz sayılması için hangi duyuru ve bekleme şartlarının yerine getirilmesi gerekir?",
         A="Elimdeki çeki kaybettim. Mahkemeden çekin iptalini nasıl isterim, süreç ne kadar sürer?",
         F="Şirket kasasından çalınan senet için muhatabın ödeme yapmaması ve senedin geçersiz sayılması amacıyla ne yapmalıyız?",
         kanit="Türkiye Ticaret Sicil Gazetesinde 3 kez yasal ilanın yapılmış olduğu, ilk ilan tarihi olan 06.04.2022 tarihinden itibaren 3 aylık yasal bekleme süresinin geçtiği",
         ek=[("emsal", "2022/721")]),
    dict(kaynak="emsal", esas="2023/69",
         H="Araçtaki piyasa değeri düşüşü için tahkim yoluyla kısmen tazmin alan kişi, aynı zarar için ikinci kez mahkemeye başvurabilir mi?",
         A="Kazadan sonra değer kaybı için sigorta tahkime başvurdum ve kısmen kazandım. Kalan kısım için şimdi mahkemede dava açabilir miyim?",
         F="Kiracıma karşı açtığım alacak davasını kazandım ve karar kesinleşti. Aynı kira dönemi için başka bir dava daha açabilir miyim?",
         kanit="Birinci dava ile ikinci davanın müddeabihlerinin (konularının) yani dava ile elde edilmek istenen sonucun aynı olması, dava sebeplerinin yani davanın dayandığı olayların aynı olması ve davanın taraflarının aynı olması halinde maddi anlamda kesin hüküm oluşturur"),
    dict(kaynak="emsal", esas="2020/30",
         H="Başkasına ait tescilli işarete benzeyen bir işaretin tescilini talep etmek, taklit ürünün kim tarafından yapıldığı ispatlanmadan hak ihlali sayılır mı?",
         A="Rakip firma markamıza benzeyen bir marka için başvuru yaptı, piyasada da benzer bir yağ ürünü var ama ürünü onların ürettiğini kanıtlayamıyoruz. Tecavüz davasını kazanır mıyız?",
         F="Bir rakibin logomuza benzeyen bir logo için tescil başvurusu yaptığını gördük. Piyasada benzer ambalajlı ürünler de var ama kimin ürettiği belli değil. Rakibe karşı dava açabilir miyiz?",
         kanit="başvuru yapmasının tek başına davacının marka haklarına tecavüz niteliğinde olmadığı"),
    dict(kaynak="emsal", esas="2022/214",
         H="Enerji iletim hattının kurallara aykırı kullanılması nedeniyle istenen yaptırım bedelinin tahsiline ilişkin anlaşmazlığa hangi yargı kolu bakar?",
         A="İletim şirketinin kestiği sistem kullanım ceza faturası için ticaret mahkemesinde alacak davası açıldı. Bu dava adli yargıda mı görülür?",
         F="Rüzgar santralimiz şebeke bağlantı anlaşmasındaki kuralları ihlal ettiği gerekçesiyle ceza faturası aldı. Bu faturayla ilgili davayı hangi yargı yerinde açmalıyız?",
         kanit="Yukarıda açıklanan yasal düzenleme uyarınca, tarafların sıfatı ve dava konusu bütün olarak değerlendirildiğinde mevcut davada idari yargı görevlidir."),
    dict(kaynak="emsal", esas="2023/873",
         H="Sayaçsız veya sözleşmesiz enerji tüketimi nedeniyle istenen bedele dair davalar ticaret mahkemesinin alanına girer mi, hangi ölçüt esas alınır?",
         A="Elektrik şirketi kaçak elektrik tespit edip icra takibi yaptı, itiraz ettim, şimdi ticaret mahkemesinde dava açtılar. Bu mahkeme görevli mi?",
         F="Dağıtım şirketi berber dükkanımda usulsüz enerji kullanımı tespit edip bedel istiyor. Açılan dava asliye hukukta mı yoksa ticaret mahkemesinde mi görülmeli?",
         kanit="Türk Ticaret Kanunu, kanun gereği ticari dava sayılan davalar haricinde, ticari davayı ticari iş esasına göre değil, ticari işletme esasına göre belirlemiştir."),
    dict(kaynak="emsal", esas="2022/1",
         H="Özel yetki içeren vekaletnameye dayanarak avukatın talepten tamamen vazgeçmesi durumunda mahkeme nasıl bir karar verir ve bu karar neyi sonuçlar?",
         A="Marka tecavüzü davamızdan vazgeçmek istiyoruz. Avukatımız tek başına davadan feragat edebilir mi, bunun sonucu ne olur?",
         F="Açtığımız tazminat davasından vazgeçmeye karar verdik. Vazgeçersek ileride aynı konuda yeniden dava açabilir miyiz?",
         kanit="feragatin kesin hüküm gibi hukuki sonuçlar doğuracağı açıkça belirtilmiştir"),
    # ---------------- Danıştay ----------------
    dict(kaynak="danistay", esas="2022/3049",
         H="Yetkili kurumdan onay almadan başlatılan lisansüstü programa yapılan öğrenci kabulü, yıllar sonra idarece geri alınabilir mi?",
         A="Üniversite, izin almadan açtığı doktora programındaki kaydımı yıllar sonra sildi. Kazanılmış hakkım yok mu?",
         F="Özel okul, bakanlık onayı olmadan açtığı bir bölüme beni kaydetti, iki yıl sonra kaydım iptal edildi. Bu iptal hukuka uygun mu?",
         kanit="İzinsiz açıldığı anlaşılan doktora programına açık hata kapsamında yapılmış olan kaydın, kazanılmış hak olarak kabulünün mümkün olmadığı"),
    dict(kaynak="danistay", esas="2021/194",
         H="Mükellef hakkındaki inceleme raporu idareye ulaştıktan sonra yararlanılan vergi barışı düzenlemesi, raporda belirlenen vergilerin cezalı olarak istenmesine engel olur mu?",
         A="Hakkımda vergi inceleme raporu yazıldı, vergi dairesi matrah artırımı yapabileceğimi bildirdi, yaptım ve ödedim. Buna rağmen cezalı vergi kesildi, bu hukuka uygun mu?",
         F="Şirketimiz hakkında vergi müfettişi rapor yazdıktan sonra af kapsamında başvurup ödeme yaptık. İdare yine de raporu esas alıp cezalı vergi istiyor, bu mümkün mü?",
         kanit="matrah artırımı başvurusuna olanak sağlayan vergi tekniği raporu ile aynı tarihte düzenlenen ve yine aynı tarihte vergi dairesi kayıtlarına giren rapor"),
    dict(kaynak="danistay", esas="2019/2511",
         H="Akaryakıt istasyonunun izleme cihazı devre dışıyken satış yapılması halinde, istasyonun bağlı olduğu şirkete düzenleyici kurumca yaptırım uygulanabilir mi?",
         A="Akaryakıt dağıtım şirketiyiz, bayimiz otomasyon sistemini arıza bahanesiyle kapatıp satış yapmış. EPDK cezayı bize kesti, bayinin kusurundan biz sorumlu muyuz?",
         F="Bayimiz, satışları kayda geçiren elektronik sistemi devre dışı bırakıp yakıt sattı. Düzenleyici kurum cezayı lisans sahibi olarak bize verdi, bu doğru mu?",
         kanit="otomasyon cihazına bağlı olmayan tanktan akatyakıt ikmali yapıldığı anlaşıldığından"),
    dict(kaynak="danistay", esas="2024/6205",
         H="Yatırım yoluyla kazanılan statünün kaldırılması işlemine karşı açılan davada, dilekçede kimin hangi işlemi dava ettiği belirsizse mahkeme ne yapar?",
         A="Yatırım yoluyla aldığımız Türk vatandaşlığı Cumhurbaşkanı kararıyla geri alındı. Dava dilekçesinde çocuklarımızdan da bahsettik, mahkeme dilekçeyi reddetti. Neden?",
         F="Gayrimenkul alarak vatandaş olmuştum, vatandaşlığım iptal edildi. Dava dilekçemde eşimin ve çocuklarımın durumunu da yazdım ama mahkeme dilekçeyi usulden geri çevirdi. Neyi yanlış yaptık?",
         kanit="davacıların çocukları bakımından Türk vatandaşlığının geri alınması işleminin iptalinin istenilip istenilmediği hususunda tereddüt meydana gelmiştir"),
    dict(kaynak="danistay", esas="2022/3036",
         H="Madencilik faaliyeti için kullanılan orman alanlarında ödenen bedellere uygulanan yüzde elli teşvik hangi tarihten itibaren ve ne kadar süreyle geçerlidir?",
         A="Rödövansla maden işletiyoruz, orman izin bedelinde yüzde 50 indirim istedik ama ruhsatın ilk izni 1991'de alındığı için reddedildi. Haklılar mı?",
         F="Kiraladığımız taş ocağı orman arazisinde. İzin bedelinde indirim istedik ama sahanın ilk izni çok eskiden alındığı için reddedildi. İndirim süresi neye göre hesaplanır?",
         kanit="işletme izin belgesinin düzenlendiği tarihten itibaren ilk 10 yıllık süre geçtiğinden dava konusu işlemde hukuka aykırılık bulunmadığı"),
    dict(kaynak="danistay", esas="2021/829",
         H="Yabancıya insani gerekçeyle oturma hakkı tanınmaması işlemine karşı dava, merkezi idarenin görüşü alınmış olsa bile hangi ildeki idare mahkemesinde açılır?",
         A="Yabancı müvekkilimin insani ikamet izni başvurusunu İstanbul Valiliği reddetti ama ret Ankara'daki Göç İdaresinin görüşüne dayanıyor. Davayı Ankara'da mı İstanbul'da mı açmalıyım?",
         F="İzmir'de yaşıyorum, oturma izni talebim merkezden gelen olumsuz görüş üzerine valilikçe reddedildi. Davayı İzmir'de mi yoksa başkentte mi açmalıyım?",
         kanit="İstanbul ilinin idari yargı yeri bakımından bağlı bulunduğu İstanbul İdare Mahkemesi yetkili bulunmaktadır"),
    # ---------------- AYM Bireysel Başvuru ----------------
    dict(kaynak="aym_bb", esas="2022/6674",
         H="Örgüt üyeliği şüphesiyle başlatılan ceza soruşturmasının yıllarca sonuçlandırılmaması, kişinin suçsuz sayılma hakkını ve davanın makul sürede bitirilmesi hakkını zedeler mi?",
         A="2017'de başlayan FETÖ soruşturmasında yıllarca hiçbir işlem yapılmadı, bu yüzden kamu görevinden çıkarıldım. Anayasa Mahkemesine hak ihlali başvurusu yapabilir miyim?",
         F="Hakkımda yolsuzluk şüphesiyle soruşturma açıldı, beş yıldır hiçbir işlem yapılmadı ama bu yüzden terfim durduruldu. Bu durum hangi haklarımı ihlal ediyor?",
         kanit="hakkında yürütülen soruşturmada uzun süre işlem yapılmadığını, kamu görevinden bu soruşturma nedeniyle çıkarıldığını"),
    dict(kaynak="aym_bb", esas="2013/7720",
         H="Güvenlik nedeniyle yerleşim yerinden göç eden kişiye, geçmişteki örgüte destek mahkumiyeti gerekçesiyle devlet tazminatı ödenmemesi anayasal haklara aykırı mıdır?",
         A="Terör olayları yüzünden köyümüzü terk etmek zorunda kaldık. Zarar tespit komisyonu eski bir yataklık mahkumiyetim yüzünden başvurumu reddetti, bu hak ihlali mi?",
         F="Çatışmalar nedeniyle bağımızı yıllarca işleyemedik. Terör zararları için yaptığım başvuru, geçmişteki bir mahkumiyetim gerekçe gösterilerek reddedildi. Bu karar anayasal haklarımı ihlal eder mi?",
         kanit="terör örgütüne yardım ve yataklık etmek suçundan mahkum olduğu anlaşılan davacının terör olayları nedeniyle güvenlik kaygısı neticesi ikamet etmekte olduğu mezradan ayrılmasından dolayı uğradığı iddia edilen zararların 5233 sayılı kanun kapsamında karşılanması mümkün olmadığından"),
    dict(kaynak="aym_bb", esas="2022/33267",
         H="Kamu görevine iade sonrası yapılan yer değiştirme işlemine karşı açılan davanın esasa girilmeden zamanında açılmadığı gerekçesiyle reddi, yargı yoluna başvuru hakkını zedeler mi?",
         A="KHK ile ihraç edilen bir polis olarak OHAL Komisyonu kararıyla göreve döndüm ama başka yere atandım. Atamaya açtığım dava süre aşımından reddedildi, hak ihlali var mı?",
         F="Öğretmenim, sürgün niteliğinde başka bir ile tayin edildim. Tayine karşı açtığım dava süresinde açılmadığı gerekçesiyle esasına girilmeden reddedildi. Anayasa Mahkemesine gidebilir miyim?",
         kanit="iptal davasının süre aşımından reddedilerek esasının incelenmemesi nedeniyle başvurucunun mahkemeye erişim hakkına yönelik bir müdahalenin bulunduğu görülmektedir"),
    dict(kaynak="aym_bb", esas="2013/2495",
         H="Emeklilik sonrasında geç ödenen toplu ödemenin, ödeme günündeki güncel değerler yerine ayrılış tarihindeki değerlerle hesaplanması mülkiyet güvencesini zedeler mi?",
         A="Emekli ikramiyem emekli olduğum yılın katsayısıyla ödendi, yıllar sonra ödendiği için değeri eridi. Anayasa Mahkemesine gidebilir miyim?",
         F="Kamudan ayrıldıktan sonra hak ettiğim toplu para gecikmeyle ve eski hesaplamayla ödendi, enflasyon karşısında değeri çok düştü. Bu durum hangi temel hakkımı ihlal eder?",
         kanit="emekli ikramiyesi güncel katsayılar yerine emekli olduğu tarihteki katsayılar üzerinden hesaplanan başvurucunun mülkiyet hakkının ihlal edildiği iddiasına ilişkindir"),
    # ---------------- AYM Norm Denetimi ----------------
    dict(kaynak="aym_norm", esas="2015/64",
         H="Yerel yönetimlere karşı yürütülen takiplerde, cebri icraya geçmeden önce idareden ödemeye yeterli varlık bildirmesinin istenmesi zorunluluğu anayasal mıdır?",
         A="Belediyeden alacağım var ama icra takibinde belediyenin mallarını hemen haczettiremiyorum, bu kısıtlama anayasal mı?",
         F="Yüklenici olarak ilçe belediyesinden hakediş alacağımız var. İcra dairesi haciz yapmadan önce belediyenin mal bildirmesini bekliyor, bu bekleme kuralı anayasal mı?",
         kanit="İcra dairesince haciz kararı alınmadan önce belediyeden borca yeter miktarda haczedilebilecek mal gösterilmesi istenir",
         ek=[("aym_norm", "2014/197"), ("aym_bb", "2013/5604")]),
    dict(kaynak="aym_norm", esas="1986/24",
         H="Suç şüphesi bulunan kişinin üzerinde ve eşyasında delil aranmasına imkân tanıyan eski usul kanunu hükmü, kişi özgürlüğü ve mahremiyet güvenceleri bakımından Anayasa'ya uygun mudur?",
         A="Polis şüphe üzerine müvekkilimi karakola götürüp üstünü aradı ve esrar buldu. Bu tür bir aramaya izin veren kanun maddesi Anayasa'ya uygun mu?",
         F="Polis parkta rastgele durdurduğu bir gencin çantasını hâkim kararı olmadan aradı. Bu tür aramalara izin veren kanun hükümleri Anayasa'ya uygun mu?",
         kanit="Arama, ceza muhakemesi hukukunda, suçluların yakalanması ve suç delillerinin ortaya çıkarılması için başvurulan geçici bir koruma tedbiridir."),
]

STOP = {"için", "veya", "olan", "olarak", "gibi", "ile", "ama", "daha", "hangi", "nedir", "midir", "mıdır",
        "mudur", "müdür", "yoksa", "bile", "kadar", "sonra", "önce", "şimdi", "bizim", "bize", "benim"}


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def stems(text):
    """Kelimeleri ilk 5 harfine indirir: 'tebligat', 'tebligatı', 'tebligatın' -> 'tebli'."""
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return {w[:5] for w in re.findall(r"[a-zçğıöşü]{3,}", text) if w not in STOP}


def overlap(query, text):
    """Sorgu kelimelerinin metinde geçme oranı."""
    q = stems(query)
    return len(q & stems(text)) / len(q)


def main():
    # Hedef kararlar 2000'lik havuzdan seçildi; büyük havuzlar onu kapsar ama aynı esas numaralı
    # başka kararlar da içerebilir (10 binde emsal 2022/181 iki kez geçiyor). Bu yüzden 2000'lik havuz okunur.
    docs = pd.read_parquet(config.dataset_dir(config.BASE_SIZE) / "kararlar.parquet")

    def find(source, esas):
        match = docs[(docs.source == source) & (docs.esas_no == esas)]
        assert len(match) == 1, f"{source} {esas}: {len(match)} eşleşme"
        return match.iloc[0]

    rows, report = [], []
    for i, item in enumerate(ITEMS, 1):
        doc = find(item["kaynak"], item["esas"])
        assert norm(item["kanit"]) in norm(doc.text), f"T{i:02d}: kanıt karar metninde bulunamadı"
        extras = ";".join(find(s, e).id for s, e in item.get("ek", []))
        court = doc.court if pd.notna(doc.court) else ""
        kunye = " ".join(p for p in [config.SOURCE_LABELS[doc.source], court,
                                     f"{item['esas']} E.", doc.karar_tarihi] if p)
        # taban seviye: aynı kaynaktan 20 alakasız karar
        others = docs[(docs.source == doc.source) & (docs.id != doc.id)].sample(20, random_state=0)
        queries = {code: item[code] for code in "HAF"}
        queries["D"] = (PETITIONS_DIR / f"T{i:02d}.txt").read_text(encoding="utf-8").strip()
        for code, query in queries.items():
            rows.append({"test_id": f"T{i:02d}-{code}", "kaynak": doc.source, "karar_id": doc.id,
                         "kunye": kunye, "sorgu_tipi": TYPE_NAMES[code], "sorgu": query,
                         "kanit": item["kanit"], "ek_dogru_kararlar": extras})
            report.append({"sorgu_tipi": TYPE_NAMES[code],
                           "hedef_karar": overlap(query, doc.text),
                           "alakasiz_kararlar": sum(overlap(query, t) for t in others.text) / len(others)})

    table = pd.DataFrame(rows)
    for version, codes in VERSIONS.items():
        subset = table[table.test_id.str[-1].isin(codes)]
        path = config.test_set_path(version)
        subset.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{version}: {len(subset)} sorgu -> {path}")

    print("\nSorgu kelimelerinin karar metninde geçme oranı (%), 30 sorgu ortalaması:")
    summary = 100 * pd.DataFrame(report).groupby("sorgu_tipi").mean()
    summary["hedefe_ozgu_fark"] = summary.hedef_karar - summary.alakasiz_kararlar
    print(summary.round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
