r"""Deney sonuçlarından hocaya gösterilecek özet tablo: data/deney_ozeti.xlsx

Kaynaklar:
    data/experiments.csv                 bütün embedding denemeleri
    data/rerank_sonuc/experiments.csv    Drive'dan inen kopya; buradan yalnızca reranker satırları alınır

Sonuç sütunları tek: geniş ölçü yapılmışsa o, yapılmamışsa tek hedef ölçüsü yazılır (ilk 100 yalnızca geniş).
Satırların hepsi kalır; her deney grubunun önüne kalın bir başlık satırı konur.

İkinci sayfa "en iyiler": her havuz ve görev (sorgu tipi) için bölümü ne olursa olsun en iyi 3 deney,
hem ilk5'e göre (son sıralama) hem ilk100'e göre (reranker'a aday getirici). Görev sonuçları deneylerin
sorgu sorgu sonuç dosyalarından (eval/*.csv) hesaplanır.

Kullanım:
    .venv\Scripts\python.exe ozet_tablo.py
"""
import re

import pandas as pd
from openpyxl.styles import Font

SECTIONS = {
    1: "1. Parçalama yöntemleri: 2k havuz, ham veri, Magibu (tek hedef ölçütü)",
    2: "2. Büyük havuz ve diğer modeller: 10k havuz, ham veri, 3 model, dilekçe testi",
    3: "3. Temiz veri ve küçük parçalar: 2k ve 10k, 3 model",
    4: "4. Reranker: embedding'in ilk 100 adayı yeniden sıralandı (2k havuz)",
}
METHODS = {
    "per_decision": "tüm karar", "fixed_tokens": "sabit token", "sentence": "cümle", "recursive": "recursive",
    "sections_structural": "bölüm (yapısal)", "sections_fixed": "bölüm (sabit)", "sections_paragraph": "bölüm (paragraf)",
}
TEST_SETS = {"v1": "v1", "v2": "v2", "dilekce_v1": "dilekçe", "tekrar_T20-F_x100": "tekrar testi (T20, 100 kez)"}
TASKS = {"avukat_dili": "avukat dili", "hukukcu_dili": "hukukçu dili", "farkli_olay": "farklı olay", "dilekce": "dilekçe"}
CONFIG = ["havuz", "veri", "model", "yöntem", "test seti", "reranker", "reranker metni"]  # bir deneyi tanımlayan sütunlar

# sorgu sorgu sonuç dosyalarının klasör adları
FOLDERS = {"magibu/embeddingmagibu-200m": "magibu_embedding", "ytu-ce-cosmos/turkish-e5-large": "cosmos_e5",
           "google/embeddinggemma-300m": "embeddinggemma", "alibayram/embeddingmagibu2-text-v1": "magibu2_text"}
RERANKER_FOLDERS = {"yok": "yok", "ytu-ce-cosmos/modernbert-tr-reranker": "modernbert_tr",
                    "BAAI/bge-reranker-v2-m3": "bge_m3", "Qwen/Qwen3-Reranker-0.6B": "qwen3_06b"}
RERANK_DIRS = {"parça": "rerank", "temiz": "rerank_temiz", "orijinal": "rerank_orijinal"}


def method(row):
    match = re.fullmatch(r"(.+)_c(\d+)_o(\d+)", row.deneme)
    label = METHODS[row.deneme] if match is None else f"{METHODS[match[1]]} {match[2]}/{match[3]}"
    if re.search(r"tam k.nye", str(row["not"])):  # "not" sütunu cp1254 yazılmış, "ü" bozuk okunuyor
        label += " (tam künye)"
    return label


def data_kind(row):
    note = str(row["not"])
    if "aday:" in note:  # reranker satırı: adaylar temiz indeksten mi geldi
        return "temiz" if "/temiz/" in note else "ham"
    return "temiz" if note.startswith("temiz") else "ham"


def rerank_text(row):
    match = re.search(r"metin: (\w+)", row["not"])
    return "parça" if match is None else match[1]  # pilotta reranker eşleşen parçayı okudu


def section_of(main, rerank):
    sections = pd.Series(2, index=main.index)
    sections[main.karar_sayisi == 2000] = 1
    sections[main.tarih.str.startswith("#").cumsum() > 0] = 3  # "#####" satırından sonrası temiz veri
    return pd.concat([sections, pd.Series(4, index=rerank.index)])


def load_rows():
    main = pd.read_csv("data/experiments.csv", encoding="utf-8-sig")
    rerank = pd.read_csv("data/rerank_sonuc/experiments.csv", encoding="utf-8-sig")
    rerank = rerank[rerank.reranker.notna()]
    frame = pd.concat([main, rerank], ignore_index=True)
    frame["bolum"] = section_of(main, rerank).to_numpy()
    return frame[~frame.tarih.str.startswith("#")]


def summarize(frame):
    table = pd.DataFrame({
        "bölüm": frame.bolum,
        "havuz": frame.karar_sayisi.map({2000: "2k", 9977: "10k"}),
        "veri": frame.apply(data_kind, axis=1),
        "model": frame.model,
        "yöntem": frame.apply(method, axis=1),
        "test seti": frame.test_seti.map(TEST_SETS),
    })
    for k in (1, 5, 10):
        table[f"ilk{k}"] = frame[f"ilk{k}_genis_%"].fillna(frame[f"ilk{k}_%"])
    table["ilk100"] = frame["ilk100_genis_%"]
    is_rerank = frame.bolum == 4
    table["reranker"] = frame.reranker.where(is_rerank)
    table["reranker metni"] = frame[is_rerank].apply(rerank_text, axis=1)
    return table.round(1)


def eval_path(row):
    """Bu satırın sorgu sorgu sonuç dosyası."""
    file = f"{row.deneme}_{row.test_seti}.csv"
    if row.bolum == 4:
        source = re.search(r"aday: n2000/(\S+)", row["not"])[1]
        return f"data/rerank_sonuc/{RERANK_DIRS[rerank_text(row)]}/{source}/{RERANKER_FOLDERS[row.reranker]}/{file}"
    size = "n2000" if row.karar_sayisi == 2000 else "n10000"
    variant = ""
    if row.bolum == 3:
        variant = "temiz_tam_kunye/" if re.search(r"tam k.nye", row["not"]) else (
            "ham_kisa_baslik/" if row["not"].startswith("ham") else "temiz/")
    return f"data/{size}/{variant}{FOLDERS[row.model]}/eval/{file}"


def per_task(frame, table):
    """Her deneyin sonuçları sorgu tipine (görev) göre ayrı ayrı: avukat dili, hukukçu dili, farklı olay, dilekçe."""
    rows = []
    for i, row in frame[frame.test_seti.isin(["v2", "dilekce_v1"])].iterrows():
        results = pd.read_csv(eval_path(row), encoding="utf-8-sig")
        wide = pd.notna(row["ilk1_genis_%"])  # geniş ölçü yapılmışsa o, yoksa tek hedef
        column = ("sira_genis" if wide else "sira") + ("_sonra" if row.bolum == 4 else "")
        check = 100 * (results[column] <= 5).mean()
        if abs(check - table.at[i, "ilk5"]) > 0.1:
            print(f"UYARI: {eval_path(row)} ilk5 {check:.1f}, tabloda {table.at[i, 'ilk5']}")
        for task, ranks in results.groupby("sorgu_tipi")[column]:
            scores = {f"ilk{k}": 100 * (ranks <= k).mean() for k in (1, 5, 10)}
            # eski koşuların dosyalarında yalnızca ilk 10 sıra var
            scores["ilk100"] = 100 * (ranks <= 100).mean() if pd.notna(row["ilk100_genis_%"]) else None
            rows.append({**table.loc[i, CONFIG].to_dict(), "görev": TASKS[task], **scores})
    return pd.DataFrame(rows).round(1)


def best(tasks):
    """Havuz ve görev başına en iyi 3: son sıralama (ilk5'e göre) ve aday getirici (ilk100'e göre)."""
    tasks = tasks.copy()
    no_rerank = tasks.reranker.isna() | (tasks.reranker == "yok")  # "yok" satırı aynı embedding denemesinin tekrarı
    tasks.loc[no_rerank, ["reranker", "reranker metni"]] = None
    groups = []
    for pool in ("10k", "2k"):
        for task in TASKS.values():
            group = tasks[(tasks.havuz == pool) & (tasks.görev == task)]
            parts = []
            # reranker ilk 100'ü yalnızca yeniden sıralar; aday getiriciler embedding denemeleri arasından seçilir
            for label, order, candidates in (("son sıralama", ["ilk5", "ilk1", "ilk10"], group),
                                             ("aday getirici", ["ilk100", "ilk5", "ilk1"], group[group.reranker.isna()])):
                top = (candidates.dropna(subset=order[0]).sort_values(order, ascending=False)
                       .drop_duplicates(CONFIG).head(3))
                parts.append(top.assign(liste=f"{label} ({order[0]})"))
            groups.append((f"{pool} havuz · {task} (30 sorgu)", pd.concat(parts)))
    return groups


def write_sheet(writer, name, groups, columns, widths):
    """Her grubun önüne kalın bir başlık satırı koyarak yazar."""
    rows = []
    for title, group in groups:
        rows.append({columns[0]: title})
        rows.extend(group[columns].to_dict("records"))
    pd.DataFrame(rows, columns=columns).to_excel(writer, index=False, sheet_name=name)
    sheet = writer.sheets[name]
    for i, row in enumerate(rows, start=2):  # 1. satır sütun adları
        if len(row) == 1:
            sheet.cell(i, 1).font = Font(bold=True, size=12)
    for letter, width in zip("ABCDEFGHIJKL", widths):
        sheet.column_dimensions[letter].width = width
    sheet.freeze_panes = "A2"


def write_excel(table, best_groups, path):
    sections = [(SECTIONS[s], group) for s, group in table.groupby("bölüm", sort=True)]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        write_sheet(writer, "özet", sections, list(table.columns.drop("bölüm")),
                    (8, 7, 32, 24, 26, 7, 7, 7, 7, 38, 14))
        write_sheet(writer, "en iyiler", best_groups, ["liste", *CONFIG[:5], "ilk1", "ilk5", "ilk10", "ilk100",
                                                       "reranker", "reranker metni"],
                    (24, 7, 7, 32, 30, 10, 7, 7, 7, 7, 38, 14))


if __name__ == "__main__":
    frame = load_rows()
    table = summarize(frame)
    best_groups = best(per_task(frame, table))
    write_excel(table, best_groups, "data/deney_ozeti.xlsx")
    print(table.groupby("bölüm").size().to_string())
    print("yazıldı: data/deney_ozeti.xlsx")
