"""
wczytaj.py — wczytywanie współrzędnych punktów z plików.

Formaty:  .xlsx (własnym czytnikiem, bez openpyxl)  ·  .csv / .txt / .dat

Oba idą przez tę samą logikę: plik zamieniamy najpierw na zwykłą tabelę
komórek, a dopiero potem szukamy w niej punktów. Dzięki temu każdy układ
obsłużony dla arkusza działa też dla tekstu i odwrotnie.

Rozpoznawane układy, w kolejności prób:

  1. PO NAGŁÓWKACH, DŁUGI    kolumna typu (typ/rodzaj/kod) + kolumny X, Y
                             — grupy biorą się z wartości w kolumnie typu
  2. PO NAGŁÓWKACH, SZEROKI   pary kolumn nazwanych X/Y, E/N, easting/northing…
  3. PO KSZTAŁCIE             bloki kolumn liczbowych rozdzielone pustą kolumną;
                             kolumna numeracji 1,2,3… jest odrzucana, a nazwa
                             grupy brana z napisu nad blokiem

Układ 1 to format, w którym program sam eksportuje geometrię, więc własne wyniki
da się wczytać z powrotem. Układ 3 ratuje pliki bez żadnych nagłówków.

Czego program NIE rozstrzygnie za użytkownika: czy pierwsza kolumna to wschód
czy północ. W geodezji X bywa północą, a na mapie X jest osią poziomą — z samych
liczb tego nie widać, stąd przełącznik `zamien_xy`.
"""

from __future__ import annotations

import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Grupa", "wczytaj_punkty", "wczytaj_tabele"]

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# Nazwy kolumn. Skróty dopasowujemy jako całe słowa, żeby „x" nie trafiło
# w „max" ani „n" w „nazwa".
NAZWY_1 = ("x", "e", "easting", "wschod", "wschód", "wsp_x", "wspx", "xm", "x_m")
NAZWY_2 = ("y", "n", "northing", "polnoc", "północ", "wsp_y", "wspy", "ym", "y_m")
NAZWY_TYPU = ("typ", "type", "rodzaj", "grupa", "kod", "klasa", "kategoria")

# Nagłówki kolumn, które NIE są nazwą grupy punktów. Bez tej listy podpis „nr"
# nad kolumną numeracji zostałby wzięty za nazwę profilu.
NAZWY_KOLUMN = NAZWY_1 + NAZWY_2 + NAZWY_TYPU + (
    "nr", "numer", "lp", "id", "no", "num", "punkt", "pkt", "point",
    "stacja", "station", "z", "h", "rzedna", "rzędna", "wysokosc", "wysokość")

# po tych słowach zgadujemy, że grupa to punkty wzbudzenia
NAZWY_ZRODEL = ("strzał", "strzal", "shot", "źródł", "zrodl", "source", "wybuch")
SKROTY_ZRODEL = r"\b(ps|sp)\b"


@dataclass
class Grupa:
    """Jedna grupa punktów: nazwa + współrzędne (N, 2)."""

    nazwa: str
    punkty: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.punkty)

    @property
    def wyglada_na_zrodla(self) -> bool:
        """Czy nazwa sugeruje punkty strzałowe. Tylko podpowiedź, nie wyrok."""
        n = self.nazwa.lower()
        return (any(w in n for w in NAZWY_ZRODEL)
                or re.search(SKROTY_ZRODEL, n) is not None)


# =====================================================================
# komórki → tabela
# =====================================================================
def _na_liczbe(v) -> float | None:
    """Liczba z komórki, także gdy zapisana tekstem i z przecinkiem."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().replace(" ", "").replace(" ", "")
    if not s or s.count(",") > 1:
        return None
    s = s.replace(",", ".")
    try:
        f = float(s)
    except ValueError:
        return None
    return f if np.isfinite(f) else None


def _naglowek(v) -> str:
    """Nazwa kolumny sprowadzona do porównywalnej postaci: 'X [m]' → 'x'."""
    if not isinstance(v, str):
        return ""
    s = re.sub(r"[\[\(].*?[\]\)]", " ", v).strip().lower()
    return re.sub(r"[^0-9a-ząćęłńóśźż_]+", "", s)


def _litery_na_numer(litery: str) -> int:
    n = 0
    for ch in litery:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n


def _tabele_xlsx(sciezka: str) -> list[tuple[str, list[list]]]:
    """Wszystkie arkusze jako listy wierszy; puste komórki jako None."""
    with zipfile.ZipFile(sciezka) as z:
        napisy: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            napisy = ["".join(t.text or "" for t in si.iter(NS + "t"))
                      for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si")]
        nazwy = []
        if "xl/workbook.xml" in z.namelist():
            nazwy = [s.get("name") for s in
                     ET.fromstring(z.read("xl/workbook.xml")).iter(NS + "sheet")]
        pliki = sorted(n for n in z.namelist()
                       if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if not pliki:
            raise ValueError("plik xlsx nie zawiera żadnego arkusza")

        wynik = []
        for i, plik in enumerate(pliki):
            kom: dict[tuple[int, int], object] = {}
            for wiersz in ET.fromstring(z.read(plik)).iter(NS + "row"):
                r = int(wiersz.get("r"))
                for c in wiersz.iter(NS + "c"):
                    k = _litery_na_numer("".join(ch for ch in c.get("r") if ch.isalpha()))
                    typ = c.get("t")
                    # Excel zapisuje napisy albo we wspólnej tablicy (t="s"),
                    # albo wprost w komórce (t="inlineStr") — bywa i tak, i tak
                    if typ == "inlineStr":
                        blok = c.find(NS + "is")
                        tekst = ("".join(t.text or "" for t in blok.iter(NS + "t"))
                                 if blok is not None else "")
                        if tekst:
                            kom[(r, k)] = tekst
                        continue
                    v = c.find(NS + "v")
                    if v is None or v.text is None:
                        continue
                    if typ == "s":
                        try:
                            nr = int(v.text)
                        except ValueError:
                            continue
                        if nr < len(napisy):
                            kom[(r, k)] = napisy[nr]
                    else:
                        kom[(r, k)] = v.text
            if not kom:
                continue
            r_max = max(r for r, _ in kom)
            k_max = max(k for _, k in kom)
            tab = [[kom.get((r, k)) for k in range(1, k_max + 1)]
                   for r in range(1, r_max + 1)]
            wynik.append((nazwy[i] if i < len(nazwy) else f"arkusz {i + 1}", tab))
    return wynik


def _tabele_csv(sciezka: str) -> list[tuple[str, list[list]]]:
    with open(sciezka, "r", encoding="utf-8-sig", errors="replace") as f:
        tekst = f.read()
    linie = [l for l in tekst.splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    if not linie:
        raise ValueError("plik nie zawiera danych")
    try:
        sep = csv.Sniffer().sniff("\n".join(linie[:25]), delimiters=",;\t| ").delimiter
    except csv.Error:
        sep = max(",;\t| ", key=lambda s: linie[0].count(s))
    tab = [[p.strip() or None for p in l.split(sep)] for l in linie]
    szer = max(len(w) for w in tab)
    tab = [w + [None] * (szer - len(w)) for w in tab]
    return [("", tab)]


def wczytaj_tabele(sciezka: str) -> list[tuple[str, list[list]]]:
    """Plik jako surowe tabele — przydatne do podglądu i testów."""
    low = sciezka.lower()
    if low.endswith(".xlsx"):
        return _tabele_xlsx(sciezka)
    if low.endswith((".csv", ".txt", ".dat", ".tsv")):
        return _tabele_csv(sciezka)
    raise ValueError(f"nieobsługiwany format pliku: {sciezka}")


# =====================================================================
# tabela → grupy punktów
# =====================================================================
def _kolumny_liczbowe(tab: list[list], od: int,
                      minimum: int = 2) -> dict[int, list[tuple[int, float]]]:
    """
    {numer kolumny: [(wiersz, wartość), …]} dla kolumn z co najmniej `minimum`
    liczbami. Przy szukaniu nagłówka wystarczy jedna — plik z jednym punktem
    też ma prawo się wczytać. Przy zgadywaniu po kształcie wymagamy dwóch,
    żeby pojedyncza liczba gdzieś z boku nie udawała kolumny.
    """
    szer = max((len(w) for w in tab), default=0)
    out = {}
    for k in range(szer):
        wart = [(r, _na_liczbe(tab[r][k])) for r in range(od, len(tab))
                if k < len(tab[r])]
        wart = [(r, v) for r, v in wart if v is not None]
        if len(wart) >= minimum:
            out[k] = wart
    return out


def _wiersz_naglowka(tab: list[list]) -> int:
    """
    Numer wiersza z nazwami kolumn albo -1.

    Nagłówkiem jest ostatni wiersz przed danymi, w którym są napisy, a nie
    liczby. Szukamy tylko w kilku pierwszych wierszach, bo dalej to już dane.
    """
    najwczesniej = None
    for k, wart in _kolumny_liczbowe(tab, 0, minimum=1).items():
        r = wart[0][0]
        najwczesniej = r if najwczesniej is None else min(najwczesniej, r)
    if najwczesniej is None:
        return -1
    for r in range(najwczesniej - 1, -1, -1):
        if any(isinstance(v, str) and _naglowek(v) for v in tab[r]):
            return r
    return -1


def _po_nazwach(tab: list[list], r_nag: int, zrodlo: str) -> list[Grupa] | None:
    """Układy rozpoznane po nazwach kolumn — najpewniejsze, więc próbowane pierwsze."""
    if r_nag < 0:
        return None
    nag = {k: _naglowek(v) for k, v in enumerate(tab[r_nag])}
    i1 = [k for k, n in nag.items() if n in NAZWY_1]
    i2 = [k for k, n in nag.items() if n in NAZWY_2]
    if not i1 or not i2:
        return None
    ityp = next((k for k, n in nag.items() if n in NAZWY_TYPU), None)
    dane = tab[r_nag + 1:]

    # --- układ długi: jedna para kolumn + kolumna typu
    if ityp is not None:
        wg: dict[str, list[tuple[float, float]]] = {}
        for w in dane:
            if max(i1[0], i2[0], ityp) >= len(w):
                continue
            a, b = _na_liczbe(w[i1[0]]), _na_liczbe(w[i2[0]])
            if a is None or b is None:
                continue
            klucz = str(w[ityp]).strip() if w[ityp] is not None else "(bez typu)"
            wg.setdefault(klucz, []).append((a, b))
        grupy = [Grupa(k, np.array(v)) for k, v in wg.items() if v]
        if grupy:
            return grupy

    # --- układ szeroki: kolejne pary X/Y, nazwa z napisu nad parą albo z arkusza
    grupy = []
    poprzedni_koniec = -1
    for nr, (ka, kb) in enumerate(zip(sorted(i1), sorted(i2)), start=1):
        pary = []
        for w in dane:
            if max(ka, kb) >= len(w):
                continue
            a, b = _na_liczbe(w[ka]), _na_liczbe(w[kb])
            if a is not None and b is not None:
                pary.append((a, b))
        if not pary:
            continue
        # Nazwy szukamy też w kolumnach tuż przed parą, bo podpis grupy stoi
        # zwykle nad kolumną numeracji — ale nie dalej niż do końca poprzedniej
        # pary, inaczej druga grupa przejęłaby nazwę pierwszej.
        od = max(poprzedni_koniec + 1, ka - 2, 0)
        nazwa = _nazwa_bloku(tab, r_nag, list(range(od, kb + 1))) or (
            zrodlo or f"grupa {nr}" if len(i1) > 1 else zrodlo or "punkty")
        grupy.append(Grupa(nazwa, np.array(pary)))
        poprzedni_koniec = kb
    return grupy or None


def _nazwa_bloku(tab: list[list], do_wiersza: int, kolumny: list[int]) -> str:
    """Napis stojący nad blokiem kolumn — zwykle nazwa profilu albo grupy."""
    for r in range(do_wiersza, -1, -1):
        for k in kolumny:
            if k < len(tab[r]) and isinstance(tab[r][k], str):
                s = tab[r][k].strip()
                if s and _na_liczbe(s) is None and _naglowek(s) not in NAZWY_KOLUMN:
                    return s
    return ""


def _czy_numeracja(v: np.ndarray) -> bool:
    """Kolumna 1, 2, 3, … czyli numeracja, a nie współrzędna."""
    return len(v) > 1 and np.array_equal(v, np.arange(1, len(v) + 1))


def _po_ksztalcie(tab: list[list], r_nag: int, zrodlo: str) -> list[Grupa]:
    """
    Ostatnia deska ratunku: same liczby, bez czytelnych nagłówków.

    Kolumny liczbowe grupujemy w bloki rozdzielone pustą kolumną. W bloku
    odrzucamy numerację, a ze reszty bierzemy dwie pierwsze kolumny — bo
    konwencja „numer, X, Y, [Z]" jest powszechna, a wysokość, jeśli jest,
    stoi za współrzędnymi poziomymi.
    """
    kol = _kolumny_liczbowe(tab, r_nag + 1)
    if not kol:
        return []
    bloki, biezacy = [], []
    for k in sorted(kol):
        if biezacy and k != biezacy[-1] + 1:
            bloki.append(biezacy)
            biezacy = []
        biezacy.append(k)
    if biezacy:
        bloki.append(biezacy)

    grupy = []
    for blok in bloki:
        # Blok szerszy niż „numer, X, Y, Z" to prawie na pewno nie lista punktów,
        # tylko tabela wyników — bez nagłówków nie da się zgadnąć, co jest czym,
        # więc lepiej odmówić niż wczytać przypadkowe dwie kolumny.
        if not 2 <= len(blok) <= 4:
            continue
        wsp = {r for k in blok for r, _ in kol[k]}
        wiersze = sorted(r for r in wsp
                         if all(any(rr == r for rr, _ in kol[k]) for k in blok))
        if len(wiersze) < 2:
            continue
        dane = {k: np.array([v for r, v in kol[k] if r in set(wiersze)]) for k in blok}
        kandydaci = [k for k in blok if not _czy_numeracja(dane[k])] or blok
        if len(kandydaci) < 2:
            kandydaci = blok
        ka, kb = kandydaci[0], kandydaci[1]
        nazwa = _nazwa_bloku(tab, max(r_nag, 0), blok)
        if not nazwa:
            nazwa = (f"{zrodlo} " if zrodlo else "") + f"kolumny {ka + 1}–{kb + 1}"
        grupy.append(Grupa(nazwa, np.column_stack([dane[ka], dane[kb]])))
    return grupy


def _grupy_z_tabeli(tab: list[list], zrodlo: str) -> list[Grupa]:
    r_nag = _wiersz_naglowka(tab)
    return _po_nazwach(tab, r_nag, zrodlo) or _po_ksztalcie(tab, r_nag, zrodlo)


# =====================================================================
def wczytaj_punkty(sciezka: str, zamien_xy: bool = False) -> list[Grupa]:
    """
    Wczytuje grupy punktów z pliku.

    zamien_xy zamienia kolumny miejscami — patrz uwaga w nagłówku modułu.
    """
    tabele = wczytaj_tabele(sciezka)
    wiele = len(tabele) > 1
    grupy: list[Grupa] = []
    for nazwa_arkusza, tab in tabele:
        for g in _grupy_z_tabeli(tab, nazwa_arkusza if wiele else ""):
            if len(g.punkty):
                grupy.append(g)
    if not grupy:
        raise ValueError(
            "nie znaleziono par współrzędnych — nazwij kolumny (X i Y albo "
            "typ, X, Y) lub rozdziel grupy punktów pustą kolumną")
    if zamien_xy:
        grupy = [Grupa(g.nazwa, g.punkty[:, ::-1]) for g in grupy]
    return grupy


if __name__ == "__main__":
    import sys

    for sc in sys.argv[1:] or ["Wspolrzedne.xlsx"]:
        print(f"\n{sc}")
        try:
            for g in wczytaj_punkty(sc):
                p = g.punkty
                print(f"  {g.nazwa:22s} {len(p):4d} pkt   "
                      f"{p[:, 0].min():.1f}..{p[:, 0].max():.1f}  ×  "
                      f"{p[:, 1].min():.1f}..{p[:, 1].max():.1f}"
                      f"{'   [źródła?]' if g.wyglada_na_zrodla else ''}")
        except ValueError as e:
            print(f"  {e}")
