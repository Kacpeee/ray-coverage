"""
test_wczytaj.py — testy wczytywania współrzędnych.

Uruchomienie:  python test_wczytaj.py     (albo: pytest test_wczytaj.py)

Sens: parser zgaduje, które kolumny są współrzędnymi, a które numeracją, i gdzie
kończy się jedna grupa punktów a zaczyna druga. Taki kod łatwo napisać tak, że
działa na jednym pliku i sypie się na następnym — stąd testy na układach, które
realnie się trafiają.
"""

import os
import tempfile

import numpy as np

from wczytaj import Grupa, wczytaj_punkty

TU = os.path.dirname(os.path.abspath(__file__))


def _plik(tresc, rozszerzenie=".csv"):
    f = tempfile.NamedTemporaryFile("w", suffix=rozszerzenie, delete=False,
                                    encoding="utf-8")
    f.write(tresc)
    f.close()
    return f.name


# ------------------------------------------------------------- prawdziwy plik
def test_wspolrzedne_xlsx():
    """Arkusz z pomiaru: trzy grupy obok siebie, każda 'numer, X, Y'."""
    sciezka = os.path.join(TU, "Wspolrzedne.xlsx")
    if not os.path.exists(sciezka):
        return                                    # plik z danymi bywa nieobecny
    grupy = {g.nazwa: g for g in wczytaj_punkty(sciezka)}
    assert set(grupy) == {"T-271", "W-270a", "PS"}
    assert len(grupy["T-271"]) == 48
    assert len(grupy["W-270a"]) == 48
    assert len(grupy["PS"]) == 15
    assert grupy["PS"].wyglada_na_zrodla
    assert not grupy["T-271"].wyglada_na_zrodla
    # kolumna numeracji ma zostać odrzucona, a nie wzięta za współrzędną
    p = grupy["T-271"].punkty
    assert p.shape == (48, 2)
    assert p[:, 0].min() > 5_000_000 and p[:, 1].min() > 5_000_000
    # geofony co 5 m
    d = np.hypot(*np.diff(p, axis=0).T)
    assert abs(np.median(d) - 5.0) < 0.05


# -------------------------------------------------------------- układ długi
def test_csv_z_kolumna_typu():
    """Format, w którym program sam eksportuje geometrię."""
    p = _plik("typ,x,y\nPS,10.5,20.5\nPS,11,21\nczujnik,30,40\nczujnik,31,41\n")
    grupy = {g.nazwa: g for g in wczytaj_punkty(p)}
    assert set(grupy) == {"PS", "czujnik"}
    assert len(grupy["PS"]) == 2 and len(grupy["czujnik"]) == 2
    assert grupy["PS"].wyglada_na_zrodla
    assert tuple(grupy["PS"].punkty[0]) == (10.5, 20.5)
    os.unlink(p)


def test_csv_z_komentarzem_i_srednikiem():
    p = _plik("# obszar 0,0 .. 100,200\ntyp;x;y\nwybuch;1;2\nczujnik;3;4\n")
    grupy = {g.nazwa: g for g in wczytaj_punkty(p)}
    assert set(grupy) == {"wybuch", "czujnik"}
    assert grupy["wybuch"].wyglada_na_zrodla        # stara nazwa też ma działać
    os.unlink(p)


# ------------------------------------------------------------- układ szeroki
def test_csv_dwie_kolumny_bez_naglowka():
    p = _plik("1.5 2.5\n3.5 4.5\n5.5 6.5\n")
    grupy = wczytaj_punkty(p)
    assert len(grupy) == 1 and len(grupy[0]) == 3
    assert tuple(grupy[0].punkty[0]) == (1.5, 2.5)
    os.unlink(p)


def test_kolumna_numeracji_odrzucana():
    p = _plik("nr,x,y\n1,100.5,200.5\n2,101.5,201.5\n3,102.5,202.5\n")
    g = wczytaj_punkty(p)[0]
    assert g.punkty.shape == (3, 2)
    assert tuple(g.punkty[0]) == (100.5, 200.5)     # nie (1, 100.5)
    os.unlink(p)


def test_dwie_grupy_rozdzielone_pusta_kolumna():
    """Układ z arkusza pomiarowego: profile obok siebie, przedzielone pustą."""
    p = _plik("P1,,,P2,\n"
              "1.5,2.5,,10.5,20.5\n"
              "3.5,4.5,,30.5,40.5\n"
              "5.5,6.5,,50.5,60.5\n")
    grupy = {g.nazwa: g for g in wczytaj_punkty(p)}
    assert set(grupy) == {"P1", "P2"}, list(grupy)
    assert len(grupy["P1"]) == 3 and len(grupy["P2"]) == 3
    assert tuple(grupy["P2"].punkty[0]) == (10.5, 20.5)
    os.unlink(p)


def test_numer_x_y_z():
    """Bardzo częsty układ z pomiarów: numer, współrzędne i wysokość."""
    p = _plik("1 100.5 200.5 -750.2\n2 101.5 201.5 -751.0\n3 102.5 202.5 -749.8\n")
    g = wczytaj_punkty(p)[0]
    assert g.punkty.shape == (3, 2)
    assert tuple(g.punkty[0]) == (100.5, 200.5)      # bez numeru i bez Z
    os.unlink(p)


def test_naglowki_easting_northing():
    p = _plik("nr;easting;northing\n1;5581670,03;5713812,59\n2;5581672,96;5713808,54\n")
    g = wczytaj_punkty(p)[0]
    assert g.punkty.shape == (2, 2)
    assert abs(g.punkty[0][0] - 5581670.03) < 1e-6   # przecinek dziesiętny
    os.unlink(p)


def test_liczby_zapisane_tekstem():
    # średnik jako separator — przecinek jest tu znakiem dziesiętnym,
    # a spacja rozdziela tysiące, jak to bywa po eksporcie z Excela
    p = _plik("typ;x;y\nPS; 1 234,5 ; 2 000,25 \n")
    g = wczytaj_punkty(p)[0]
    assert abs(g.punkty[0][0] - 1234.5) < 1e-9, g.punkty
    assert abs(g.punkty[0][1] - 2000.25) < 1e-9
    os.unlink(p)


def test_kolumny_x_y_bez_kolumny_typu():
    p = _plik("X [m],Y [m]\n10,20\n30,40\n")
    grupy = wczytaj_punkty(p)
    assert len(grupy) == 1 and len(grupy[0]) == 2
    assert tuple(grupy[0].punkty[1]) == (30.0, 40.0)
    os.unlink(p)


# ---------------------------------------------------------------- zamiana osi
def test_zamiana_xy():
    p = _plik("typ,x,y\nPS,1,2\nPS,3,4\n")
    zwykle = wczytaj_punkty(p)[0].punkty
    zamien = wczytaj_punkty(p, zamien_xy=True)[0].punkty
    assert np.array_equal(zamien, zwykle[:, ::-1])
    os.unlink(p)


# ---------------------------------------------------------------- odmowy
def test_plik_bez_wspolrzednych_odrzucony():
    p = _plik("ala,ma,kota\nkot,ma,ale\n")
    try:
        wczytaj_punkty(p)
    except ValueError:
        os.unlink(p)
        return
    raise AssertionError("plik bez liczb powinien rzucić ValueError")


def test_tabela_wynikow_nie_jest_lista_punktow():
    """siatka_pokrycie.csv ma 9 kolumn liczbowych — to nie są punkty."""
    p = _plik("ix,iy,x_srodka,y_srodka,liczba_promieni,suma_drog_m,"
              "waga_promienie,waga_dlugosc,anizotropia\n"
              "0,0,5,5,12,80.5,0.5,0.4,0.7\n0,1,5,15,14,90.5,0.6,0.5,0.8\n")
    try:
        wczytaj_punkty(p)
    except ValueError:
        os.unlink(p)
        return
    raise AssertionError("tabela wyników nie powinna być czytana jako punkty")


def test_nieznany_format_odrzucony():
    try:
        wczytaj_punkty("cokolwiek.docx")
    except ValueError:
        return
    raise AssertionError("nieobsługiwane rozszerzenie powinno rzucić ValueError")


# ------------------------------------------------------------------ xlsx
def _xlsx(arkusze):
    """
    Minimalny plik .xlsx zbudowany z niczego.

    Dzięki temu testy formatów arkusza nie zależą od zewnętrznych plików
    ani od openpyxl, którego program celowo nie wymaga.
    arkusze: {nazwa: [[komórka, …], …]}, komórka = str albo liczba.
    """
    import zipfile
    from xml.sax.saxutils import escape

    def litera(k):
        s = ""
        while k > 0:
            k, r = divmod(k - 1, 26)
            s = chr(65 + r) + s
        return s

    f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    f.close()
    with zipfile.ZipFile(f.name, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                   'package/2006/content-types"><Default Extension="xml" '
                   'ContentType="application/xml"/></Types>')
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        z.writestr("xl/workbook.xml",
                   f'<workbook xmlns="{ns}"><sheets>' +
                   "".join(f'<sheet name="{escape(n)}" sheetId="{i+1}"/>'
                           for i, n in enumerate(arkusze)) +
                   "</sheets></workbook>")
        for i, (_nazwa, wiersze) in enumerate(arkusze.items(), start=1):
            xml = [f'<worksheet xmlns="{ns}"><sheetData>']
            for r, w in enumerate(wiersze, start=1):
                xml.append(f'<row r="{r}">')
                for k, v in enumerate(w, start=1):
                    if v is None or v == "":
                        continue
                    ref = f"{litera(k)}{r}"
                    if isinstance(v, str):
                        xml.append(f'<c r="{ref}" t="inlineStr"><is><t>'
                                   f'{escape(v)}</t></is></c>')
                    else:
                        xml.append(f'<c r="{ref}"><v>{v}</v></c>')
                xml.append("</row>")
            xml.append("</sheetData></worksheet>")
            z.writestr(f"xl/worksheets/sheet{i}.xml", "".join(xml))
    return f.name


def test_xlsx_uklad_dlugi_z_kolumna_typu():
    """Arkusz w układzie 'typ, X, Y' — dotąd czytany był tylko z CSV."""
    p = _xlsx({"Arkusz1": [["typ", "X", "Y"],
                           ["PS", 10.0, 20.0], ["PS", 11.0, 21.0],
                           ["geofon", 30.0, 40.0]]})
    grupy = {g.nazwa: g for g in wczytaj_punkty(p)}
    assert set(grupy) == {"PS", "geofon"}, list(grupy)
    assert len(grupy["PS"]) == 2 and len(grupy["geofon"]) == 1
    os.unlink(p)


def test_xlsx_liczby_zapisane_tekstem():
    """Współrzędne wklejone jako tekst — częste po eksporcie z innego programu."""
    p = _xlsx({"Arkusz1": [["nr", "x", "y"],
                           [1, "5581670,03", "5713812,59"],
                           [2, "5581672,96", "5713808,54"]]})
    g = wczytaj_punkty(p)[0]
    assert g.punkty.shape == (2, 2)
    assert abs(g.punkty[0][0] - 5581670.03) < 1e-6
    os.unlink(p)


def test_xlsx_kilka_arkuszy():
    p = _xlsx({"strzaly": [["x", "y"], [1.0, 2.0], [3.0, 4.0]],
               "geofony": [["x", "y"], [5.0, 6.0], [7.0, 8.0]]})
    grupy = wczytaj_punkty(p)
    assert len(grupy) == 2, [g.nazwa for g in grupy]
    assert {len(g) for g in grupy} == {2}
    assert any("strzaly" in g.nazwa for g in grupy), [g.nazwa for g in grupy]
    os.unlink(p)


def test_xlsx_sasiadujace_grupy_bez_pustej_kolumny():
    """Dwa profile obok siebie, bez przerwy — rozpoznane po nagłówkach X/Y."""
    p = _xlsx({"Arkusz1": [["prof A", None, "prof B", None],
                           ["X", "Y", "X", "Y"],
                           [1.0, 2.0, 10.0, 20.0],
                           [3.0, 4.0, 30.0, 40.0]]})
    grupy = {g.nazwa: g for g in wczytaj_punkty(p)}
    assert set(grupy) == {"prof A", "prof B"}, list(grupy)
    assert tuple(grupy["prof B"].punkty[0]) == (10.0, 20.0)
    os.unlink(p)


def test_xlsx_wiele_grup_roznej_dlugosci():
    """Realny wariant: kilka profili odbiorników o różnej liczbie punktów."""
    def blok(n, s):
        return [[i + 1, 1000.0 + s + i, 2000.0 + s + 2 * i] for i in range(n)]
    dlugosci = (48, 48, 30, 12)
    wiersze = [["Profil A", None, None, None, "Profil B", None, None, None,
                "Profil C", None, None, None, "Wzbudzenia", None, None]]
    for i in range(max(dlugosci)):
        w = []
        for k, n in enumerate(dlugosci):
            w += (blok(n, k * 100)[i] if i < n else [None, None, None])
            if k < 3:
                w.append(None)                    # pusta kolumna rozdzielająca
        wiersze.append(w)
    p = _xlsx({"Arkusz1": wiersze})
    grupy = {g.nazwa: g for g in wczytaj_punkty(p)}
    assert set(grupy) == {"Profil A", "Profil B", "Profil C", "Wzbudzenia"}, list(grupy)
    assert [len(grupy[n]) for n in
            ("Profil A", "Profil B", "Profil C", "Wzbudzenia")] == list(dlugosci)
    assert grupy["Wzbudzenia"].punkty.shape == (12, 2)   # numeracja odrzucona
    os.unlink(p)


def test_xlsx_nazwa_grupy_nad_kolumna_numeracji():
    """Podpis profilu stoi nad 'nr', a nie nad 'X' — i grupy się nie mylą."""
    wiersze = [["Profil A", None, None, "Profil B", None, None],
               ["nr", "X", "Y", "nr", "X", "Y"]]
    for i in range(5):
        wiersze.append([i + 1, 10.0 + i, 20.0 + i, i + 1, 100.0 + i, 200.0 + i])
    p = _xlsx({"Arkusz1": wiersze})
    grupy = {g.nazwa: g for g in wczytaj_punkty(p)}
    assert set(grupy) == {"Profil A", "Profil B"}, list(grupy)
    assert tuple(grupy["Profil B"].punkty[0]) == (100.0, 200.0)
    os.unlink(p)


# ---------------------------------------------------------------- nazewnictwo
def test_rozpoznawanie_zrodel_po_nazwie():
    tak = ["PS", "ps", "punkty strzałowe", "SHOT", "źródła", "wybuchy", "SP"]
    nie = ["T-271", "W-270a", "geofony", "czujniki", "profil A", "gips"]
    for n in tak:
        assert Grupa(n, np.zeros((1, 2))).wyglada_na_zrodla, n
    for n in nie:
        assert not Grupa(n, np.zeros((1, 2))).wyglada_na_zrodla, n


# ----------------------------------------------------------------------
if __name__ == "__main__":
    testy = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bledy = 0
    for t in testy:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            bledy += 1
            print(f"  BŁĄD {t.__name__}: {e}")
    print(f"\n{len(testy) - bledy}/{len(testy)} testów przeszło")
    raise SystemExit(1 if bledy else 0)
