"""
test_raycov.py — testy rdzenia.

Uruchomienie:  python test_raycov.py        (albo: pytest test_raycov.py)

Sens tych testów: algorytm traversalu łatwo napisać tak, że "prawie działa" —
gubi ostatnią komórkę, liczy jedną dwa razy albo wychodzi poza siatkę przy
promieniu idealnie poziomym. Na mapie pokrycia takiego błędu nie widać gołym
okiem, bo mapa i tak wygląda sensownie. Dlatego trzeba to sprawdzić liczbowo.
"""

import numpy as np

from raycov import Grid, clip_segment, compute, edge_points, snap, traverse

TOL = 1e-9


def _cells(p0, p1, g):
    return list(traverse(p0, p1, g))


# ---------------------------------------------------------------- siatka
def test_grid_geometry():
    g = Grid.from_size(100, 200, 10, 20)
    assert g.dx == 10 and g.dy == 10
    assert g.shape == (20, 10)
    assert g.n_cells == 200
    xc, yc = g.centers()
    assert xc[0, 0] == 5 and yc[0, 0] == 5
    assert xc[-1, -1] == 95 and yc[-1, -1] == 195


def test_grid_rejects_nonsense():
    for bad in [dict(xmin=0, ymin=0, xmax=0, ymax=10, nx=2, ny=2),
                dict(xmin=0, ymin=0, xmax=10, ymax=10, nx=0, ny=2)]:
        try:
            Grid(**bad)
        except ValueError:
            continue
        raise AssertionError(f"Grid({bad}) powinien rzucić ValueError")


# ---------------------------------------------------------- przypadki proste
def test_horizontal_ray_crosses_every_column_once():
    g = Grid.from_size(100, 200, 10, 20)
    cells = _cells((0, 55), (100, 55), g)
    assert len(cells) == g.nx
    assert [c[0] for c in cells] == list(range(g.nx))
    assert all(c[1] == 5 for c in cells)                  # y=55 → wiersz 5
    assert all(abs(c[2] - g.dx) < TOL for c in cells)


def test_vertical_ray_crosses_every_row_once():
    g = Grid.from_size(100, 200, 10, 20)
    cells = _cells((25, 0), (25, 200), g)
    assert len(cells) == g.ny
    assert [c[1] for c in cells] == list(range(g.ny))
    assert all(abs(c[2] - g.dy) < TOL for c in cells)


def test_diagonal_through_square_grid():
    g = Grid.from_size(100, 100, 10, 10)
    cells = _cells((0, 0), (100, 100), g)
    assert len(cells) == 10                               # przekątna: po jednej na wiersz
    total = sum(c[2] for c in cells)
    assert abs(total - np.hypot(100, 100)) < 1e-9


def test_ray_shorter_than_one_cell():
    g = Grid.from_size(100, 200, 10, 20)
    cells = _cells((11, 11), (14, 13), g)
    assert len(cells) == 1
    assert cells[0][:2] == (1, 1)
    assert abs(cells[0][2] - np.hypot(3, 2)) < TOL


def test_ray_outside_grid_is_empty():
    g = Grid.from_size(100, 200, 10, 20)
    assert _cells((-50, -50), (-10, -10), g) == []
    assert _cells((200, 50), (300, 50), g) == []


def test_zero_length_ray_is_empty():
    g = Grid.from_size(100, 200, 10, 20)
    assert _cells((50, 50), (50, 50), g) == []


def test_ray_from_outside_is_clipped():
    g = Grid.from_size(100, 200, 10, 20)
    total = sum(c[2] for c in _cells((-500, 100), (500, 100), g))
    assert abs(total - 100) < TOL                          # tylko część w obszarze


# ---------------------------------------------------- własności ogólne
def test_path_length_is_conserved():
    """Suma odcinków w komórkach == długość promienia wewnątrz siatki."""
    g = Grid.from_size(137.5, 61.25, 17, 23)               # celowo niekrągłe
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(5000):
        p0 = rng.uniform(-60, 200, 2)
        p1 = rng.uniform(-60, 200, 2)
        got = sum(c[2] for c in _cells(p0, p1, g))
        seg = clip_segment(p0, p1, g)
        want = 0.0 if seg is None else float(np.hypot(*(np.subtract(seg[1], seg[0]))))
        worst = max(worst, abs(got - want))
    assert worst < 1e-9, f"największy błąd długości: {worst}"


def test_each_cell_visited_at_most_once():
    g = Grid.from_size(100, 200, 13, 29)
    rng = np.random.default_rng(7)
    for _ in range(3000):
        cells = _cells(rng.uniform(0, 100, 2), rng.uniform(0, 200, 2), g)
        keys = [(c[0], c[1]) for c in cells]
        assert len(keys) == len(set(keys))


def test_indices_always_inside_grid():
    g = Grid.from_size(100, 200, 11, 7)
    rng = np.random.default_rng(99)
    for _ in range(3000):
        for ix, iy, _seg in traverse(rng.uniform(-30, 130, 2), rng.uniform(-30, 230, 2), g):
            assert 0 <= ix < g.nx and 0 <= iy < g.ny


def test_direction_does_not_matter():
    """Promień w obie strony musi dać ten sam wynik — inaczej geometria
    źródło↔czujnik zmieniałaby mapę, co byłoby bez sensu fizycznie."""
    g = Grid.from_size(100, 200, 9, 19)
    rng = np.random.default_rng(3)
    for _ in range(500):
        p0, p1 = rng.uniform(0, 100, 2), rng.uniform(0, 200, 2)
        fwd = {(c[0], c[1]): c[2] for c in _cells(p0, p1, g)}
        bwd = {(c[0], c[1]): c[2] for c in _cells(p1, p0, g)}
        assert fwd.keys() == bwd.keys()
        assert all(abs(fwd[k] - bwd[k]) < TOL for k in fwd)


# ---------------------------------------------------------------- pokrycie
def test_coverage_matches_traversal():
    g = Grid.from_size(100, 200, 10, 20)
    cov = compute(g, [(0, 100)], [(100, 100)])
    assert cov.n_rays == 1
    assert cov.hits.sum() == len(_cells((0, 100), (100, 100), g))
    assert abs(cov.length.sum() - 100) < TOL


def _compute_wolno(g, src, rec, n_sektorow=60):
    """Wynik złożony pętlą po traverse() — wzorzec dla wektorowego compute()."""
    import math
    hits = np.zeros(g.shape, dtype=np.int64)
    length = np.zeros(g.shape)
    mxx = np.zeros(g.shape)
    sektory = np.zeros((n_sektorow, *g.shape), dtype=bool)
    rowsum = np.zeros(len(src) * len(rec))
    for k, (i, j) in enumerate((i, j) for i in range(len(src)) for j in range(len(rec))):
        s, r = np.asarray(src[i], float), np.asarray(rec[j], float)
        d = r - s
        n = math.hypot(d[0], d[1])
        if n < 1e-12 * g.scale:
            continue
        ux, uy = d[0] / n, d[1] / n
        sekt = min(int(math.atan2(uy, ux) % math.pi / math.pi * n_sektorow + 1e-9),
                   n_sektorow - 1)
        for ix, iy, seg in traverse(s, r, g):
            hits[iy, ix] += 1
            length[iy, ix] += seg
            mxx[iy, ix] += seg * ux * ux
            sektory[sekt, iy, ix] = True
            rowsum[k] += seg
    return hits, length, mxx, sektory, rowsum


def test_compute_zgadza_sie_z_petla_po_traverse():
    """
    compute() liczy wszystkie promienie hurtem, macierzami — traverse() chodzi
    po jednym. Muszą wychodzić te same liczby, inaczej szybsza droga po cichu
    zmienia wyniki. Geometria celowo z przypadkami brzegowymi (promień poziomy,
    pionowy, po granicy kratek, wychodzący poza obszar, zerowej długości), bo
    właśnie tam takie przepisanie się wykłada — nie na losowym ukosie.
    """
    rng = np.random.default_rng(2024)
    for nx, ny, x0, y0 in [(10, 20, 0.0, 0.0), (13, 7, 250.0, -100.0),
                           (1, 1, 0.0, 0.0), (17, 23, 5.7e6, 6.4e5)]:
        g = Grid.from_size(100, 200, nx, ny, x0, y0)
        brzegowe = [(g.xmin, g.ymin + 55), (g.xmin + 25, g.ymin),        # osiowe
                    (g.xmin + g.dx, g.ymin + g.dy),                      # w węźle
                    (g.xmin - 500, g.ymin + 100), (g.xmax, g.ymax),      # poza / róg
                    (g.xmin + 50, g.ymin + 50)]
        losowe = np.column_stack([rng.uniform(g.xmin - 40, g.xmax + 40, 9),
                                  rng.uniform(g.ymin - 40, g.ymax + 40, 9)])
        src = np.vstack([brzegowe, losowe])
        rec = np.vstack([[(g.xmax, g.ymin + 55), (g.xmin + 25, g.ymax),
                          (g.xmin + 3 * g.dx, g.ymin + 3 * g.dy),
                          (g.xmax + 500, g.ymin + 100), (g.xmin, g.ymin),
                          (g.xmin + 50, g.ymin + 50)],                   # zerowy
                         losowe[::-1]])

        h, dl, mxx, sekt, rowsum = _compute_wolno(g, src, rec)
        cov = compute(g, src, rec)
        assert np.array_equal(cov.hits, h), f"hits, siatka {nx}×{ny}"
        assert np.allclose(cov.length, dl, rtol=1e-9, atol=1e-7), f"length {nx}×{ny}"
        assert np.allclose(cov._mxx, mxx, rtol=1e-9, atol=1e-7), f"tensor {nx}×{ny}"
        assert np.array_equal(cov.sektory, sekt), f"sektory {nx}×{ny}"
        got = np.bincount(cov.g_rows, weights=cov.g_vals, minlength=cov.n_rays)
        assert np.allclose(got, rowsum, atol=1e-7), f"macierz G {nx}×{ny}"


def test_G_row_sums_equal_ray_lengths():
    """Każdy wiersz macierzy G sumuje się do długości drogi promienia —
    to jest warunek konieczny, żeby inwersja t = G·s miała sens."""
    g = Grid.from_size(100, 200, 10, 20)
    src, rec = edge_points(g, "L", 5), edge_points(g, "P", 7)
    cov = compute(g, src, rec)
    rowsum = np.bincount(cov.g_rows, weights=cov.g_vals, minlength=cov.n_rays)
    k = 0
    for s in src:
        for r in rec:
            assert abs(rowsum[k] - np.hypot(*(r - s))) < 1e-9
            k += 1


def test_anisotropy_bounds():
    g = Grid.from_size(100, 100, 8, 8)
    cov = compute(g, np.vstack([edge_points(g, "L", 6), edge_points(g, "G", 6)]),
                  np.vstack([edge_points(g, "P", 6), edge_points(g, "D", 6)]))
    a = cov.anisotropy[~np.isnan(cov.anisotropy)]
    assert a.size and a.min() >= -TOL and a.max() <= 1 + TOL


def test_parallel_rays_give_maximum_anisotropy():
    """Wiązka promieni idealnie równoległych → anizotropia = 1."""
    g = Grid.from_size(100, 100, 5, 5)
    ys = np.linspace(10, 90, 9)
    cov = compute(g, [(0, y) for y in ys], [(100, y) for y in ys],
                  pairs=[(i, i) for i in range(len(ys))])
    a = cov.anisotropy[cov.hits > 0]
    assert np.allclose(a, 1.0, atol=1e-9)


def test_coincident_source_and_receiver_is_skipped():
    g = Grid.from_size(100, 200, 10, 20)
    cov = compute(g, [(50, 50)], [(50, 50)])
    assert cov.hits.sum() == 0


def test_pairs_argument_limits_rays():
    g = Grid.from_size(100, 200, 10, 20)
    src, rec = edge_points(g, "L", 4), edge_points(g, "P", 4)
    full = compute(g, src, rec)
    subset = compute(g, src, rec, pairs=[(0, 0), (1, 1)])
    assert full.n_rays == 16 and subset.n_rays == 2
    assert subset.hits.sum() < full.hits.sum()


# -------------------------------------------- przesunięty początek układu
def test_origin_shifts_area_not_shape():
    g = Grid.from_size(100, 200, 10, 20, x0=250, y0=-100)
    assert g.origin == (250, -100)
    assert (g.width, g.height) == (100, 200)
    assert g.dx == 10 and g.dy == 10
    xc, yc = g.centers()
    assert xc[0, 0] == 255 and yc[0, 0] == -95


def test_index_of_handles_negative_coordinates():
    """int() ucina w stronę zera — dla ujemnych mylił się o komórkę."""
    g = Grid(-100, -200, 0, 0, 10, 20)
    assert g.index_of(-95, -195) == (0, 0)
    assert g.index_of(-5, -5) == (9, 19)
    assert g.index_of(-0.001, -0.001) == (9, 19)
    assert g.index_of(-1e9, 1e9) == (0, 19)          # poza obszarem → brzeg


def test_coverage_is_translation_invariant():
    """Przesunięcie całej geometrii razem z siatką nie może zmienić wyniku."""
    base = compute(Grid.from_size(100, 200, 10, 20),
                   edge_points(Grid.from_size(100, 200, 10, 20), "L", 5),
                   edge_points(Grid.from_size(100, 200, 10, 20), "P", 7))
    for x0, y0 in [(250, -100), (1e5, 1e5), (5.7e6, 6.4e5)]:
        g = Grid.from_size(100, 200, 10, 20, x0=x0, y0=y0)
        cov = compute(g, edge_points(g, "L", 5), edge_points(g, "P", 7))
        assert np.array_equal(cov.hits, base.hits), f"hits różne przy ({x0}, {y0})"
        assert np.allclose(cov.length, base.length, rtol=1e-9, atol=1e-6)


def test_length_conserved_at_large_coordinates():
    g = Grid.from_size(137.5, 61.25, 17, 23, x0=5.7e6, y0=6.4e5)
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(2000):
        p0 = np.add((5.7e6, 6.4e5), rng.uniform(-60, 200, 2))
        p1 = np.add((5.7e6, 6.4e5), rng.uniform(-60, 200, 2))
        got = sum(c[2] for c in _cells(p0, p1, g))
        seg = clip_segment(p0, p1, g)
        want = 0.0 if seg is None else float(np.hypot(*(np.subtract(seg[1], seg[0]))))
        worst = max(worst, abs(got - want))
    assert worst < 1e-6, f"największy błąd długości: {worst}"


# ---------------------------------------------------------------- snap
def test_snap_rounds_to_whole_values():
    assert snap(12.3, 47.8, 1.0) == (12.0, 48.0)
    assert snap(-2.4, -2.6, 1.0) == (-2.0, -3.0)
    assert snap(12.3, 47.8, 5.0) == (10.0, 50.0)
    assert snap(12.5, 17.5, 5.0) == (15.0, 20.0)      # połówka w górę, nie bankiersko


def test_snap_off_is_identity():
    for step in (0.0, -1.0, float("nan")):
        assert snap(12.345, 6.789, step) == (12.345, 6.789)


def test_snap_respects_origin():
    """Zakotwiczenie w narożniku obszaru: węzły liczone od niego, nie od zera."""
    assert snap(12.3, 4.9, 1.0, origin=(0.5, 0.5)) == (12.5, 4.5)


def test_snapped_point_lands_in_expected_cell():
    g = Grid.from_size(100, 200, 10, 20, x0=250, y0=-100)
    x, y = snap(263.7, -87.2, 1.0)
    assert (x, y) == (264.0, -87.0)
    assert g.index_of(x, y) == (1, 1)


# ---------------------------------------------------------- macierz G
def test_build_matrix_false_skips_G_but_keeps_maps():
    g = Grid.from_size(100, 200, 10, 20)
    src, rec = edge_points(g, "L", 5), edge_points(g, "P", 7)
    full = compute(g, src, rec)
    lean = compute(g, src, rec, build_matrix=False)
    assert np.array_equal(full.hits, lean.hits)
    assert np.array_equal(full.length, lean.length)
    assert lean.g_vals.size == 0 and not lean.has_matrix
    try:
        lean.to_sparse()
    except RuntimeError:
        return
    raise AssertionError("to_sparse() bez macierzy G powinno rzucić RuntimeError")


# ------------------------------------------------------- luka kątowa
def test_luka_katowa_bounds():
    """Jeden promień → prawie cała skala pusta; brak promieni → NaN."""
    g = Grid.from_size(100, 100, 10, 10)
    cov = compute(g, np.array([[0.0, 50.0]]), np.array([[100.0, 50.0]]),
                  build_matrix=False)
    l = cov.luka_katowa
    trafione = l[cov.hits > 0]
    assert trafione.size and np.all(trafione >= 180 - 2 * (180 / 60) - TOL)
    assert np.isnan(l[cov.hits == 0]).all()


def test_luka_katowa_widzi_dziure_ktorej_anizotropia_nie_widzi():
    """
    Sedno sprawy. Źródła po lewej, czujniki po prawej: przez środek obszaru nie
    przechodzi żaden promień bliski pionowi. Anizotropia mówi „prawie idealnie",
    bo tensor 2. rzędu mierzy kształt chmury kierunków, a nie dziury w niej.
    Luka kątowa ten brakujący klin pokazuje — i po to powstała.
    """
    g = Grid.from_size(100, 200, 20, 40)
    cov = compute(g, edge_points(g, "L", 15), edge_points(g, "P", 25),
                  build_matrix=False)
    ix, iy = 10, 20                                  # środek obszaru
    assert cov.hits[iy, ix] > 20
    assert cov.anisotropy[iy, ix] < 0.2              # „chmura okrągła"
    assert cov.luka_katowa[iy, ix] > 45              # a jednak brakuje klina

    # brzeg widzi jeszcze mniej stron — obie miary muszą to potwierdzić
    assert cov.luka_katowa[iy, 0] > cov.luka_katowa[iy, ix]


def test_luka_katowa_zgadza_sie_z_dokladna():
    """Sektory zaniżają lukę najwyżej o dwie swoje szerokości — nie więcej."""
    g = Grid.from_size(100, 200, 20, 40)
    src, rec = edge_points(g, "L", 15), edge_points(g, "P", 25)
    cov = compute(g, src, rec, build_matrix=False)
    szer = 180 / cov.sektory.shape[0]

    for ix, iy in [(10, 20), (7, 23), (5, 10), (19, 20)]:
        katy = []
        for s in src:
            for r in rec:
                d = r - s
                u = d / np.hypot(*d)
                for jx, jy, _ in traverse(s, r, g):
                    if (jx, jy) == (ix, iy):
                        katy.append(np.degrees(np.arctan2(u[1], u[0])) % 180)
        katy = np.array(sorted(katy))
        dokladna = np.diff(np.concatenate([katy, [katy[0] + 180]])).max()
        zmierzona = cov.luka_katowa[iy, ix]
        assert dokladna - 2 * szer - TOL <= zmierzona <= dokladna + TOL


def test_pokrycie_katowe_nie_zalezy_od_gestosci_sektorow():
    """
    Drobniejsze sektory służą luce; pokrycie kątowe scala je z powrotem do
    dwunastu, więc jego wartości mają zostać takie jak przy dwunastu.
    """
    g = Grid.from_size(100, 200, 20, 40)
    src, rec = edge_points(g, "L", 15), edge_points(g, "P", 25)
    a = compute(g, src, rec, build_matrix=False, n_sektorow=12).pokrycie_katowe
    b = compute(g, src, rec, build_matrix=False, n_sektorow=60).pokrycie_katowe
    assert np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))


def test_pokrycie_katowe_zawsze_w_zakresie_0_1():
    """
    Pokrycie jest ułamkiem, więc nie ma prawa wyjść poza 0–1 przy ŻADNEJ
    liczbie sektorów. Gdy n_sektorow nie dzieli się przez SEKTORY_GRUBE,
    scalanie się nie odbywa; sztywne dzielenie przez dwanaście dawało wtedy
    1,667 dla kratki oglądanej ze wszystkich stron.
    """
    g = Grid.from_size(100, 200, 20, 40)
    src, rec = edge_points(g, "L", 15), edge_points(g, "P", 25)
    for n in (7, 12, 20, 24, 36, 50, 60):
        p = compute(g, src, rec, build_matrix=False, n_sektorow=n).pokrycie_katowe
        t = p[np.isfinite(p)]
        assert t.size, f"n_sektorow={n}: same NaN-y"
        assert t.min() > 0.0 and t.max() <= 1.0 + TOL, f"n_sektorow={n}: {t.max()}"


# -------------------------------------------------------- średnia cięciwa
def test_srednia_ciecziwa_matches_cauchy():
    """
    Wzór Cauchy'ego: π · pole / obwód. Dla oczka 5 × 5 m daje 3,927 m —
    ta liczba stoi w opisie paska skali, więc niech ją coś pilnuje.
    """
    g = Grid.from_size(100, 100, 20, 20)                 # oczko 5 × 5
    assert g.dx == 5 and g.dy == 5
    assert abs(g.srednia_ciecziwa - np.pi * 25 / 20) < TOL
    assert abs(g.srednia_ciecziwa - 3.9269908) < 1e-6

    # oczko dwa razy większe → cięciwa dwa razy dłuższa (skala liniowa)
    g2 = Grid.from_size(100, 100, 10, 10)                # oczko 10 × 10
    assert abs(g2.srednia_ciecziwa - 2 * g.srednia_ciecziwa) < TOL

    # oczko wydłużone: między krótszym a dłuższym bokiem
    gw = Grid.from_size(100, 800, 20, 20)                # oczko 5 × 40
    assert gw.dx < gw.srednia_ciecziwa < gw.dy


def test_srednia_ciecziwa_agrees_with_measured_paths():
    """
    Wzór zakłada promienie ze wszystkich stron równomiernie. Sprawdzamy, ile
    naprawdę wychodzi na geometrii PS ↔ czujniki: średnia droga w kratce =
    suma dróg / liczba przejść. Rozjazd rzędu kilku procent jest wpisany
    w metodę — gdyby urósł do dziesiątek, ekwiwalent przestałby coś znaczyć.
    """
    g = Grid.from_size(100, 200, 20, 40)                 # oczko 5 × 5
    cov = compute(g, edge_points(g, "L", 15), edge_points(g, "P", 25),
                  build_matrix=False)
    m = cov.hits > 0
    zmierzona = cov.length[m].sum() / cov.hits[m].sum()
    assert abs(zmierzona / g.srednia_ciecziwa - 1) < 0.10


# ---------------------------------------------------------------- zapis
def test_csv_has_no_nan():
    """
    Kratki bez promieni mają NaN w pokryciu kątowym i anizotropii. W CSV muszą
    wyjść jako 0 — Excel nie liczy słowa „nan", a Surfer wczytuje je losowo.
    Kratkę bez informacji poznaje się po kolumnie liczba_promieni = 0.
    """
    import os
    import tempfile

    g = Grid.from_size(100, 200, 10, 20)
    # jeden PS i jeden czujnik: prawie cała siatka zostaje bez promieni
    cov = compute(g, edge_points(g, "L", 2)[:1], edge_points(g, "P", 2)[:1],
                  build_matrix=False)
    assert np.isnan(cov.anisotropy).any(), "test bez NaN-ów niczego nie sprawdza"

    sciezka = os.path.join(tempfile.mkdtemp(), "pokrycie.csv")
    tekst = open(cov.to_csv(sciezka), encoding="utf-8").read().lower()
    assert "nan" not in tekst and "inf" not in tekst

    naglowek = open(sciezka, encoding="utf-8").readline().strip().split(",")
    dane = np.loadtxt(sciezka, delimiter=",", skiprows=1)
    puste = dane[dane[:, naglowek.index("liczba_promieni")] == 0]
    for kol in ("ekwiwalent_promieni", "pokrycie_katowe", "anizotropia",
                "luka_katowa_st", "suma_drog_m"):
        assert puste.size and np.all(puste[:, naglowek.index(kol)] == 0), kol


# ----------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  BŁĄD {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} testów przeszło")
    raise SystemExit(1 if failed else 0)