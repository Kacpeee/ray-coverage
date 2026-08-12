"""
raycov.py — rdzeń obliczeniowy pokrycia promieniami.

Czysta logika: bez rysowania, bez interfejsu, bez wczytywania plików.
Dzięki temu da się to przetestować (patrz test_raycov.py) i użyć
zarówno z GUI, jak i z własnego skryptu wsadowego.

Model: ośrodek jednorodny → promień biegnie po najkrótszej drodze,
czyli po odcinku prostym między źródłem a odbiornikiem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = ["Grid", "Coverage", "traverse", "clip_segment", "compute",
           "snap", "edge_points"]

# Tolerancja *względna*: mnożona przez rząd wielkości współrzędnych.
# Przy współrzędnych rzędu 10^6 (PL-2000, UTM) bezwzględne 1e-12 leży poniżej
# rozdzielczości typu double i przestaje cokolwiek zabezpieczać.
EPS = 1e-12

# Rozdzielczość histogramu azymutów. Drobne sektory (180/60 = 3°) są potrzebne
# do największej luki kątowej — przy 15° najwęższe dziury ginęły w zaokrągleniu.
# Pokrycie kątowe scala je z powrotem do SEKTORY_GRUBE, bo gdyby liczyło ułamek
# z sześćdziesięciu, kratka z dwunastoma promieniami nigdy nie przekroczyłaby
# 0,2 — i metryka mierzyłaby liczbę promieni zamiast rozrzutu kierunków.
SEKTORY_DROBNE = 60
SEKTORY_GRUBE = 12

# Ile przecięć promień-z-komórką compute() bierze na raz. Liczy wszystkie
# promienie hurtem, macierzami, więc pamięć rośnie z liczbą promieni razy
# długość najdłuższego z nich — porcjowanie trzyma to w ryzach niezależnie
# od tego, ile ktoś postawi punktów. 2 mln to około 100 MB szczytowo.
BUDZET = 2_000_000


# =====================================================================
# SIATKA
# =====================================================================
@dataclass(frozen=True)
class Grid:
    """
    Regularna siatka prostokątna nałożona na obszar badany.

    Obszar jest zadany dwoma narożnikami, więc początek układu nie musi
    leżeć w (0, 0) — można pracować wprost we współrzędnych terenowych
    (np. PL-2000) albo przesunąć zero tam, gdzie stoi reper.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("xmax musi być > xmin, a ymax > ymin")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx i ny muszą być dodatnie")

    @classmethod
    def from_size(cls, width: float, height: float, nx: int, ny: int,
                  x0: float = 0.0, y0: float = 0.0) -> "Grid":
        """Siatka o zadanym rozmiarze, z lewym dolnym narożnikiem w (x0, y0)."""
        return cls(float(x0), float(y0),
                   float(x0) + float(width), float(y0) + float(height),
                   int(nx), int(ny))

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def origin(self) -> tuple[float, float]:
        """Lewy dolny narożnik — punkt (0, 0) siatki."""
        return self.xmin, self.ymin

    @property
    def dx(self) -> float:
        return self.width / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        """Kształt macierzy wyników: (wiersze, kolumny) = (ny, nx)."""
        return self.ny, self.nx

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def extent(self) -> list[float]:
        """Do matplotlib.imshow(extent=...)."""
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Współrzędne środków komórek jako dwie macierze (ny, nx)."""
        xc = self.xmin + (np.arange(self.nx) + 0.5) * self.dx
        yc = self.ymin + (np.arange(self.ny) + 0.5) * self.dy
        return np.meshgrid(xc, yc)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Wciśnij punkt do wnętrza obszaru."""
        return (min(max(x, self.xmin), self.xmax),
                min(max(y, self.ymin), self.ymax))

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        """
        Indeksy (ix, iy) komórki zawierającej punkt; punkt spoza obszaru
        trafia do komórki brzegowej.

        Uwaga: floor, nie int() — int() ucina w stronę zera, więc dla
        współrzędnych ujemnych (a te są możliwe, odkąd początek układu jest
        dowolny) myliłby się o jedną komórkę.
        """
        ix = math.floor((x - self.xmin) / self.dx)
        iy = math.floor((y - self.ymin) / self.dy)
        return min(max(ix, 0), self.nx - 1), min(max(iy, 0), self.ny - 1)

    def cell_center(self, ix: int, iy: int) -> tuple[float, float]:
        """Środek komórki (ix, iy) we współrzędnych obszaru."""
        return self.xmin + (ix + 0.5) * self.dx, self.ymin + (iy + 0.5) * self.dy

    @property
    def srednia_ciecziwa(self) -> float:
        """
        Średnia długość przecięcia kratki przez promień — wzór Cauchy'ego
        z geometrii całkowej: π · pole / obwód.

        Służy do przeliczenia sumy dróg [m] na ekwiwalentną liczbę promieni,
        czyli „ile przeciętnych przejść jest warte to, co przez kratkę
        przeszło". Kratka musnięta po rogu przez 16 promieni ma sumę dróg
        rzędu 16 m, czyli ekwiwalent 2 — a nie 16, jak pokazują zliczenia.
        """
        return math.pi * self.dx * self.dy / (2 * (self.dx + self.dy))

    @property
    def scale(self) -> float:
        """Rząd wielkości współrzędnych — baza dla tolerancji numerycznych."""
        return max(abs(self.xmin), abs(self.xmax),
                   abs(self.ymin), abs(self.ymax), self.width, self.height, 1.0)


# =====================================================================
# PRZYCIĄGANIE DO PEŁNYCH WARTOŚCI (snap)
# =====================================================================
def snap(x: float, y: float, step: float,
         origin: Sequence[float] = (0.0, 0.0)) -> tuple[float, float]:
    """
    Przyciąga punkt do najbliższego węzła pomocniczej siatki o oczku `step`.

    Domyślnie węzły leżą na pełnych wielokrotnościach `step` w układzie
    globalnym: step=1 daje pełne metry, step=5 — co 5 m. `origin` pozwala
    zakotwiczyć te węzły gdzie indziej, np. w narożniku obszaru badanego.

    step <= 0 (albo NaN) oznacza „nie przyciągaj” i zwraca punkt bez zmian —
    dzięki temu wywołanie wygląda tak samo przy snapie włączonym i wyłączonym.

    Zaokrąglenie jest „w górę przy połówce” (0.5 → 1), a nie bankierskie jak
    round(), żeby wynik był przewidywalny dla operatora.
    """
    if not step > 0 or not math.isfinite(step):
        return float(x), float(y)
    ox, oy = float(origin[0]), float(origin[1])
    return (ox + math.floor((x - ox) / step + 0.5) * step,
            oy + math.floor((y - oy) / step + 0.5) * step)


# =====================================================================
# PRZYCIĘCIE ODCINKA DO OBSZARU (Liang–Barsky)
# =====================================================================
def clip_segment(
    p0: Sequence[float], p1: Sequence[float], g: Grid
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Zwraca część odcinka leżącą wewnątrz siatki albo None."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0

    for p, q in ((-dx, x0 - g.xmin), (dx, g.xmax - x0),
                 (-dy, y0 - g.ymin), (dy, g.ymax - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
        else:
            r = q / p
            if p < 0.0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)

    if t0 >= t1:
        return None
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


# =====================================================================
# TRAVERSAL SIATKI — Amanatides & Woo (1987)
# =====================================================================
def traverse(
    p0: Sequence[float], p1: Sequence[float], g: Grid
) -> Iterator[tuple[int, int, float]]:
    """
    Generuje (ix, iy, długość_odcinka_w_komórce) dla komórek przeciętych
    przez odcinek p0→p1.

    Idea: zamiast sprawdzać wszystkie nx*ny komórek, idziemy wzdłuż promienia
    i za każdym krokiem przeskakujemy do najbliższej ściany komórki. Koszt to
    O(liczby przeciętych komórek) ≈ O(nx + ny), a nie O(nx * ny).

    Każda komórka pojawia się dokładnie raz, a suma długości równa się
    długości odcinka wewnątrz siatki.
    """
    seg = clip_segment(p0, p1, g)
    if seg is None:
        return
    (x0, y0), (x1, y1) = seg

    rx, ry = x1 - x0, y1 - y0
    length = math.hypot(rx, ry)
    tol = EPS * g.scale                # patrz komentarz przy EPS
    if length < tol:
        return
    ux, uy = rx / length, ry / length

    ix, iy = g.index_of(x0, y0)
    step_x = 1 if ux > 0 else -1
    step_y = 1 if uy > 0 else -1

    if abs(ux) > EPS:
        t_max_x = (g.xmin + (ix + (1 if ux > 0 else 0)) * g.dx - x0) / ux
        t_delta_x = g.dx / abs(ux)
    else:
        t_max_x = t_delta_x = np.inf

    if abs(uy) > EPS:
        t_max_y = (g.ymin + (iy + (1 if uy > 0 else 0)) * g.dy - y0) / uy
        t_delta_y = g.dy / abs(uy)
    else:
        t_max_y = t_delta_y = np.inf

    t = 0.0
    while t < length - tol:
        t_next = min(t_max_x, t_max_y, length)
        if t_next - t > tol:
            yield ix, iy, t_next - t
        t = t_next
        if t >= length - tol:
            break
        if t_max_x < t_max_y:
            ix += step_x
            t_max_x += t_delta_x
        else:
            iy += step_y
            t_max_y += t_delta_y
        if not (0 <= ix < g.nx and 0 <= iy < g.ny):
            break


# =====================================================================
# TE SAME DWA KROKI, ALE DLA WSZYSTKICH PROMIENI NARAZ
# =====================================================================
# clip_segment() i traverse() opisują algorytm po jednym promieniu i tak je
# najłatwiej czytać — dlatego zostają. Tyle że przy kilkunastu tysiącach
# promieni pętla Pythona po nich zajmuje sekundy, a GUI przelicza wszystko po
# każdym postawionym punkcie. Poniższe funkcje robią dokładnie to samo, tylko
# na macierzach: jeden przebieg numpy zamiast miliona kroków interpretera.
def _przytnij_wiele(p0: np.ndarray, p1: np.ndarray, g: Grid):
    """
    Liang–Barsky dla całej paczki odcinków. Zwraca (t0, t1, ok), gdzie t to
    parametry przycięcia wzdłuż p0→p1, a ok mówi, czy z odcinka cokolwiek
    zostało. Kolejność czterech warunków jest ta sama co w clip_segment().
    """
    d = p1 - p0
    t0 = np.zeros(len(p0))
    t1 = np.ones(len(p0))
    ok = np.ones(len(p0), dtype=bool)
    for p, q in ((-d[:, 0], p0[:, 0] - g.xmin), (d[:, 0], g.xmax - p0[:, 0]),
                 (-d[:, 1], p0[:, 1] - g.ymin), (d[:, 1], g.ymax - p0[:, 1])):
        rownolegly = p == 0.0
        ok &= ~(rownolegly & (q < 0.0))          # równoległy i na zewnątrz
        # dzielnik podmieniony na 1 tylko po to, żeby numpy nie ostrzegał —
        # wyniki z tych pozycji i tak odsiewa maska `rownolegly`
        r = np.where(rownolegly, 0.0, q / np.where(rownolegly, 1.0, p))
        wchodzi = ~rownolegly & (p < 0.0)
        ok &= ~(wchodzi & (r > t1))
        t0 = np.where(wchodzi & ok, np.maximum(t0, r), t0)
        wychodzi = ~rownolegly & (p > 0.0)
        ok &= ~(wychodzi & (r < t0))
        t1 = np.where(wychodzi & ok, np.minimum(t1, r), t1)
    return t0, t1, ok & (t0 < t1)


def _ciagi(ile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rozwinięcie liczb w ciągi pozycji: [3, 0, 2] → wiersze [0,0,0,2,2]
    i pozycje [0,1,2,0,1]. Tak buduje się „poszarpaną" tablicę bez pętli.
    """
    wiersze = np.repeat(np.arange(ile.size), ile)
    poczatki = np.concatenate([[0], np.cumsum(ile)[:-1]])
    return wiersze, np.arange(int(ile.sum())) - np.repeat(poczatki, ile)


# =====================================================================
# WYNIK
# =====================================================================
@dataclass
class Coverage:
    """Wynik zliczania. Wszystkie macierze mają kształt (ny, nx)."""

    grid: Grid
    hits: np.ndarray          # ile promieni przeszło przez komórkę
    length: np.ndarray        # suma dróg promieni w komórce [m]
    _mxx: np.ndarray
    _mxy: np.ndarray
    _myy: np.ndarray
    n_rays: int
    g_rows: np.ndarray        # macierz tomografii G w formacie COO:
    g_cols: np.ndarray        #   G[ray, iy*nx + ix] = długość w komórce
    g_vals: np.ndarray
    sektory: np.ndarray       # (n_sektorow, ny, nx) bool — czy azymut trafiony
    has_matrix: bool = True   # False, gdy compute(build_matrix=False)

    # ---------- wielkości pochodne ----------
    #
    # cached_property, nie property: pasek pod kursorem czyta anizotropię,
    # pokrycie i lukę przy każdym drgnięciu myszy, a luka kątowa to sto
    # dwadzieścia przebiegów po całej siatce. Liczone za każdym razem od nowa
    # dawały przy gęstej siatce wyraźne opóźnienie kursora. Wynik zależy tylko
    # od pól policzonych w compute(), więc liczy się raz na Coverage.
    @cached_property
    def density(self) -> np.ndarray:
        """Gęstość dróg promieni [1/m] — suma dróg na jednostkę pola."""
        return self.length / self.grid.cell_area

    @cached_property
    def weight_hits(self) -> np.ndarray:
        """Waga 0–1 z liczby promieni."""
        m = self.hits.max()
        return self.hits / m if m else np.zeros_like(self.hits, dtype=float)

    @cached_property
    def weight_length(self) -> np.ndarray:
        """Waga 0–1 z sumy dróg — uczciwsza przy brzegach obszaru."""
        m = self.length.max()
        return self.length / m if m else np.zeros_like(self.length)

    @cached_property
    def ekwiwalent(self) -> np.ndarray:
        """Suma dróg przeliczona na liczbę przeciętnych przejść przez kratkę."""
        return self.length / self.grid.srednia_ciecziwa

    @cached_property
    def pokrycie_katowe(self) -> np.ndarray:
        """
        Ułamek kierunków, z których kratka dostaje promienie (0–1).

        Pełny zakres azymutów (180°, bo promień nieskierowany) dzielimy na
        sektory i liczymy, ile z nich jest trafionych. 1 oznacza promienie ze
        wszystkich stron, 0,25 — tylko z ćwiartki kierunków.

        To miara bliźniacza do anizotropii, ale prostsza w odczycie: mówi wprost
        „ile stron świata widzi ta kratka", podczas gdy anizotropia waży
        kierunki długością drogi. Kratki bez pokrycia dostają NaN, bo brak
        promieni to brak informacji, a nie kierunkowość równa zeru.
        """
        # Dzielimy przez faktyczną liczbę sektorów w tablicy, nie przez
        # SEKTORY_GRUBE: gdy compute() dostanie n_sektorow niebędące
        # wielokrotnością dwunastu, scalanie się nie odbywa i sztywna dwunastka
        # dawała pokrycie powyżej 1 (przy n_sektorow=20 wychodziło 1,667).
        s = self._grube_sektory()
        return np.where(self.hits > 0, s.sum(axis=0) / s.shape[0], np.nan)

    def _grube_sektory(self) -> np.ndarray:
        """Drobne sektory scalone do SEKTORY_GRUBE — patrz komentarz przy stałej."""
        n = self.sektory.shape[0]
        if n == SEKTORY_GRUBE or n % SEKTORY_GRUBE:
            return self.sektory
        return self.sektory.reshape(SEKTORY_GRUBE, n // SEKTORY_GRUBE,
                                    *self.sektory.shape[1:]).any(axis=1)

    @cached_property
    def luka_katowa(self) -> np.ndarray:
        """
        Największy klin azymutów [°], z którego do kratki nie przyszedł ani
        jeden promień. 0 = promienie ze wszystkich stron, 180 = jeden kierunek.

        Istnieje, bo anizotropia tego nie pokazuje. Anizotropia liczy się
        z tensora drugiego rzędu, a ten mierzy tylko, czy chmura kierunków jest
        okrągła — dziur w niej nie widzi. Trzy kierunki rozstawione co 60° dają
        anizotropię 0,000, dokładnie tyle samo co pełne 180° pokryte gęsto,
        choć 177 ze 180 stopni jest wtedy pustych. Przy źródłach po jednej
        stronie obszaru i czujnikach po drugiej to nie jest przypadek graniczny,
        tylko codzienność: przez środek nie przechodzi żaden promień bliski
        pionowi, anizotropia wychodzi 0,1 („idealnie"), a naprawdę brakuje
        tam klina szerokiego na kilkadziesiąt stopni. Ta metryka go pokazuje.

        Zakres azymutów to 180°, nie 360°, bo promień jest nieskierowany.
        Luka liczona jest w sektorach, więc wychodzi zaniżona o mniej niż dwie
        ich szerokości: sektor musnięty jednym promieniem liczy się jak pełny.
        Kratki bez promieni dostają NaN — brak informacji, nie luka 180°.
        """
        n = self.sektory.shape[0]
        puste = ~self.sektory
        biezaca = np.zeros(self.grid.shape, dtype=np.int32)
        najdluzsza = np.zeros(self.grid.shape, dtype=np.int32)
        # Dwa obiegi, bo azymut zawija się modulo 180°: seria pustych sektorów
        # może przechodzić przez koniec tablicy i wracać na jej początek.
        for i in range(2 * n):
            biezaca = np.where(puste[i % n], biezaca + 1, 0)
            najdluzsza = np.maximum(najdluzsza, biezaca)
        stopnie = np.minimum(najdluzsza, n) * (180.0 / n)
        return np.where(self.hits > 0, stopnie, np.nan)

    @cached_property
    def anisotropy(self) -> np.ndarray:
        """
        Anizotropia pokrycia kątowego z tensora gęstości promieni.
        0 = promienie ze wszystkich azymutów, 1 = wszystkie równoległe.
        Komórka trafiona wieloma równoległymi promieniami jest gorzej
        rozwiązana niż trafiona kilkoma z różnych stron.
        """
        tr = self._mxx + self._myy
        root = np.sqrt(np.maximum((self._mxx - self._myy) ** 2
                                  + 4.0 * self._mxy ** 2, 0.0))
        lam1 = 0.5 * (tr + root)
        lam2 = 0.5 * (tr - root)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(lam1 > 0, 1.0 - lam2 / lam1, np.nan)

    # ---------- podsumowanie ----------
    def stats(self) -> dict:
        h = self.hits
        nz = h[h > 0]
        return {
            "promieni": self.n_rays,
            "komórek": self.grid.n_cells,
            "bez_pokrycia": int((h == 0).sum()),
            "min": int(nz.min()) if nz.size else 0,
            "mediana": float(np.median(nz)) if nz.size else 0.0,
            "max": int(h.max()),
            "nierównomierność": float(h.max() / nz.min()) if nz.size else float("inf"),
        }

    def summary(self) -> str:
        s = self.stats()
        g = self.grid
        return (
            f"obszar {g.xmax - g.xmin:g} × {g.ymax - g.ymin:g} m, "
            f"siatka {g.nx} × {g.ny} (komórka {g.dx:g} × {g.dy:g} m)\n"
            f"promieni: {s['promieni']}, komórek: {s['komórek']}\n"
            f"bez pokrycia: {s['bez_pokrycia']} "
            f"({100 * s['bez_pokrycia'] / s['komórek']:.0f}%)\n"
            f"promieni w komórce  min/mediana/max: "
            f"{s['min']} / {s['mediana']:.0f} / {s['max']}\n"
            f"nierównomierność (max/min): {s['nierównomierność']:.1f}×"
        )

    # ---------- zapis ----------
    def to_csv(self, path: str) -> str:
        """Tabela po jednym wierszu na kratkę. Do QGIS/Surfera/Excela."""
        xc, yc = self.grid.centers()
        tab = np.column_stack([
            xc.ravel(), yc.ravel(),
            self.hits.ravel(),
            self.ekwiwalent.ravel(),
            self.pokrycie_katowe.ravel(),
            self.anisotropy.ravel(),
            self.luka_katowa.ravel(),
            self.length.ravel(),
            np.repeat(np.arange(self.grid.nx)[None, :], self.grid.ny, 0).ravel(),
            np.repeat(np.arange(self.grid.ny)[:, None], self.grid.nx, 1).ravel(),
        ])
        # Kratki bez promieni mają w pokryciu kątowym i anizotropii NaN (brak
        # informacji). W CSV wychodziło z tego słowo „nan", którego Excel nie
        # liczy, a Surfer potrafi wczytać jako 0 albo wysypać się na wierszu.
        # Zapisujemy więc 0 — kratkę bez informacji poznaje się po kolumnie
        # liczba_promieni = 0, i tylko tak wolno te zera czytać: anizotropia 0
        # sama w sobie znaczy „promienie ze wszystkich stron", czyli coś dobrego.
        tab = np.where(np.isfinite(tab), tab, 0.0)
        np.savetxt(
            path, tab, delimiter=",",
            # Format kolumnami, nie jeden wspólny: „%g" przy współrzędnych
            # terenowych daje 5.58118e+06 i gubi metry, a przez to wszystkie
            # kratki w kolumnie dostają tę samą wartość x.
            fmt=["%.3f", "%.3f", "%d", "%.4f", "%.4f", "%.4f", "%.1f",
                 "%.4f", "%d", "%d"],
            header="x_srodka,y_srodka,liczba_promieni,ekwiwalent_promieni,"
                   "pokrycie_katowe,anizotropia,luka_katowa_st,suma_drog_m,ix,iy",
            comments="",
        )
        return path

    def to_sparse(self):
        """Macierz G jako scipy.sparse (jeśli scipy jest zainstalowane)."""
        if not self.has_matrix:
            raise RuntimeError(
                "Ten wynik policzono z build_matrix=False — macierzy G nie ma. "
                "Powtórz compute(...) bez tego przełącznika."
            )
        from scipy.sparse import coo_matrix
        return coo_matrix(
            (self.g_vals, (self.g_rows, self.g_cols)),
            shape=(self.n_rays, self.grid.n_cells),
        ).tocsr()


# =====================================================================
# ZLICZANIE
# =====================================================================
def compute(
    grid: Grid,
    sources: Iterable[Sequence[float]],
    receivers: Iterable[Sequence[float]],
    pairs: Iterable[tuple[int, int]] | None = None,
    build_matrix: bool = True,
    n_sektorow: int = SEKTORY_DROBNE,
) -> Coverage:
    """
    Zlicza pokrycie dla wszystkich par (źródło, odbiornik).

    pairs pozwala ograniczyć zestaw par, np. gdy w terenie nie każdy
    czujnik rejestrował każdy strzał albo gdy chcesz odrzucić pary
    o zbyt małym odstępie.

    build_matrix=False pomija składanie macierzy G. Same mapy pokrycia jej
    nie potrzebują, a to ona zjada większość czasu i pamięci — warto wyłączyć
    tam, gdzie compute() leci w pętli (suwaki w GUI, testy geometrii).

    n_sektorow dzieli zakres azymutów na tyle przedziałów. Promień jest
    nieskierowany, więc zakres to 180°, a nie 360° — przy 60 sektorach jeden
    ma 3°. Z tego histogramu liczą się luka kątowa i pokrycie kątowe; to
    drugie scala sektory z powrotem do dwunastu, więc dla porównywalności
    wyników trzymaj n_sektorow jako wielokrotność SEKTORY_GRUBE.

    Liczone jest to hurtem, macierzowo — nie promień po promieniu. Wynik jest
    ten sam co z pętli po traverse() (pilnuje tego test), ale kilkanaście razy
    szybciej, bo cała robota dzieje się w numpy zamiast w interpreterze.
    Zamiast kroczyć wzdłuż promienia, dla każdego z nich liczymy parametry
    wszystkich przecięć z liniami siatki, sortujemy je i bierzemy środek każdego
    odcinka między kolejnymi przecięciami — środek leży wewnątrz komórki, więc
    wskazuje ją bez żadnego balansowania na granicy.
    """
    g = grid
    src = np.atleast_2d(np.asarray(list(sources), dtype=float).reshape(-1, 2))
    rec = np.atleast_2d(np.asarray(list(receivers), dtype=float).reshape(-1, 2))
    if pairs is None:
        i_src = np.repeat(np.arange(len(src)), len(rec))
        i_rec = np.tile(np.arange(len(rec)), len(src))
    else:
        par = np.asarray(list(pairs), dtype=np.int64).reshape(-1, 2)
        i_src, i_rec = par[:, 0], par[:, 1]
    n_rays = int(i_src.size)

    n_sektorow = max(int(n_sektorow), 1)
    n_kom = g.n_cells
    # Akumulatory trzymamy spłaszczone: bincount sumuje po jednym indeksie,
    # a kształt (ny, nx) przywracamy dopiero na końcu.
    hits = np.zeros(n_kom, dtype=np.int64)
    length = np.zeros(n_kom)
    mxx = np.zeros(n_kom)
    mxy = np.zeros(n_kom)
    myy = np.zeros(n_kom)
    # bool, nie licznik: obie metryki kątowe pytają tylko „czy trafiony", a przy
    # 60 sektorach na kratkę licznik int32 zjadałby pięć razy tyle pamięci
    sektory = np.zeros(n_sektorow * n_kom, dtype=bool)
    macierz: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    tol = EPS * g.scale

    if n_rays:
        p0, p1 = src[i_src], rec[i_rec]
        d = p1 - p0
        dlugosc = np.hypot(d[:, 0], d[:, 1])
        t0, t1, ok = _przytnij_wiele(p0, p1, g)
        ok &= dlugosc > tol               # źródło i czujnik w tym samym punkcie

        # Kierunek bierzemy z pełnego odcinka, nie z przyciętego — ten sam,
        # a przy promieniach ledwo muskających obszar mniej wrażliwy numerycznie.
        mian = np.where(ok, dlugosc, 1.0)
        ux, uy = d[:, 0] / mian, d[:, 1] / mian
        # Sektor azymutu jest wspólny dla całego promienia, więc liczymy go raz.
        # Kąt modulo π, bo promień biegnący w lewo niesie tę samą informację
        # co biegnący w prawo. Drobne dociągnięcie przed obcięciem: kąt dokładnie
        # na granicy sektora wychodzi czasem 6.999999999999999 zamiast 7 i wpada
        # do sąsiada — a regularny wachlarz kierunków trafia w granice zawsze.
        sekt = np.minimum(
            ((np.arctan2(uy, ux) % math.pi) / math.pi * n_sektorow + 1e-9
             ).astype(np.int64), n_sektorow - 1)

        # końce odcinka po przycięciu, w jednostkach kratki (0..nx, 0..ny)
        a = p0 + t0[:, None] * d
        b = p0 + t1[:, None] * d
        dl_w_siatce = (t1 - t0) * dlugosc
        zywe = np.flatnonzero(ok & (dl_w_siatce > tol))
        au = (a[zywe, 0] - g.xmin) / g.dx
        bu = (b[zywe, 0] - g.xmin) / g.dx
        av = (a[zywe, 1] - g.ymin) / g.dy
        bv = (b[zywe, 1] - g.ymin) / g.dy
        # ile linii siatki przecina promień: tyle jest w przedziale liczb
        # całkowitych leżących ściśle między początkiem a końcem
        ile_x = np.maximum(np.ceil(np.maximum(au, bu))
                           - np.floor(np.minimum(au, bu)) - 1, 0).astype(np.int64)
        ile_y = np.maximum(np.ceil(np.maximum(av, bv))
                           - np.floor(np.minimum(av, bv)) - 1, 0).astype(np.int64)
        szer = ile_x + ile_y + 2          # + oba końce promienia

        krok = max(int(BUDZET // max(int(szer.max(initial=1)), 1)), 1)
        for lo in range(0, zywe.size, krok):
            sl = slice(lo, min(lo + krok, zywe.size))
            m = sl.stop - sl.start
            W = int(szer[sl].max())
            # Wiersz = jeden promień, w kolumnach parametry przecięć (0..1).
            # Wypełnienie jedynką, nie nieskończonością: nadmiarowe kolumny dają
            # wtedy różnicę zero i odpadają razem z odcinkami zerowej długości,
            # bez produkowania NaN-ów po drodze.
            T = np.ones((m, W))
            T[:, 0] = 0.0
            kursor = np.full(m, 1, dtype=np.int64)
            for ile, od, do in ((ile_x[sl], au[sl], bu[sl]),
                                (ile_y[sl], av[sl], bv[sl])):
                if ile.sum():
                    wiersze, poz = _ciagi(ile)
                    linia = np.repeat(np.floor(np.minimum(od, do)), ile) + poz + 1
                    od_r, do_r = np.repeat(od, ile), np.repeat(do, ile)
                    T[wiersze, kursor[wiersze] + poz] = (linia - od_r) / (do_r - od_r)
                kursor += ile
            T.sort(axis=1, kind="stable")     # dwa gotowe ciągi → scalenie, nie sort

            dl = dl_w_siatce[zywe[sl]]
            lewo, prawo = T[:, :-1], T[:, 1:]
            odcinki = prawo - lewo
            # kratka musnięta na długości poniżej tolerancji to zaokrąglenie,
            # nie przejście — i tak samo odsiewa to traverse()
            dobre = odcinki > (tol / dl)[:, None]
            if not dobre.any():
                continue
            # Dalej już tylko po realnych odcinkach: macierz z wypełnieniem
            # zostaje z tyłu, a to ona zajmowałaby pamięć.
            wiersz = np.broadcast_to(np.arange(m)[:, None], odcinki.shape)[dobre]
            seg = odcinki[dobre] * dl[wiersz]
            srodek = 0.5 * (lewo[dobre] + prawo[dobre])
            u = au[sl][wiersz] + srodek * (bu[sl] - au[sl])[wiersz]
            v = av[sl][wiersz] + srodek * (bv[sl] - av[sl])[wiersz]
            # środek odcinka leży wewnątrz kratki, więc obcięcie do liczby
            # całkowitej daje jej indeks; clip tylko na wypadek zaokrągleń
            plaskie = (np.clip(v.astype(np.int64), 0, g.ny - 1) * g.nx
                       + np.clip(u.astype(np.int64), 0, g.nx - 1))

            promien = zywe[sl][wiersz]
            uxs, uys = ux[promien], uy[promien]
            hits += np.bincount(plaskie, minlength=n_kom)
            length += np.bincount(plaskie, weights=seg, minlength=n_kom)
            mxx += np.bincount(plaskie, weights=seg * uxs * uxs, minlength=n_kom)
            mxy += np.bincount(plaskie, weights=seg * uxs * uys, minlength=n_kom)
            myy += np.bincount(plaskie, weights=seg * uys * uys, minlength=n_kom)
            sektory[sekt[promien] * n_kom + plaskie] = True
            if build_matrix:
                macierz.append((promien, plaskie, seg))

    if macierz:
        rows, cols, vals = (np.concatenate([c[i] for c in macierz])
                            for i in range(3))
    else:
        rows = np.zeros(0, dtype=np.int64)
        cols = np.zeros(0, dtype=np.int64)
        vals = np.zeros(0, dtype=float)

    ksztalt = g.shape
    return Coverage(
        grid=g, hits=hits.reshape(ksztalt), length=length.reshape(ksztalt),
        _mxx=mxx.reshape(ksztalt), _mxy=mxy.reshape(ksztalt),
        _myy=myy.reshape(ksztalt), n_rays=n_rays,
        g_rows=rows, g_cols=cols, g_vals=vals,
        sektory=sektory.reshape(n_sektorow, *ksztalt),
        has_matrix=build_matrix,
    )


# =====================================================================
def edge_points(grid: Grid, edge: str, n: int) -> np.ndarray:
    """n punktów rozstawionych równomiernie wzdłuż krawędzi: L, P, G lub D."""
    if n < 1:
        raise ValueError("n musi być dodatnie")
    t = (np.arange(n) + 0.5) / n
    if edge == "L":
        return np.column_stack([np.full(n, grid.xmin), grid.ymin + t * grid.height])
    if edge == "P":
        return np.column_stack([np.full(n, grid.xmax), grid.ymin + t * grid.height])
    if edge == "G":
        return np.column_stack([grid.xmin + t * grid.width, np.full(n, grid.ymax)])
    if edge == "D":
        return np.column_stack([grid.xmin + t * grid.width, np.full(n, grid.ymin)])
    raise ValueError("edge musi być jednym z: L, P, G, D")


if __name__ == "__main__":
    # obszar 100 × 200 m z zerem przesuniętym w teren, punkty co pełny metr
    g = Grid.from_size(100, 200, 10, 20, x0=250.0, y0=-100.0)
    src = [snap(x, y, 1.0) for x, y in edge_points(g, "L", 15)]
    rec = [snap(x, y, 1.0) for x, y in edge_points(g, "P", 25)]
    cov = compute(g, src, rec)
    print(f"początek układu: {g.origin}")
    print(cov.summary())