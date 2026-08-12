"""
app.py — interaktywne projektowanie geometrii pomiaru.

Uruchomienie:   python app.py

Wymaga tylko numpy i matplotlib. Cała logika siedzi w raycov.py — tutaj jest
wyłącznie interfejs, więc jedno da się zmieniać bez psucia drugiego.

Obsługa:
  lewy przycisk myszy   postaw PS / czujnik (zależnie od trybu)
  prawy przycisk myszy  usuń pojedynczy punkt spod kursora — czerwona obwódka
                        pokazuje, który to; przy nałożonych punktach decyduje
                        aktualny tryb (1 / 2)
  klawisze 1 / 2        tryb: PS / czujnik
  klawisz  c            zmień skalę barw
  klawisz  r            pokaż lub ukryj promienie
  klawisz  n            wpisz lub schowaj liczby w kratkach

Widok mapy:
  kółko myszy           przybliż / oddal wokół kursora
  środkowy przycisk     chwyć i przesuń
  klawisz  0            wróć do pełnego obszaru
Zoom przeżywa przeliczenie — stawianie punktów go nie kasuje. Pasek narzędzi
matplotliba (lupa, rączka, dom) działa normalnie obok tego.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

from raycov import Grid, compute, edge_points
from wczytaj import Grupa, wczytaj_punkty

# --------------------------------------------------------------------- wygląd
INK, PANEL, LINE, PAPER, MUTED = "#f2f5f8", "#e4eaf1", "#b8c4d2", "#16212e", "#5a6b7d"
SHOT, GEO, WARN = "#e07b00", "#0f7d94", "#c02f2f"
EMPTY = "#d6dbe1"                  # komórka bez pokrycia
CMAPS = ["klasy", "viridis", "inferno", "Greys", "magma"]

# --------------------------------------------------- klasy jakości pokrycia
# Skala BEZWZGLĘDNA — w sztukach promieni na kratkę, niezależnie od oczka.
# Te same liczby obowiązują przy oczku 5 × 5 m i przy 25 × 25 m: klasa mówi,
# ile promieni faktycznie przeszło przez kratkę, a nie ile ich „wypada" na jej
# rozmiar. Kratka jest większa → naturalnie wpada w wyższą klasę i tak ma być.
KLASY_PROGI = [3, 8, 11, 16]       # dolne granice klas 2..5
KLASY_BARWY = ["#d13b3b",          # 0 – 2    czerwony
               "#ec7a1c",          # 3 – 7    pomarańczowy
               "#e8b31f",          # 8 – 10   żółty
               "#2f7fd1",          # 11 – 15  niebieski
               "#2e9e57"]          # >= 16    zielony

plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": INK,
    "text.color": PAPER, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "font.size": 9,
    "font.family": "monospace",
})


def klasy_opisy() -> list[str]:
    """Etykiety klas do paska skali i legend: '0–2', '3–7', …, '≥16'."""
    p = KLASY_PROGI
    opisy = [f"0–{p[0] - 1}"]
    opisy += [f"{p[i]}–{p[i + 1] - 1}" for i in range(len(p) - 1)]
    return opisy + [f"≥{p[-1]}"]


def liczba(v: float) -> str:
    """
    Liczba do pola tekstowego — bez notacji wykładniczej.

    Format „%g" przy współrzędnych terenowych daje 5.58117e+06, a zatwierdzenie
    takiego pola gubiłoby kilka metrów. Milimetr wystarczy w zupełności.
    """
    return f"{v:.3f}".rstrip("0").rstrip(".") or "0"


class App:
    METRICS = ["liczba promieni", "suma dróg [m]", "waga 0–1", "anizotropia",
               "pokrycie kątowe", "luka kątowa [°]"]
    # Trzy ostatnie to metryki kątowe: kształt chmury kierunków, ułamek
    # pokrytych azymutów i największa dziura. Mierzą co innego — patrz
    # docstring luka_katowa w raycov.py.
    ANIZO = 3                          # indeks metryki anizotropii w METRICS
    POKRYCIE = 4                       # indeks pokrycia kątowego
    LUKA = 5                           # indeks największej luki kątowej
    MODES = ["PS", "czujnik"]          # PS = punkt strzałowy

    def __init__(self, width=100.0, height=200.0, nx=10, ny=20,
                 x0=0.0, y0=0.0):
        self.grid = Grid.from_size(width, height, nx, ny, x0, y0)
        self.src = self._edge("L", 15)
        self.rec = self._edge("P", 25)
        self.mode = 0
        self.metric = 0
        self.icmap = 0
        self.show_nums = True
        self.show_rays = False
        self.mask_zero = True
        self.hover = None
        self.picked = None          # punkt, który zdejmie prawy klik
        self.cov = None
        self._bg = None             # bitmapa tła do blitowania
        self._saving = False        # trwa savefig — nie ruszać bufora
        self._pan = None            # stan przeciągania widoku środkowym klawiszem
        self._reset_view = True     # następny draw() ma dopasować widok do obszaru
        self._pkt = {}              # punkty jako macierze — patrz _tablica()

        self._build()
        self.recompute()

    def _edge(self, edge, n):
        """Punkty wzdłuż krawędzi obszaru."""
        return [(float(x), float(y))
                for x, y in edge_points(self.grid, edge, n)]

    def _tablica(self, ktore):
        """
        Punkty ("src" / "rec") jako macierz N × 2, trzymana między zdarzeniami.

        _pick() woła się przy każdym drgnięciu myszy, a przepisywanie listy
        kilkuset krotek do numpy dwa razy na zdarzenie widać przy ciągnięciu
        kursora. Cache kasuje recompute(), czyli każde miejsce, w którym punkty
        mogą się zmienić; różnica długości łapie to nawet, gdyby ktoś kiedyś
        dopisał punkt z pominięciem tej drogi.
        """
        pkt = getattr(self, ktore)
        tab = self._pkt.get(ktore)
        if tab is None or len(tab) != len(pkt):
            tab = np.asarray(pkt, dtype=float).reshape(-1, 2)
            self._pkt[ktore] = tab
        return tab

    # ================================================================ budowa
    def _build(self):
        self.fig = plt.figure(figsize=(14.5, 8.6))
        self.fig.canvas.manager.set_window_title("Pokrycie promieniami — geometria pomiaru")
        self._can_blit = hasattr(self.fig.canvas, "copy_from_bbox")

        self.ax = self.fig.add_axes([0.045, 0.09, 0.42, 0.85])
        self.cax = self.fig.add_axes([0.478, 0.09, 0.013, 0.85])
        self.ax.set_aspect("equal")

        def label(y, text):
            self.fig.text(0.60, y, text, color=MUTED, size=8.5,
                          weight="bold", family="monospace")

        def panel(rect):
            a = self.fig.add_axes(rect)
            a.set_facecolor(PANEL)
            for s in a.spines.values():
                s.set_color(LINE)
            return a

        self._boxes = []                 # do obsługi migającego kursora

        def textbox(rect, name, initial, on_submit):
            tb = TextBox(panel(rect), name, initial=initial,
                         color="#ffffff", hovercolor=PANEL)
            tb.label.set_color(MUTED)
            tb.text_disp.set_color(PAPER)
            tb.cursor.set_color(PAPER)
            if on_submit is not None:
                tb.on_submit(on_submit)
            self._boxes.append(tb)
            return tb

        # ---- obszar badany
        label(0.958, "OBSZAR  BADANY  [m]")
        self.tb_w = textbox([0.660, 0.918, 0.09, 0.030], "szer. X ",
                            liczba(self.grid.width), lambda t: self._resize(w=t))
        self.tb_h = textbox([0.885, 0.918, 0.09, 0.030], "wys. Y ",
                            liczba(self.grid.height), lambda t: self._resize(h=t))
        self.tb_x0 = textbox([0.660, 0.878, 0.09, 0.030], "zero X₀ ",
                             liczba(self.grid.xmin), lambda t: self._set_origin(x=t))
        self.tb_y0 = textbox([0.885, 0.878, 0.09, 0.030], "zero Y₀ ",
                             liczba(self.grid.ymin), lambda t: self._set_origin(y=t))
        self.fig.text(0.605, 0.851, "X₀, Y₀ = lewy dolny róg liczonego obszaru; "
                                    "punkty zostają na swoich miejscach",
                      color=MUTED, size=7.5)

        # ---- siatka
        label(0.818, "SIATKA")
        self.tb_dx = textbox([0.665, 0.778, 0.085, 0.030], "oczko X [m] ",
                             liczba(self.grid.dx), lambda t: self._set_oczko(dx=t))
        self.tb_dy = textbox([0.885, 0.778, 0.085, 0.030], "oczko Y [m] ",
                             liczba(self.grid.dy), lambda t: self._set_oczko(dy=t))
        self.txt_cell = self.fig.text(0.605, 0.742, "", color=PAPER, size=8.5)

        # ---- co stawiam
        label(0.678, "STAWIAM  NA  MAPIE")
        self.rb_mode = RadioButtons(panel([0.605, 0.588, 0.135, 0.076]),
                                    self.MODES, active=0,
                                    activecolor=SHOT, label_props={"color": [PAPER, PAPER]})
        self.rb_mode.on_clicked(self._set_mode)

        self.tb_n = textbox([0.815, 0.630, 0.055, 0.030], "ile sztuk ", "15", None)

        self._edge_btns = []
        for i, (edge, name) in enumerate(zip("LPGD", ["lewa", "prawa", "góra", "dół"])):
            b = Button(panel([0.755 + i * 0.058, 0.588, 0.052, 0.030]), name,
                       color=INK, hovercolor=PANEL)
            b.label.set_color(PAPER)
            b.label.set_size(8)
            b.on_clicked(lambda _e, ed=edge: self._fill_edge(ed))
            self._edge_btns.append(b)

        self.b_cs = Button(panel([0.605, 0.544, 0.155, 0.030]), "usuń PS",
                           color=INK, hovercolor=PANEL)
        self.b_cr = Button(panel([0.775, 0.544, 0.155, 0.030]), "usuń czujniki",
                           color=INK, hovercolor=PANEL)
        for b, arr in ((self.b_cs, "src"), (self.b_cr, "rec")):
            b.label.set_color(MUTED)
            b.label.set_size(8)
            b.on_clicked(lambda _e, a=arr: self._clear(a))

        # ---- mapa
        label(0.492, "MAPA")
        self.rb_metric = RadioButtons(panel([0.605, 0.372, 0.175, 0.104]),
                                      self.METRICS, active=0, activecolor=GEO,
                                      label_props={"color": [PAPER] * len(self.METRICS),
                                                   "fontsize": [8.5] * len(self.METRICS)})
        self.rb_metric.on_clicked(self._set_metric)

        self.cb = CheckButtons(panel([0.795, 0.372, 0.175, 0.104]),
                               ["liczby", "promienie", "maska zer"],
                               actives=[self.show_nums, self.show_rays, self.mask_zero],
                               label_props={"color": [PAPER] * 3},
                               frame_props={"edgecolor": MUTED},
                               check_props={"facecolor": GEO})
        self.cb.on_clicked(self._toggle)

        # ---- pliki
        label(0.302, "PLIKI")
        self.b_wczytaj = Button(panel([0.605, 0.256, 0.085, 0.032]),
                                "wczytaj punkty", color=INK, hovercolor=PANEL)
        self.b_qgis = Button(panel([0.697, 0.256, 0.085, 0.032]),
                             "→ QGIS", color=INK, hovercolor=PANEL)
        self.b_csv = Button(panel([0.789, 0.256, 0.085, 0.032]), "siatka → CSV",
                            color=INK, hovercolor=PANEL)
        self.b_png = Button(panel([0.881, 0.256, 0.089, 0.032]), "rysunek → PNG",
                            color=INK, hovercolor=PANEL)
        for b in (self.b_wczytaj, self.b_qgis, self.b_csv, self.b_png):
            b.label.set_color(PAPER)
            b.label.set_size(7)
        self.b_wczytaj.on_clicked(self._wczytaj_plik)
        self.b_qgis.on_clicked(self._export_qgis)
        self.b_csv.on_clicked(self._export_csv)
        self.b_png.on_clicked(self._export_png)
        self.txt_plik = self.fig.text(0.605, 0.228, "", color=MUTED, size=7.5)

        # ---- statystyka
        label(0.196, "STATYSTYKA  POKRYCIA")
        self.txt_stats = self.fig.text(0.605, 0.045, "", color=PAPER, size=9,
                                       va="bottom", linespacing=1.9)
        self.txt_hover = self.fig.text(0.045, 0.025, "", color=MUTED, size=8.5)
        self.txt_hover.set_animated(True)     # rysowany osobno, przy blitowaniu

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("draw_event", self._cache_bg)

        # migający kursor w polach tekstowych
        self._caret_on = True
        self._caret_timer = None
        # odłożone przerysowanie po zoomie — patrz _pozniej()
        self._timer_zoom = None
        try:
            self._caret_timer = self.fig.canvas.new_timer(interval=530)
            self._caret_timer.add_callback(self._blink_caret)
            self._caret_timer.start()
            self._timer_zoom = self.fig.canvas.new_timer(interval=200)
            self._timer_zoom.single_shot = True
            self._timer_zoom.add_callback(self.draw)
        except (AttributeError, NotImplementedError):
            pass                     # backend bez pętli zdarzeń (Agg w testach)

    def _blink_caret(self):
        """
        matplotlib stawia karetkę w polu tekstowym, ale nią nie miga — nie widać
        wtedy, że pole jest aktywne i czeka na wpisanie.

        Przerysowujemy wyłącznie prostokąt tego jednego pola (tło, tekst,
        karetka) i blitujemy go. Pełne odświeżenie figury kosztuje ~0,2 s, więc
        miganie co pół sekundy uczyniłoby pisanie nieznośnym.
        """
        active = [tb for tb in self._boxes if tb.capturekeystrokes]
        if not active:
            return
        self._caret_on = not self._caret_on
        for tb in active:
            tb.cursor.set_visible(self._caret_on)
            try:
                tb.ax.draw_artist(tb.ax.patch)
                tb.ax.draw_artist(tb.text_disp)
                if self._caret_on:
                    tb.ax.draw_artist(tb.cursor)
                self.fig.canvas.blit(tb.ax.bbox)
            except (AttributeError, RuntimeError):
                self.fig.canvas.draw_idle()

    # ================================================================ blitting
    #
    # Pełne przerysowanie figury kosztuje ~230 ms — przy każdym drgnięciu myszy
    # to jest 9 klatek na sekundę i rosnąca kolejka zdarzeń. A przy ruchu
    # kursora zmieniają się tylko dwie rzeczy: pasek u dołu i obwódka
    # celownika. Więc zapamiętujemy resztę jako bitmapę i dorysowujemy na niej
    # te dwa obiekty (stąd animated=True — pełny rysunek ma je pomijać).
    #
    def _cache_bg(self, _ev=None):
        # savefig też wywołuje rysowanie, ale w swoim dpi: zapamiętane wtedy tło
        # miałoby zły rozmiar, a domalowany celownik wszedłby do pliku
        if not self._can_blit or self._saving:
            return
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # tło nie zawiera animowanych obiektów, więc po pełnym rysowaniu pasek
        # i obwódka zniknęłyby z ekranu do najbliższego ruchu myszy
        if getattr(self, "hl", None) is not None:
            self._blit()

    def _blit(self):
        if not self._can_blit or self._bg is None:
            self.fig.canvas.draw_idle()
            return
        c = self.fig.canvas
        c.restore_region(self._bg)
        self.ax.draw_artist(self.hl)
        self.fig.draw_artist(self.txt_hover)
        c.blit(self.fig.bbox)

    # =========================================================== reakcje panelu
    def _set_mode(self, lab):
        self.mode = self.MODES.index(lab)
        self.rb_mode.activecolor = SHOT if self.mode == 0 else GEO

    def _set_metric(self, lab):
        self.metric = self.METRICS.index(lab)
        self.draw()

    def _toggle(self, lab):
        self.show_nums, self.show_rays, self.mask_zero = self.cb.get_status()
        self.draw()

    def _resize(self, w=None, h=None):
        try:
            width = float(w) if w is not None else self.grid.width
            height = float(h) if h is not None else self.grid.height
        except ValueError:
            return
        if width <= 0 or height <= 0:
            return
        # Punktów NIE dociskamy do nowego obszaru. Obszar to okno analizy,
        # a punkty to dane — przy wczytanych współrzędnych terenowych
        # dociśnięcie oznaczałoby trwałe przesunięcie pomiaru. Promienie do
        # punktów spoza okna i tak liczą się poprawnie, bo traverse() przycina
        # je do siatki.
        self.grid = Grid.from_size(width, height, self.grid.nx, self.grid.ny,
                                   *self.grid.origin)
        self.refit()                 # obszar zmienił rozmiar
        self.recompute()

    def _set_origin(self, x=None, y=None):
        """
        Przesuwa obszar liczony. Punkty zostają na swoich miejscach — to okno
        analizy jedzie, nie dane. Promienie do punktów spoza okna nadal liczą
        się poprawnie, bo traverse() przycina je do siatki.
        """
        try:
            x0 = float(x) if x is not None else self.grid.xmin
            y0 = float(y) if y is not None else self.grid.ymin
        except ValueError:
            return
        if (x0, y0) == self.grid.origin:
            return
        self.grid = Grid.from_size(self.grid.width, self.grid.height,
                                   self.grid.nx, self.grid.ny, x0, y0)
        self.refit()                 # obszar pojechał gdzie indziej
        self.recompute()

    def _on_release(self, _ev):
        if self._pan is not None:
            self._pan = None
            self._view_changed(full=True)    # dociągnij liczby do nowego kadru

    MAX_N = 500

    def _set_oczko(self, dx=None, dy=None):
        """
        Rozmiar kratki wpisany w metrach.

        Wpisane oczko zostaje dokładnie takie, jakie podałeś — to obszar
        dociągamy do całkowitej liczby kratek, a nie odwrotnie. Inaczej po
        wpisaniu „5" wychodziłoby 5,03 m i raster w QGIS miałby inne oczko,
        niż mówi opis.
        """
        g = self.grid
        try:
            new_dx = float(str(dx).replace(",", ".")) if dx is not None else g.dx
            new_dy = float(str(dy).replace(",", ".")) if dy is not None else g.dy
        except ValueError:
            self._show_oczko()                   # śmieci — przywróć poprzednie
            return
        # oczko nie mniejsze niż na MAX_N kratek i nie większe niż cały obszar
        new_dx = min(max(new_dx, g.width / self.MAX_N), g.width)
        new_dy = min(max(new_dy, g.height / self.MAX_N), g.height)
        nx = max(int(round(g.width / new_dx)), 1)
        ny = max(int(round(g.height / new_dy)), 1)
        self.grid = Grid.from_size(nx * new_dx, ny * new_dy, nx, ny, *g.origin)
        self._show_oczko()
        self._odswiez_pola()                     # obszar mógł się nieco zmienić
        # Widok dopasowujemy tylko wtedy, gdy obszar naprawdę urósł lub zmalał.
        # Dociągnięcie do całych kratek zmienia go zwykle o metry i kasowanie
        # przy tym zoomu byłoby uciążliwe.
        if (abs(self.grid.width - g.width) > 0.01 * g.width
                or abs(self.grid.height - g.height) > 0.01 * g.height):
            self.refit()
        self.recompute()

    def _show_oczko(self):
        """Wpisz do pól bieżące oczko, nie odpalając ich zdarzeń."""
        for tb, v in ((self.tb_dx, liczba(self.grid.dx)),
                      (self.tb_dy, liczba(self.grid.dy))):
            if tb.text != v:
                tb.eventson = False
                tb.set_val(v)
                tb.eventson = True

    def _fill_edge(self, edge):
        try:
            n = max(1, int(float(self.tb_n.text)))
        except ValueError:
            return
        target = self.src if self.mode == 0 else self.rec
        target.extend(self._edge(edge, n))
        self.recompute()

    def _clear(self, which):
        setattr(self, which, [])
        self.recompute()

    # ================================================================= mysz
    def _pick(self, x, y):
        """
        Punkt pod kursorem jako ("src"|"rec", indeks), albo None.

        Pierwszeństwo ma typ aktualnie wybrany
        w trybie: gdy czujnik stoi na PS, sam decydujesz który zdjąć
        (przełącznik 1 / 2). Jeśli w zasięgu nie ma nic tego typu, bierzemy
        drugi — żeby usuwanie nie milczało bez powodu.
        """
        tol = 0.04 * max(self.grid.width, self.grid.height)
        for name in (("src", "rec") if self.mode == 0 else ("rec", "src")):
            pts = self._tablica(name)
            if not len(pts):
                continue
            d = np.hypot(*(pts - (x, y)).T)
            i = int(d.argmin())
            if d[i] < tol:
                return name, i
        return None

    # ---------------------------------------------------------- widok mapy
    def _view_changed(self, full=False):
        """
        Po zmianie zakresów tło do blitowania i pasek skali są nieaktualne.

        full=True przebudowuje rysunek, bo od kadru zależy, które kratki dostają
        liczby — bez tego po przybliżeniu gęstej siatki liczby nigdy by się nie
        pojawiły. Jest to jednak pełne rysowanie (~0,2 s), więc przy ciągnięciu
        widoku robimy to raz, po puszczeniu przycisku.
        """
        self._bg = None
        if full:
            self.draw()
        else:
            self._fit_colorbar()
            self.fig.canvas.draw_idle()

    def _pozniej(self):
        """
        Pełne przerysowanie dopiero, gdy kręcenie kółkiem ustanie.

        Kółko sypie zdarzeniami po kilkanaście na sekundę, a przebudowa rysunku
        (liczby w kratkach zależą od kadru) kosztuje grubo ponad 0,1 s. Zoom
        robimy więc od razu i tanio, a dociągnięcie liczb odkładamy — inaczej
        każdy obrót kółkiem zostawia w kolejce robotę, której nikt nie zobaczy,
        bo zaraz przykryje ją następny obrót.
        """
        if self._timer_zoom is None:
            self._view_changed(full=True)     # backend bez timera (Agg w testach)
            return
        self._view_changed()                  # sam kadr: tanio i natychmiast
        self._timer_zoom.stop()
        self._timer_zoom.start()

    def _on_scroll(self, ev):
        """Kółko myszy — zoom wokół punktu pod kursorem, jak w MATLABie."""
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        f = 1 / 1.3 if ev.button == "up" else 1.3
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(ev.xdata + (x0 - ev.xdata) * f,
                         ev.xdata + (x1 - ev.xdata) * f)
        self.ax.set_ylim(ev.ydata + (y0 - ev.ydata) * f,
                         ev.ydata + (y1 - ev.ydata) * f)
        self._pozniej()

    def _pan_to(self, ev):
        """
        Przesunięcie widoku. Deltę liczymy przez aktualną transformację, ale
        stosujemy do zakresów zapamiętanych przy wciśnięciu — przesunięcie
        układu skraca się w różnicy, więc widok nie ucieka w trakcie ciągnięcia.
        """
        px, py, xl, yl = self._pan
        inv = self.ax.transData.inverted()
        (ax_, ay), (bx, by) = inv.transform([(px, py), (ev.x, ev.y)])
        dx, dy = ax_ - bx, ay - by
        self.ax.set_xlim(xl[0] + dx, xl[1] + dx)
        self.ax.set_ylim(yl[0] + dy, yl[1] + dy)
        self._view_changed()

    def _on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        if ev.button == 2:                           # środkowy — chwyć i przesuń
            self._pan = (ev.x, ev.y, self.ax.get_xlim(), self.ax.get_ylim())
            return
        if getattr(getattr(self.fig.canvas, "toolbar", None), "mode", ""):
            return                                   # aktywne zoom/pan — nie stawiaj
        if ev.button == 3:                           # prawy — usuń pojedynczy
            hit = self._pick(ev.xdata, ev.ydata)
            if hit is None:
                return
            getattr(self, hit[0]).pop(hit[1])
            self.picked = None
        elif ev.button == 1:
            (self.src if self.mode == 0 else self.rec).append(
                self.grid.clamp(ev.xdata, ev.ydata))
        else:
            return
        self.recompute()

    def _on_move(self, ev):
        if self._pan is not None and ev.x is not None:
            self._pan_to(ev)
            return
        if ev.inaxes is not self.ax or ev.xdata is None or self.cov is None:
            if self.hover is not None or self.picked is not None:
                self.hover = self.picked = None
                self.txt_hover.set_text("")
                self._show_picked()
                self._blit()
            return

        # co zdejmie prawy klik — liczone przy każdym ruchu, bo w jednej
        # komórce może stać kilka punktów
        picked = self._pick(ev.xdata, ev.ydata)
        moved = picked != self.picked
        if moved:
            self.picked = picked
            self._show_picked()

        # po oddaleniu widać teren poza obszarem — tam nie ma czego pokazywać
        g = self.grid
        if not (g.xmin <= ev.xdata <= g.xmax and g.ymin <= ev.ydata <= g.ymax):
            if self.hover is not None:
                self.hover = None
                self.txt_hover.set_text("")
                moved = True
            if moved:
                self._blit()
            return

        ix, iy = self.grid.index_of(ev.xdata, ev.ydata)
        if (ix, iy) != self.hover:
            self.hover = (ix, iy)
            xc, yc = self.grid.cell_center(ix, iy)
            f = self._field()
            # nanmax, bo anizotropia jest NaN w kratkach bez pokrycia
            fmax = float(np.nanmax(f)) if np.isfinite(f).any() else 0.0
            w = f[iy, ix] / fmax if fmax and np.isfinite(f[iy, ix]) else 0.0
            what = ("   → prawy klik zdejmie: "
                    + ("PS" if self.picked[0] == "src" else "czujnik")
                    ) if self.picked else ""
            ekw = self.cov.length[iy, ix] / self.grid.srednia_ciecziwa
            # Trzy metryki kątowe stoją obok siebie celowo: dopiero razem widać,
            # że kratka z anizotropią 0,1 potrafi mieć pokrycie 0,25 i nie
            # widzieć klina szerokiego na 60°.
            def kat(tab, fmt, jednostka=""):
                v = tab[iy, ix]
                # kreska bez jednostki: „—°" wyglądałoby jak zero stopni
                return "—" if not np.isfinite(v) else format(v, fmt) + jednostka

            self.txt_hover.set_text(
                f"komórka [{ix}, {iy}]   środek {xc:.2f} ; {yc:.2f} m   "
                f"promieni {self.cov.hits[iy, ix]}   "
                f"suma dróg {self.cov.length[iy, ix]:.2f} m "
                f"(ekwiwalent {ekw:.1f})   waga {w:.3f}   "
                f"anizotropia {kat(self.cov.anisotropy, '.2f')}   "
                f"pokrycie {kat(self.cov.pokrycie_katowe, '.2f')}   "
                f"luka {kat(self.cov.luka_katowa, '.0f', '°')}{what}")
            moved = True
        if moved:
            self._blit()

    def _show_picked(self):
        """Obwódka na punkcie, który zniknie po prawym kliknięciu."""
        if self.picked is None:
            self.hl.set_data([], [])
        else:
            name, i = self.picked
            x, y = getattr(self, name)[i]
            self.hl.set_data([x], [y])

    def _on_key(self, ev):
        if ev.key in "12":
            self.rb_mode.set_active(int(ev.key) - 1)
        elif ev.key == "c":
            self.icmap = (self.icmap + 1) % len(CMAPS)
            self.draw()
        elif ev.key == "r":
            self.cb.set_active(1)
        elif ev.key == "n":
            self.cb.set_active(0)
        elif ev.key == "0":
            self.refit()
            self.draw()

    # ============================================================ wczytywanie
    def _tk(self):
        """Korzeń tkintera — przy backendzie TkAgg już istnieje."""
        import tkinter as tk
        korzen = getattr(tk, "_default_root", None)
        if korzen is None:                       # inny backend: własny, ukryty
            korzen = tk.Tk()
            korzen.withdraw()
        return tk, korzen

    def _okno_przypisania(self, grupy):
        """
        Pyta, która grupa to PS, a która czujniki. Nazwy z pliku bywają lokalne
        (T-271, W-270a), więc zgadywanie po nich to tylko podpowiedź — decyzja
        należy do użytkownika. Zwraca (przypisania, zamień_xy) albo None.
        """
        tk, korzen = self._tk()
        okno = tk.Toplevel(korzen)
        okno.title("Wczytane grupy punktów")
        okno.resizable(False, False)
        okno.grab_set()

        tk.Label(okno, text="Co jest czym?", font=("TkDefaultFont", 10, "bold")
                 ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))
        wybory = {}
        for i, g in enumerate(grupy, start=1):
            tk.Label(okno, text=f"{g.nazwa}   ({len(g)} pkt)", anchor="w"
                     ).grid(row=i, column=0, sticky="w", padx=(12, 8), pady=2)
            v = tk.StringVar(value="PS" if g.wyglada_na_zrodla else "czujnik")
            tk.OptionMenu(okno, v, "PS", "czujnik", "pomiń").grid(
                row=i, column=1, sticky="ew", padx=(0, 12), pady=2)
            wybory[g.nazwa] = v

        zamien = tk.BooleanVar(value=False)
        tk.Checkbutton(okno, text="zamień kolumny X ↔ Y", variable=zamien).grid(
            row=len(grupy) + 1, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 0))
        tk.Label(okno, text="(zaznacz, jeśli mapa wyjdzie obrócona)", fg="#666"
                 ).grid(row=len(grupy) + 2, column=0, columnspan=2, sticky="w", padx=12)

        wynik = {}
        def ok():
            wynik["dane"] = ({n: v.get() for n, v in wybory.items()}, zamien.get())
            okno.destroy()
        ramka = tk.Frame(okno)
        ramka.grid(row=len(grupy) + 3, column=0, columnspan=2, pady=12)
        tk.Button(ramka, text="Wczytaj", width=12, command=ok).pack(side="left", padx=4)
        tk.Button(ramka, text="Anuluj", width=12, command=okno.destroy).pack(side="left", padx=4)

        okno.wait_window()
        return wynik.get("dane")

    def _wczytaj_plik(self, _ev=None, sciezka=None, przypisz=None, zamien=False):
        """
        Wczytuje punkty z pliku i dopasowuje obszar do danych.

        Argumenty poza pierwszym służą do wywołania bez okien (testy, skrypty).
        """
        if sciezka is None:
            self._tk()                           # upewnij się, że jest korzeń Tk
            from tkinter import filedialog
            sciezka = filedialog.askopenfilename(
                title="Wybierz plik ze współrzędnymi",
                filetypes=[("Arkusze i tabele", "*.xlsx *.csv *.txt *.dat"),
                           ("Wszystkie pliki", "*.*")])
            if not sciezka:
                return
        try:
            grupy = wczytaj_punkty(sciezka)
        except Exception as e:                   # zły format, uszkodzony plik…
            self._info_plik(f"nie wczytano: {e}", WARN)
            return

        if przypisz is None:
            wynik = self._okno_przypisania(grupy)
            if wynik is None:
                return
            przypisz, zamien = wynik
        if zamien:
            grupy = [Grupa(g.nazwa, g.punkty[:, ::-1]) for g in grupy]

        src, rec, opis = [], [], []
        for g in grupy:
            rola = przypisz.get(g.nazwa, "pomiń")
            cel = src if rola == "PS" else rec if rola == "czujnik" else None
            if cel is None:
                continue
            cel.extend(map(tuple, g.punkty))
            opis.append(f"{g.nazwa}→{rola} ({len(g)})")
        if not src and not rec:
            self._info_plik("nic nie wczytano — wszystkie grupy pominięte", WARN)
            return

        # Obszar dopasowany do danych, z zapasem. Rozmiar oczka zostawiamy bez
        # zmian — sam ustawiłeś go świadomie, więc wczytanie pliku go nie rusza.
        pkt = np.array(src + rec, dtype=float)
        zapas = 0.05 * max(np.ptp(pkt[:, 0]), np.ptp(pkt[:, 1]), 1.0)
        x0, y0 = pkt[:, 0].min() - zapas, pkt[:, 1].min() - zapas
        w, h = np.ptp(pkt[:, 0]) + 2 * zapas, np.ptp(pkt[:, 1]) + 2 * zapas
        nx = min(max(int(round(w / self.grid.dx)), 1), self.MAX_N)
        ny = min(max(int(round(h / self.grid.dy)), 1), self.MAX_N)
        # obszar rozciągamy do całkowitej liczby oczek, żeby ich rozmiar został
        # dokładnie ten sam, jaki jest wpisany w polach
        w2, h2 = nx * self.grid.dx, ny * self.grid.dy
        x0 -= (w2 - w) / 2
        y0 -= (h2 - h) / 2
        self.grid = Grid.from_size(w2, h2, nx, ny, x0, y0)
        self.src, self.rec = src, rec
        self._odswiez_pola()
        self.refit()
        self.recompute()
        self._info_plik(f"{os.path.basename(sciezka)}:  " + ",  ".join(opis), PAPER)
        print(f"wczytano {sciezka}: {len(src)} PS, {len(rec)} czujników; "
              f"obszar {w:.0f} × {h:.0f} m, siatka {nx} × {ny}")

    def _info_plik(self, tekst, kolor=None):
        self.txt_plik.set_text(tekst[:78])
        self.txt_plik.set_color(kolor or MUTED)
        self.fig.canvas.draw_idle()

    def _odswiez_pola(self):
        """Wpisz bieżącą geometrię do pól, nie odpalając ich zdarzeń."""
        g = self.grid
        for tb, v in ((self.tb_w, liczba(g.width)), (self.tb_h, liczba(g.height)),
                      (self.tb_x0, liczba(g.xmin)), (self.tb_y0, liczba(g.ymin)),
                      (self.tb_dx, liczba(g.dx)), (self.tb_dy, liczba(g.dy))):
            if tb.text != v:
                tb.eventson = False
                tb.set_val(v)
                tb.eventson = True

    # ============================================================== eksport
    def _export_qgis(self, _ev=None, katalog="."):
        """
        Wyniki w formatach, które QGIS otwiera wprost.

        Siatka idzie jako ESRI ASCII Grid — po jednym pliku na metrykę. To zwykły
        tekst, więc nie wymaga żadnej biblioteki, a QGIS wczytuje go jako raster
        i pozwala nadać mu tę samą klasyfikację co tutaj. Punkty idą jako
        GeoJSON, bo jest czytelny i nie gubi atrybutów.
        """
        g, cov = self.grid, self.cov
        pola = {
            "liczba_promieni": cov.hits.astype(float),
            "suma_drog_m": cov.length,
            "ekwiwalent_promieni": cov.ekwiwalent,
            "pokrycie_katowe": cov.pokrycie_katowe,
            "anizotropia": cov.anisotropy,
            "luka_katowa_st": cov.luka_katowa,
        }
        BRAK = -9999.0
        zapisane = []
        for nazwa, tab in pola.items():
            t = np.where(np.isfinite(tab), tab, BRAK)
            t = np.where(cov.hits == 0, BRAK, t)
            sciezka = os.path.join(katalog, f"siatka_{nazwa}.asc")
            with open(sciezka, "w", encoding="ascii") as f:
                f.write(f"ncols {g.nx}\nnrows {g.ny}\n"
                        f"xllcorner {g.xmin:.4f}\nyllcorner {g.ymin:.4f}\n")
                # cellsize tylko dla oczek kwadratowych; inaczej para dx/dy,
                # którą GDAL też rozumie
                if abs(g.dx - g.dy) < 1e-9:
                    f.write(f"cellsize {g.dx:.6f}\n")
                else:
                    f.write(f"dx {g.dx:.6f}\ndy {g.dy:.6f}\n")
                f.write(f"NODATA_value {BRAK:.0f}\n")
                for iy in range(g.ny - 1, -1, -1):        # ASC idzie od góry
                    f.write(" ".join(f"{v:.4f}" for v in t[iy]) + "\n")
            zapisane.append(os.path.basename(sciezka))

        obiekty = []
        for punkty, typ in ((self.src, "PS"), (self.rec, "czujnik")):
            for i, (x, y) in enumerate(punkty, 1):
                obiekty.append(
                    '{"type":"Feature","geometry":{"type":"Point","coordinates":'
                    f'[{x:.4f},{y:.4f}]}},"properties":{{"typ":"{typ}","nr":{i}}}}}')
        pkt = os.path.join(katalog, "punkty.geojson")
        with open(pkt, "w", encoding="utf-8") as f:
            # Bez deklaracji układu — sama geometria. Wpisany tu kod EPSG byłby
            # tylko zgadywaniem z rzędu wielkości liczb, a QGIS przyjąłby go bez
            # pytania; błędny układ jest gorszy niż jego brak.
            f.write('{"type":"FeatureCollection","features":[\n'
                    + ",\n".join(obiekty) + "\n]}\n")
        zapisane.append(os.path.basename(pkt))

        czytaj = os.path.join(katalog, "README_qgis.txt")
        with open(czytaj, "w", encoding="utf-8") as f:
            f.write(
                "Pliki dla QGIS\n"
                "==============\n\n"
                f"obszar   {g.xmin:.2f} .. {g.xmax:.2f}  ×  {g.ymin:.2f} .. {g.ymax:.2f}\n"
                f"siatka   {g.nx} × {g.ny}, oczko {g.dx:g} × {g.dy:g} m\n"
                f"promieni {cov.n_rays}\n\n"
                "UKŁAD WSPÓŁRZĘDNYCH\n"
                "  Pliki zawierają samą geometrię, bez deklaracji układu.\n"
                "  Nadaj go w QGIS — ten sam, w którym podane były współrzędne\n"
                "  wejściowe: prawy klik na warstwie → Ustaw układ współrzędnych.\n"
                "\nRASTRY (siatka_*.asc)\n"
                "  Warstwa → Dodaj warstwę → Dodaj warstwę rastrową.\n"
                "  Puste kratki mają wartość -9999 (NODATA), więc będą przezroczyste.\n\n"
                "  liczba_promieni      ile promieni przecięło kratkę\n"
                "  suma_drog_m          łączna droga promieni w kratce [m]\n"
                "  ekwiwalent_promieni  suma dróg / średnia cięciwa kratki\n"
                f"                       (tu: / {g.srednia_ciecziwa:.3f} m)\n"
                "                       — metry przeliczone na sztuki promieni,\n"
                "                       żeby stosowały się te same progi\n"
                "  anizotropia          kształt chmury kierunków: 0 = okrągła,\n"
                "                       1 = wszystkie promienie równoległe\n"
                "  pokrycie_katowe      ułamek trafionych azymutów (180° na 12\n"
                "                       sektorów po 15°); 1 = ze wszystkich stron.\n"
                "                       UWAGA: rośnie najwyżej tak szybko jak\n"
                "                       liczba promieni — 6 promieni nie da\n"
                "                       więcej niż 0,5, choćby szły idealnie\n"
                "                       równomiernie co 30°\n"
                "  luka_katowa_st       najszerszy klin azymutów [°], z którego\n"
                "                       nie przyszedł ŻADEN promień; 0 = kratka\n"
                "                       widziana ze wszystkich stron\n\n"
                "  UWAGA: anizotropia nie zastępuje luki. Liczy się z tensora\n"
                "  drugiego rzędu, a ten mierzy tylko, czy chmura kierunków jest\n"
                "  okrągła — dziur w niej nie widzi. Trzy kierunki co 60° dają\n"
                "  anizotropię 0,000, tyle samo co pełne 180° pokryte gęsto.\n"
                "  Przy PS po jednej stronie i czujnikach po drugiej środek\n"
                "  obszaru wychodzi na 0,1 („idealnie”), choć nie przechodzi\n"
                "  przez niego żaden promień bliski pionowi. Do oceny, czy\n"
                "  kratka jest oświetlona ze wszystkich stron, patrz na lukę.\n\n"
                "KLASYFIKACJA (Właściwości warstwy → Symbolizacja → Jednopasmowy paletowy)\n"
                "  Progi są STAŁE — te same przy każdym rozmiarze oczka:\n\n"
                + "".join(
                    f"      {opis:<8} {nazwa:<12} {barwa}\n"
                    for opis, nazwa, barwa in zip(
                        klasy_opisy(),
                        ("czerwony", "pomarańcz.", "żółty", "niebieski", "zielony"),
                        KLASY_BARWY))
                + "\n"
                "  Te same progi stosuj do 'liczba_promieni' i 'ekwiwalent_promieni'.\n"
                "  Anizotropia ma własną skalę 0–1 (0 dobrze, 1 źle),\n"
                "  luka kątowa własną 0–180° (0 dobrze, 180 źle).\n\n"
                "PUNKTY (punkty.geojson)\n"
                "  Warstwa → Dodaj warstwę → Dodaj warstwę wektorową.\n"
                "  Atrybut 'typ' rozróżnia PS od czujników.\n")
        zapisane.append(os.path.basename(czytaj))

        self._info_plik("QGIS: " + ", ".join(zapisane[:2]) + f" … ({len(zapisane)} plików)",
                        PAPER)
        print("zapisano dla QGIS:\n  " + "\n  ".join(zapisane))
        print("  układ współrzędnych nadaj w QGIS")

    def _export_csv(self, _ev):
        p = self.cov.to_csv("siatka_pokrycie.csv")
        g = self.grid
        with open("geometria.csv", "w", encoding="utf-8") as f:
            # bez tej linii nie da się odtworzyć, w jakim układzie są te punkty
            f.write(f"# obszar {liczba(g.xmin)},{liczba(g.ymin)}"
                    f" .. {liczba(g.xmax)},{liczba(g.ymax)}"
                    f"  siatka {g.nx}x{g.ny}\n")
            f.write("typ,x,y\n")
            for x, y in self.src:
                f.write(f"PS,{x:.4f},{y:.4f}\n")
            for x, y in self.rec:
                f.write(f"czujnik,{x:.4f},{y:.4f}\n")
        print(f"zapisano: {p} oraz geometria.csv")

    def _export_png(self, _ev):
        # Uwaga: przy zapisie matplotlib rysuje także obiekty animated=True
        # (Axes.draw pomija je tylko gdy not canvas.is_saving()), więc celownik
        # i pasek statusu trzeba schować ręcznie — to interfejs, nie wynik.
        picked, hover = self.picked, self.txt_hover.get_text()
        self.picked = None
        self._show_picked()
        self.txt_hover.set_text("")
        self._saving = True
        try:
            self.fig.savefig("mapa_pokrycia.png", dpi=200, facecolor=INK)
        finally:
            self._saving = False
            self.picked = picked
            self._show_picked()
            self.txt_hover.set_text(hover)
            self.fig.canvas.draw_idle()      # odśwież tło w rozdzielczości ekranu
        print("zapisano: mapa_pokrycia.png")

    # ============================================================ obliczenia
    def _field(self):
        if self.metric == 0:
            return self.cov.hits.astype(float)
        if self.metric == 1:
            return self.cov.length
        if self.metric == self.ANIZO:
            return self.cov.anisotropy      # 0–1, w kratkach bez promieni: NaN
        if self.metric == self.POKRYCIE:
            return self.cov.pokrycie_katowe  # 0–1, bez promieni: NaN
        if self.metric == self.LUKA:
            return self.cov.luka_katowa     # stopnie, bez promieni: NaN
        return self.cov.weight_hits

    def refit(self):
        """Dopasuj widok do całego obszaru przy najbliższym rysowaniu."""
        self._reset_view = True

    def recompute(self):
        # build_matrix=False: mapy pokrycia nie potrzebują macierzy G, a to ona
        # kosztuje najwięcej — przy ciągnięciu suwaka siatki to widać
        self.picked = None          # indeksy się przesunęły, celownik nieaktualny
        self._pkt.clear()           # punkty mogły się zmienić — patrz _tablica()
        self.cov = compute(self.grid, self.src, self.rec, build_matrix=False)
        g = self.grid
        self.txt_cell.set_text(f"siatka {g.nx} × {g.ny} = {g.n_cells} kratek"
                               f"   ·   progi klas "
                               f"{' / '.join(str(p) for p in KLASY_PROGI)}")
        self.draw()

    # =============================================================== rysunek
    def _fit_colorbar(self):
        """
        Przy aspect='equal' mapa nie wypełnia całego prostokąta osi — przy
        obszarze wysokim i wąskim zostaje puste pole po bokach. Doklejamy
        pasek skali do rzeczywistej krawędzi rysunku, a nie do krawędzi osi.
        """
        box = self.ax.get_position()
        fw, fh = self.fig.get_size_inches()
        box_ratio = (box.height * fh) / (box.width * fw)
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()   # widoczny wycinek,
        span_x, span_y = xl[1] - xl[0], yl[1] - yl[0]     # nie cały obszar —
        if not span_x or not span_y:                      # inaczej pasek skali
            return                                        # ucieka po zoomie
        data_ratio = abs(span_y / span_x)
        if data_ratio > box_ratio:                       # ograniczone wysokością
            w = box.width * (box_ratio / data_ratio)
            x0, h, y0 = box.x0 + (box.width - w) / 2, box.height, box.y0
        else:                                            # ograniczone szerokością
            h = box.height * (data_ratio / box_ratio)
            x0, w, y0 = box.x0, box.width, box.y0 + (box.height - h) / 2
        self.cax.set_position([x0 + w + 0.018, y0, 0.013, h])

    def draw(self):
        g, cov = self.grid, self.cov
        if self._timer_zoom is not None:
            self._timer_zoom.stop()     # odłożone rysowanie właśnie się dzieje
        # Widok liczymy jawnie, zamiast czytać go z osi w trakcie rysowania:
        # plot() i add_collection() po drodze włączają autoskalowanie z powrotem,
        # więc zakresy ustawiamy dopiero na końcu. Po zmianie geometrii
        # dopasowujemy do obszaru, poza tym trzymamy zoom użytkownika — inaczej
        # kasowałby się przy każdym postawionym punkcie.
        # Margines przy dopasowaniu: przyrządy stoją często dokładnie na
        # krawędzi obszaru, a są przycinane (inaczej po zoomie wychodziłyby
        # poza mapę) — bez zapasu widać by było połówki gwiazdek.
        m = 0.02
        view = (((g.xmin - m * g.width, g.xmax + m * g.width),
                 (g.ymin - m * g.height, g.ymax + m * g.height))
                if self._reset_view
                else (self.ax.get_xlim(), self.ax.get_ylim()))
        self._reset_view = False
        self.ax.cla()

        f = self._field()                       # to WPISUJEMY w kratki
        anizo = self.metric == self.ANIZO
        pokrycie = self.metric == self.POKRYCIE
        luka = self.metric == self.LUKA
        katowa = anizo or pokrycie or luka
        # Metryki kątowe nie podlegają klasom: progi profesora są w sztukach
        # promieni, a tu mamy liczbę bezwymiarową 0–1 albo stopnie — jedno
        # i drugie o własnym, ustalonym znaczeniu.
        klasy = CMAPS[self.icmap] == "klasy" and not katowa

        # Po czym KOLORUJEMY. Progi profesora są w sztukach promieni, więc każdą
        # metrykę trzeba najpierw na sztuki sprowadzić:
        #   suma dróg [m]  → dzielimy przez średnią cięciwę (wzór Cauchy'ego)
        #   waga 0–1       → to przeskalowane zliczenia, więc bierzemy je wprost
        # Dzięki temu te mapy mają te same klasy i da się je porównywać, a
        # w kratkach nadal stoją liczby we właściwych jednostkach.
        if klasy:
            baza = (cov.length / g.srednia_ciecziwa if self.metric == 1
                    else cov.hits.astype(float))
        else:
            baza = f
        arr = np.ma.masked_invalid(baza)        # NaN anizotropii poza pokryciem
        if self.mask_zero:
            arr = np.ma.masked_where(cov.hits == 0, arr)

        if klasy:
            cmap = ListedColormap(KLASY_BARWY)
            hi = max(float(baza.max()), KLASY_PROGI[-1]) + 1.0
            norm = BoundaryNorm([0.0, *KLASY_PROGI, hi], cmap.N)
        else:
            nazwa = CMAPS[self.icmap]
            if nazwa == "klasy":
                # W anizotropii i luce dobre jest 0, w pokryciu kątowym dobre
                # jest 1 — skala musi więc lecieć w drugą stronę, inaczej ta
                # sama zieleń znaczyłaby raz „widzi wszystko", raz „nie widzi nic".
                nazwa = "RdYlGn" if pokrycie else "RdYlGn_r"
            cmap = plt.get_cmap(nazwa).copy()
            norm = None
        cmap.set_bad(EMPTY)
        # Metryki kątowe mają skale bezwzględne — anizotropia i pokrycie 0–1,
        # luka 0–180°. Nie normalizujemy ich do maksimum z danych, bo wtedy ta
        # sama wartość na dwóch mapach znaczyłaby co innego.
        if anizo or pokrycie:
            zakres = {"vmin": 0, "vmax": 1}
        elif luka:
            zakres = {"vmin": 0, "vmax": 180}
        else:
            zakres = {"vmin": 0, "vmax": max(float(np.nanmax(baza)), 1e-9)}

        im = self.ax.imshow(
            arr, origin="lower", extent=g.extent, cmap=cmap,
            interpolation="nearest",
            **({"norm": norm} if klasy else zakres))

        # spacing='uniform': każda klasa dostaje na pasku tyle samo miejsca.
        # Przy 'proportional' ostatnia klasa (>=16) rozdymała się do maksimum
        # z danych, więc legenda zmieniała proporcje po każdym postawionym
        # punkcie — a skala ma być stała. Dla map ciągłych bez zmian.
        self.cbar = getattr(self, "cbar", None) or self.fig.colorbar(
            im, cax=self.cax, spacing="uniform")
        self.cbar.update_normal(im)
        if klasy:
            granice = [0.0, *KLASY_PROGI, hi]
            self.cbar.set_ticks([(granice[i] + granice[i + 1]) / 2
                                 for i in range(len(KLASY_BARWY))])
            self.cbar.set_ticklabels(klasy_opisy())
            self.cbar.ax.tick_params(labelsize=8)
            zrodlo = ("ekwiwalent = suma dróg / %.2f m" % g.srednia_ciecziwa
                      if self.metric == 1 else "liczba promieni")
            # Druga linia MUSI zostać: bez niej ktoś przyjmie, że progi zależą
            # od oczka (tak było wcześniej) i będzie szukał przeliczenia.
            opis = (f"klasy: {zrodlo}\n"
                    f"progi stałe, niezależne od oczka ({g.dx:g}×{g.dy:g} m)")
        elif anizo:
            self.cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            self.cbar.ax.tick_params(labelsize=8)
            # „kształt chmury kierunków", nie „ile stron widzi kratka" — tensor
            # 2. rzędu nie widzi dziur w rozkładzie, od tego jest luka kątowa
            opis = ("anizotropia — kształt chmury kierunków\n"
                    "0 = chmura okrągła   1 = wszystkie równoległe")
        elif pokrycie:
            self.cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            self.cbar.ax.tick_params(labelsize=8)
            opis = ("pokrycie kątowe — ułamek trafionych azymutów\n"
                    "1 = promienie ze wszystkich stron   0 = z jednej")
        elif luka:
            self.cbar.set_ticks([0, 45, 90, 135, 180])
            self.cbar.ax.tick_params(labelsize=8)
            opis = ("największa luka kątowa [°]\n"
                    "klin azymutów, z którego nie przyszedł żaden promień")
        else:
            self.cbar.ax.tick_params(labelsize=8, reset=False)
            opis = self.METRICS[self.metric]
        # labelpad, bo etykiety klas są szerokie i opis na nie nachodził
        self.cbar.set_label(opis, color=MUTED, size=8.5, labelpad=8)
        self.cbar.outline.set_edgecolor(LINE)


        # Ile pikseli ma kratka na ekranie. Przy aspect='equal' skala jest ta sama
        # w obu osiach i wyznacza ją ciaśniejszy wymiar — sama szerokość osi
        # kłamałaby, bo mapa nie wypełnia całego prostokąta osi.
        xl, yl = view
        okno = self.ax.get_window_extent()
        skala = min(okno.width / max(abs(xl[1] - xl[0]), 1e-9),
                    okno.height / max(abs(yl[1] - yl[0]), 1e-9))     # px na metr
        px_kratki = min(skala * g.dx, skala * g.dy)

        # Zakres kratek widocznych w kadrze — z tego korzystają i przekreślenia,
        # i liczby. Poza kadrem nie ma po co niczego składać.
        ix0, iy0 = g.index_of(min(xl), min(yl))
        ix1, iy1 = g.index_of(max(xl), max(yl))

        # Przekreślenie komórek bez pokrycia — brak informacji, nie „mała
        # wartość". Rysujemy tylko to, co widać: przy oddaleniu krzyżyk ma
        # poniżej trzech pikseli i zlewa się z tłem, a samych kratek bez
        # pokrycia potrafią być dziesiątki tysięcy — czyli tyleż odcinków do
        # złożenia i narysowania na każdą zmianę. Kratkę bez promieni i tak
        # widać po szarym wypełnieniu.
        if self.mask_zero and px_kratki > 3:
            puste = cov.hits[iy0:iy1 + 1, ix0:ix1 + 1] == 0
            if 0 < int(puste.sum()) <= 8000:
                jy, jx = np.nonzero(puste)
                x0 = g.xmin + (jx + ix0) * g.dx
                y0 = g.ymin + (jy + iy0) * g.dy
                m = 0.18
                lewy, prawy = x0 + m * g.dx, x0 + (1 - m) * g.dx
                dolny, gorny = y0 + m * g.dy, y0 + (1 - m) * g.dy
                ukos = np.concatenate([
                    np.stack([np.column_stack([lewy, dolny]),
                              np.column_stack([prawy, gorny])], axis=1),
                    np.stack([np.column_stack([prawy, dolny]),
                              np.column_stack([lewy, gorny])], axis=1)])
                self.ax.add_collection(LineCollection(ukos, colors=WARN,
                                                      linewidths=0.9, alpha=0.65))

        # promienie
        n_rays = len(self.src) * len(self.rec)
        if self.show_rays and 0 < n_rays <= 6000:
            s, r = self._tablica("src"), self._tablica("rec")
            segs = np.stack([np.repeat(s, len(r), axis=0),
                             np.tile(r, (len(s), 1))], axis=1)
            # Czarne linie na jasnym, nasyconym tle potrzebują większej
            # nieprzezroczystości niż białe na ciemnym; dolny próg pilnuje, żeby
            # przy tysiącach promieni wiązka nie zniknęła zupełnie.
            self.ax.add_collection(LineCollection(
                segs, colors="black", linewidths=0.9,
                alpha=float(min(0.45, max(0.14, 90.0 / n_rays)))))

        # Siatka jako odcinki ograniczone do obszaru badanego. ax.grid() ciągnie
        # linie przez cały kadr, więc po oddaleniu wychodziły daleko poza teren
        # i wyglądało to, jakby siatka była większa niż badany obszar.
        if px_kratki > 3:                    # gęściej zlewa się w szarą plamę
            self.ax.add_collection(LineCollection(
                [[(x, g.ymin), (x, g.ymax)]
                 for x in np.linspace(g.xmin, g.xmax, g.nx + 1)] +
                [[(g.xmin, y), (g.xmax, y)]
                 for y in np.linspace(g.ymin, g.ymax, g.ny + 1)],
                colors="#33414f", alpha=0.30, linewidths=0.5))

        # Liczby w kratkach — tylko dla kratek widocznych w kadrze. Poza
        # oszczędnością (to najdroższa część rysowania) daje to sensowne
        # zachowanie przy zoomie: na gęstej siatce liczby pojawiają się, gdy
        # przybliżysz na tyle, że jest je gdzie zmieścić.
        n_vis = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        if self.show_nums and px_kratki > 26 and n_vis <= 900:
            # Kolory napisów liczymy dla całego widocznego bloku naraz. Barwa
            # bierze się z `baza` (to ona koloruje kratkę), a nie z wypisywanej
            # liczby — ale pytanie mapy barw o każdą kratkę z osobna to było
            # dziewięćset wywołań matplotliba na każde rysowanie.
            blok = np.asarray(baza)[iy0:iy1 + 1, ix0:ix1 + 1]
            with np.errstate(invalid="ignore"):
                rgba = im.cmap(im.norm(np.ma.masked_invalid(blok)))
            jasnosc = (0.299 * rgba[..., 0] + 0.587 * rgba[..., 1]
                       + 0.114 * rgba[..., 2])
            napisy = np.where(jasnosc > 0.6, "#000000", "#ffffff")
            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    v = f[iy, ix]
                    # Warunek na hits, nie na v: dla metryk kątowych zero jest
                    # wynikiem, a nie brakiem danych — luka 0° to kratka widziana
                    # ze wszystkich stron i akurat ją najbardziej chce się widzieć.
                    if not np.isfinite(v) or (self.mask_zero and cov.hits[iy, ix] == 0):
                        continue
                    # ułamki dziesiętne tylko tam, gdzie coś znaczą: waga i
                    # anizotropia mieszczą się w 0–1, reszta to sztuki, metry
                    # albo stopnie i po przecinku miałaby same zera
                    txt = (f"{v:.2f}" if self.metric in (2, self.ANIZO,
                                                        self.POKRYCIE)
                           else f"{v:.0f}")
                    self.ax.text(g.xmin + (ix + .5) * g.dx, g.ymin + (iy + .5) * g.dy,
                                 txt, ha="center", va="center", size=6.5,
                                 clip_on=True,   # Text domyślnie NIE jest
                                 color=str(napisy[iy - iy0, ix - ix0]))

        # Przyrządy. clip_on=True jest tu konieczne: po przybliżeniu markery
        # spoza kadru rysowałyby się na panelu obok mapy. Kosztem jest połówka
        # gwiazdki na samej krawędzi obszaru — mniejsze zło.
        if self.src:
            s = self._tablica("src")
            self.ax.plot(s[:, 0], s[:, 1], "*", color=SHOT, ms=11,
                         mec=INK, mew=0.7, label="PS")
        if self.rec:
            r = self._tablica("rec")
            self.ax.plot(r[:, 0], r[:, 1], "v", color=GEO, ms=7,
                         mec=INK, mew=0.7, label="czujniki")

        # obwódka celownika — ax.cla() kasuje artystów, więc tworzymy ją tutaj
        self.hl, = self.ax.plot([], [], "o", mfc="none", mec=WARN, ms=17,
                                mew=1.6, zorder=6, animated=True)
        self._show_picked()

        self.ax.set_xlim(*view[0])
        self.ax.set_ylim(*view[1])
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        if self.src or self.rec:
            # nieco wyżej, żeby nie wchodzić na wykładnik osi (+5.713e6),
            # który matplotlib wypisuje przy współrzędnych terenowych
            self.ax.legend(loc="lower left", bbox_to_anchor=(0, 1.035, 1, 0.08),
                           mode="expand", ncol=2, fontsize=8, frameon=False,
                           labelcolor=PAPER, handletextpad=0.4)

        s = cov.stats()
        bad = s["bez_pokrycia"]
        self.txt_stats.set_text(
            f"promieni          {s['promieni']:>8}\n"
            f"komórek           {g.nx:>3} × {g.ny:<3} = {s['komórek']}\n"
            f"bez pokrycia      {bad:>8}  ({100 * bad / s['komórek']:.0f}%)")
        self.txt_stats.set_color(WARN if bad else PAPER)

        self._fit_colorbar()
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # x0/y0 przesuwają zero układu (np. na współrzędne terenowe)
    app = App(width=100, height=200, nx=10, ny=20, x0=0.0, y0=0.0)
    plt.show()