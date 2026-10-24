# BioGrafix

Aplikacja do przetwarzania obrazów biometrycznych i grafiki komputerowej: binaryzacja, filtry,
morfologia, histogramy, modele kolorów i figury wektorowe, wszystko na jednym płótnie.

🇬🇧 [English version](README.md)

![BioGrafix](docs/screenshot.png)

## Możliwości

- Obrazy i figury na jednym płótnie, wyniki obok siebie.
- Pasek algorytmów: kafelek = algorytm, parametry w małym oknie.
- Cofanie / ponawianie z pełną historią kroków.
- Liczenie w tle z licznikiem czasu i możliwością przerwania.
- Obszar obrazu: operacja działa tylko w zaznaczonym fragmencie.
- Rysowanie: linie, prostokąty i okręgi (Bresenham), wielokąty, krzywe Béziera.
- Zakładka **Więcej**: metody z danego zadania w jednym miejscu.
- Interfejs po polsku i po angielsku (Widok → Język).

## Algorytmy

| Kategoria | Co zawiera |
|---|---|
| Operacje punktowe | dodawanie / mnożenie / dzielenie, jasność, Levels, rozjaśnij / przyciemnij, 5 metod skali szarości, wypełnij kolorem |
| Histogram | histogram jasności i RGB, rozciąganie i wyrównywanie (jasność / każdy kanał) |
| Filtry | uśredniający, Gauss 5×5, medianowy, Sobel (moduł), wyostrzający, własna maska, maski gotowe (Prewitt, Sobel, Laplace, narożniki) |
| Binaryzacja | ręczna, Otsu, Percent Black, Mean Iterative Selection, entropia (Kapur), **Niblack, Sauvola, Phansalkar** |
| Morfologia | dylatacja, erozja, otwarcie, domknięcie, hit-or-miss (ścienianie / pogrubianie) |
| Segmentacja | usuwanie tła (rembg) |
| Analiza | procent pokrycia kolorem (HSV) |
| Geometria | przesunięcie, obrót, skalowanie (macierze jednorodne), podniesienie stopnia krzywej Béziera |
| Narzędzia | konwerter RGB ↔ CMYK, kostka RGB 3D, stożek HSV 3D |

Pliki: PNG, JPEG (z wyborem jakości), BMP, TIFF, PPM (P3/P6). Całą scenę można zapisać jako JSON.

## Uruchomienie

Python 3.12 lub nowszy.

**Windows (PowerShell)**
```powershell
python -m venv .venv-win
.venv-win\Scripts\python.exe -m pip install -r requirements.txt
.venv-win\Scripts\python.exe main.py
```

**Linux / WSL**
```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install -r requirements.txt
python main.py
```

Każdy system potrzebuje osobnego środowiska: venv z Windowsa nie działa w WSL i odwrotnie.
Obrazy można podać od razu: `python main.py odcisk1.png odcisk2.png`.

**Wczytaj demo** (Ctrl+Shift+D) otwiera przykładowy portret i syntetyczny odcisk palca.

## Testy

```bash
python -m unittest discover -s tests -t .
```

## Struktura

```
main.py              start aplikacji
biografix/
  algorithms/        czyste algorytmy, bez GUI; jeden plik = jeden algorytm
                     (np. algorithms/thresholding/otsu.py), wspólne pomocnicze w _common.py
  operations/        rejestracja algorytmów jako kafelki i pozycje menu
  core/              rejestr, parametry, dokument + historia, liczenie w tle
  scene/             obiekty sceny, zapis / odczyt JSON
  io/                wczytywanie i zapis obrazów (w tym PPM)
  resources/         obrazy przykładowe
  ui/                okno główne, pasek algorytmów, motywy, płótno, panele, okna narzędzi
docs/                zrzuty ekranu
tests/
```

Algorytm to zwykła funkcja zarejestrowana dekoratorem `@operation`; kafelek, okno parametrów,
liczenie w tle i krok w historii powstają na podstawie tej deklaracji.

## Źródła

Portret demo: zdjęcie [Rachael Crowe](https://unsplash.com/photos/tqar8J3fZ3o) (CC0), z dodanym tonowaniem kolorów.
Odcisk palca demo jest wygenerowany syntetycznie.
