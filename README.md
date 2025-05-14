# Uživatelská příručka / Readme

Tento dokument slouží jako uživatelská příručka k softwarové části diplomové práce, zaměřené na benchmarking a inference neuronových sítí s podporou různých backendů v C++. Popisuje strukturu projektu, předpoklady pro sestavení programu a návod na jeho spuštění a použití.


## Struktura projektu

Odevzdávaná struktura diplomové práce má standardní uspořádání. Softwarová část je umístěna v adresáři `Aplikace_a_knihovny`.

Hlavní adresáře odevzdávané struktury:

* `Text_prace`: Obsahuje veškeré "zdrojové" soubory a finální soubor práce ve formátu PDF.
* `Poster`: Obsahuje soubory posteru (ve formátu .pub a .pdf).
* `Aplikace_a_knihovny`: **Obsahuje zdrojové kódy, sestavené binární soubory, potřebné knihovny třetích stran, dokumentaci a návod k sestavení a spuštění.** Tento adresář je popsán detailněji níže.
* `Vstupni_data`: Obsahuje ukázku dat z sady ICPR CHART-Infographics 2022
* `Vysledky`: Obsahuje výstup naměřených metrik z Nsight Systems, matici záměn 
* `Readme.txt`: Tento soubor s popisem struktury a návodem k použití.

Detailní struktura adresáře `Aplikace_a_knihovny`:

* `src`: Obsahuje hlavní zdrojové kódy projektu, implementaci modulárního frameworku a enginů.
    * `src/engines`: Implementace jednotlivých inference enginů a mechanismu registrace.
        * `src/engines/abstract`: Abstraktní rozhraní pro inference enginy.
        * `src/engines/frugally_deep`: Implementace enginu pro Frugally-Deep.
        * `src/engines/libtorch`: Implementace enginu pro LibTorch.
        * `src/engines/onnxruntime`: Implementace enginu pro ONNX Runtime (CPU/CUDA).
        * `src/engines/tensorrt`: Implementace enginu pro TensorRT.
        * `src/engines/registration`: Mechanismus registrace enginů (EngineFactory).
    * `src/inferencetools`: Hlavičkové soubory definující datové struktury a rozhraní frameworku.
    * ... další zdrojové soubory ...
* `third_party`: Obsahuje soubory knihoven třetích stran buď přímo (např. Frugally-Deep, Eigen, JSON, OpenCV) nebo konfigurační skripty pro nalezení externě instalovaných knihoven (CUDA, TensorRT, LibTorch, ONNX Runtime).
    * `third_party/cuda`: Konfigurace pro nalezení CUDA.
    * `third_party/frugally_deep`: Soubory knihovny Frugally-Deep a jejích závislostí.
    * `third_party/libtorch`: Konfigurace pro nalezení LibTorch.
    * `third_party/onnxruntime`: Konfigurace pro nalezení ONNX Runtime.
    * `third_party/opencv`: Konfigurace pro nalezení OpenCV.
    * `third_party/tensorrt`: Konfigurace pro nalezení TensorRT.
* `benchmarks`: Obsahuje zdrojové kódy hlavní spustitelné aplikace pro benchmarking a spouštění inference (`main.cpp`, `ArgsParser.cpp`, `DataLoader.cpp`, `ClassificationUtils.cpp`, `SegmentationUtils.cpp`).
* `build`: Adresář pro sestavení (generovaný CMake).
* `lib`: Adresář pro zkopírované dynamické knihovny (DLL/so), pokud je vyžadováno (zejména pro MSVC).
* `models`: Doporučený adresář pro umístění natrénovaných modelů (.h5, .onnx, .pt).
* `data`: Doporučený adresář pro umístění vstupních dat (obrázků), typicky rozdělený dle tříd pro klasifikaci.

## Sestavení programu

Pokyny k sestavení programu naleznete [zde](COMPILE.md)

