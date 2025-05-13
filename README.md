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

Program je sestavován pomocí nástroje CMake. K sestavení programu jsou zapotřebí:

* Kompilátor podporující minimálně standard C++17 (např. GCC, Clang, MSVC).
* Nástroj CMake (verze 3.14 nebo vyšší).
* Pro sestavení s podporou GPU enginů (TensorRT, ONNX Runtime CUDA, LibTorch GPU) je nutná instalace NVIDIA CUDA Toolkit a odpovídající verze cuDNN.

Sestavení programu probíhá ve dvou fázích: konfigurace a sestavení.

```bash
# V kořenovém adresáři projektu (Aplikace_a_knihovny)
mkdir build
cd build

# Fáze konfigurace - příklad s výchozími nastaveními
# Pro Windows s Visual Studio (přizpůsobte verzi):
# cmake .. -G "Visual Studio 17 2022"
# Pro Linux/macOS s GCC/Clang:
# cmake ..

# Fáze sestavení
# cmake --build . --config Release # Pro Visual Studio
# cmake --build . # Pro Makefiles/Ninja

## Komplilace pro Frugally Deep 
Ke zprovoznění knihovny Frugally Deep není zapotřebí žádné dodatečné instalace. Všechny potřebné knihovny, jako například frugally-deep, eigen, FunctionalPlus a json jsou již zahrnuty v řešení ve složce libs.

Program je nadále sestaven pomocí nástroje CMake s argumentem ``-DACCELERATOR=1``.
```

## Kompilace pro Onnx-Runtime s CUDA
Pro kompilaci knihovny Onnx-Runtime s CUDA Execution Providerem je zapotřebí nainstalovat CUDA CuDNN, CUDNNDownload , zdrojové soubory Onnx-Runtime OnnxRuntimeSources.

Po stažení všech knihoven lze z adresáře knihovny Onnx-Runtime spustit příkaz:
```
./build.bat
 --config RelWithDebInfo                                  
 --build_shared_lib                                       
 --parallel --use_cuda                                    
 --cuda_version 12.6                                      
  --cuda_home "<path_to_cuda_installation_home>"           
 --cudnn_home "<path_to_cudnn_installation_home>"         
 --skip_tests
```

Nástroj CMake takto sestaví knihovnu Onnx-Runtime pro CPU Execution Provider a CUDA Execution Provider.

Pro sestavení programu je dále nutné zahrnout argument ``-DACCELERATOR=3``, který značí, že kód bude kompilován pro Onnx-Runtime a dále ``-DONNX\_RUNTIME\_ROOT= <path\_to\_onnx\_runtime\_installation>``, který je kořenovým adresářem právě sestavené knihovny.

## Kompilace pro TensorRT
Poslední možností kompilace programu je kompilace se závislostmi na TensorRT. Nejdříve je třeba stáhnout soubor TensorRT  a extrahovat ho ze ZIP do požadovaného adresáře.

Pro sestavení programu je dále nutné zahrnout argument ``-DACCELERATOR=2``, který značí, že kód bude kompilován pro TensorRT a dále ``-DTENSOR\_RT\_ROOT= <path\_to\_tensor\_rt\_installation>``, která signalizuje kořenový adresář nainstalované knihovny.



## Komplilace pro Frugally Deep 
Ke zprovoznění knihovny Frugally Deep není zapotřebí žádné dodatečné instalace. Všechny potřebné knihovny, jako například frugally-deep, eigen, FunctionalPlus a json jsou již zahrnuty v řešení ve složce libs.

Program je nadále sestaven pomocí nástroje CMake s argumentem ``-DACCELERATOR=1``.


## Kompilace pro Onnx-Runtime s CUDA
Pro kompilaci knihovny Onnx-Runtime s CUDA Execution Providerem je zapotřebí nainstalovat CUDA CuDNN, CUDNNDownload , zdrojové soubory Onnx-Runtime OnnxRuntimeSources.

Po stažení všech knihoven lze z adresáře knihovny Onnx-Runtime spustit příkaz:
```
./build.bat
 --config RelWithDebInfo                                  
 --build_shared_lib                                       
 --parallel --use_cuda                                    
 --cuda_version 12.6                                      
  --cuda_home "<path_to_cuda_installation_home>"           
 --cudnn_home "<path_to_cudnn_installation_home>"         
 --skip_tests
```

Nástroj CMake takto sestaví knihovnu Onnx-Runtime pro CPU Execution Provider a CUDA Execution Provider.

Pro sestavení programu je dále nutné zahrnout argument ``-DACCELERATOR=3``, který značí, že kód bude kompilován pro Onnx-Runtime a dále ``-DONNX\_RUNTIME\_ROOT= <path\_to\_onnx\_runtime\_installation>``, který je kořenovým adresářem právě sestavené knihovny.

## Kompilace pro TensorRT
Poslední možností kompilace programu je kompilace se závislostmi na TensorRT. Nejdříve je třeba stáhnout soubor TensorRT  a extrahovat ho ze ZIP do požadovaného adresáře.

Pro sestavení programu je dále nutné zahrnout argument ``-DACCELERATOR=2``, který značí, že kód bude kompilován pro TensorRT a dále ``-DTENSOR\_RT\_ROOT= <path\_to\_tensor\_rt\_installation>``, která signalizuje kořenový adresář nainstalované knihovny.

