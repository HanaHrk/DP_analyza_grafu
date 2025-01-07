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

