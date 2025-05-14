# Kompilace projektu

Program lze kompilovat pro inferenční enginy **Frugally Deep**, **LibTorch** a **TensorRT**.
Pro informaci jak nainstalovat všechny požadované knihovny pro daný engine je popis [zde](INSTALL.md).

## Frugally Deep

[Insalace](INSTALL.md#frugally-deep)

Pro kompilaci inferenčního enginu s využitím knihovny Frugally Deep stačí pouze zahrnout *CMake* argument
`-DUSE_FRUGALLY_DEEP=1`.

## TensorRT

[Insalace](INSTALL.md#tensorrt)

Pro kompilování inferenčního enginu pomocí TensorRT je nutné zahrnout *CMake* argument `-DUSE_TENSOR_RT=1`.
Dalším povinným argumentem je cesta k *OnnxRuntime* sestaveného ze zdrojových souborů jako `-DOnnxRuntime_HOME=%VALUE%`.
Posledním povinným argumentem je cesta k *TensorRT* jako `-DTensor_HOME=%VALUE%`.

## LibTorch

[Instalace](INSTALL.md#libtorch)

Pro kompilování inferenčního enginu s využitím knihovny LibTorch je nutné zahrnout *CMake* argument `-DUSE_LIBTORCH=1`.
Dalším jediným povinným *CMake* parametrem je cesta k instalaci LibTorch jako `-DLibTorch_HOME=%VALUE%`.