# Instalace knihoven

## OpenCV

OpenCV je zapotřebí pro všechny inferenční modely pro práci s obrazovými daty.

## Frugally Deep

Knihovana Frugally Deep je pro Windows i Linux v projektu již integrována a stačí si pouze projekt naklonovat pomocí:

 ```
git clone https://github.com/HanaHrk/DP_analyza_grafu.git --recurse-submodules
```

## CUDA

*CUDA* lze stáhnout z oficiálních
stránek [NVIDIA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network).
Po jejím stažení je nastavit proměnné:

- `CUDA_PATH` - Cesta k domovskému adresáři *CUDA* instalace
- `Path` - Do systémové proměnné Path přidat cesty k
    - `%CUDA_PATH%/libnvvp`
    - `%CUDA_PATH%/bin`

## CuDNN

*CuDNN* lze stáhnout z oficiálních
stránek [NVIDIA](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local).
Po jejím stažení je nutné nastavit systémové proměnné `Path` na hodnotu `%CUDNN_PATH%/bin`.

## LibTorch

K zprovoznění knihovny LibTorch je zapotřebí několik návazných knihoven:

- **CUDA** pro inferenci vstupu na GPU
- **LibTorch** s podporou CUDA processingu
- **NVTX** jako workaround pro rozbité dependence v *LibTorch* knihovně

### Instalace CUDA

Viz. kapitola [CUDA](#cuda)

### Instalace NVTX

Knihovna *NVTX* je automaticky dotažena při klonování repozitáře pomocí:

 ```
git clone https://github.com/HanaHrk/DP_analyza_grafu.git --recurse-submodules
```

### Instalace LibTorch

Preš stránku [LibTorch](https://pytorch.org/get-started/locally/) lze stáhnout distribuci LibTorch pro libovolnou
platformu s libovolnou podporou inference na CUDA nebo CPU.

## TensorRT

- **Cuda** pro inferenci vstupu na GPU
- **CuDNN** pro inferenci vstupu na GPU
- **TensorRT** pro inferenci pomocí frameworku TensorRT
- **OnnxRuntime** jako workaround pro rozbité dependence v *LibTorch* knihovně

### Instalace CUDA

Viz. kapitola [CUDA](#cuda)

### Instalace CuDNN

Viz. kapitola [CuDNN](#cudnn)

### Instalace TensorRT

Z oficiálních stránek [NVIDIA](https://developer.nvidia.com/tensorrt/download/10x).

### Instalce OnnxRuntime

*OnnxRuntime* je potřeba nejdříve naklovat pomocí:

```
git clone https://github.com/microsoft/onnxruntime.git
```

Následně se je potřeba se dostat do naklonovaného adresáře *onnxruntime* a spustit příkaz:

```
build.bat                                           \
--config Release                                    \
--use_cuda                                          \
--use_tensorrt                                      \
--cuda_home "%CUDA_PATH%"                           \
--cudnn_home "%CUDNN_PATH%"                         \
--tensorrt_home "%TENSORRT_PATH%"                   \
--parallel                                          \
--skip_tests                                        \
--build_shared_lib 
```
Sestavení této knihovny obvykle relativně trvá.

