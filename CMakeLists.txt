cmake_minimum_required(VERSION 3.23.5)
set(CMAKE_CXX_STANDARD 17)

########### GLOBAL VARIABLES - CAN BE CHANGED
set(OpenCV_SUBDIR "/x64/vc16")

########### ACCELERATOR SECTION
OPTION(ACCELERATE_ONNX_RUNTIME_CUDA "Accelerate inference using Onnx-Runtime with CUDA." OFF)
OPTION(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU "Accelerate inference using Onnx-Runtime with OpenVINO on CPU." OFF)
OPTION(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU "Accelerate inference using Onnx-Runtime with OpenVINO on GPU." OFF)
OPTION(ACCELERATE_TENSOR_RT "Accelerate inference using TensorRT." OFF)
OPTION(ACCELERATE_FRUGALLY_DEEP "Accelerate inference using Frugally Deep." OFF)


SET(${ACCELERATE_ONNX_RUNTIME_CUDA} OFF CACHE BOOL "Accelerate inference using Onnx-Runtime with CUDA.")
SET(${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU} OFF CACHE BOOL "Accelerate inference using Onnx-Runtime with OpenVINO on CPU.")
SET(${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU} OFF CACHE BOOL "Accelerate inference using Onnx-Runtime with OpenVINO on GPU.")
SET(${ACCELERATE_TENSOR_RT} OFF CACHE BOOL "Accelerate inference using TensorRT.")
SET(${ACCELERATE_FRUGALLY_DEEP} OFF CACHE BOOL "Accelerate inference using Frugally Deep.")

if (${ACCELERATE_ONNX_RUNTIME_CUDA})
    message("Program compiled for inferencing upon OnnxRuntime running with CUDAExecutionProvider (GPU).")
endif ()
if (${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU})
    message("Program compiled for inferencing upon OnnxRuntime running with OpenVINOExecutionProvider (CPU).")
endif ()
if (${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU})
    message("Program compiled for inferencing upon OnnxRuntime running with OpenVINOExecutionProvider (GPU).")
endif ()
if (${ACCELERATE_TENSOR_RT})
    message("Program compiled for inferencing upon TensorRT (GPU).")
endif ()
if (${ACCELERATE_FRUGALLY_DEEP})
    message("Program compiled for inferencing upon FrugallyDeep (CPU).")
endif ()

########### METADATA
set(PROJECT_NAME InferenceTest)
set(SOURCES
        src/FileUtils.cpp
        src/ImageUtils.cpp
        src/TensorRTInference.cpp
        src/include/Model.h
        src/include/MathUtils.h
        src/include/Logger.h
        src/include/ImageUtils.h
        src/include/FileUtils.h
        src/include/AbstractInference.h
        src/include/TensorRTInference.h
        src/include/FrugallyDeepInference.h
        src/FrugallyDeepInference.cpp
        test/TestMain.cpp
        test/TestUtils.h
        src/StringUtils.cpp
        src/include/StringUtils.h
        src/include/OnnxAbstractInference.h
        src/OnnxCudaInference.cpp
        src/include/OnnxCudaInference.h
        src/OnnxAbstractInference.cpp
        src/AbstractInference.cpp
        src/OnnxOpenVinoCpuInference.cpp
        src/OnnxOpenVinoGpuInference.cpp
)
########### ADD EXECUTABLES
project(${PROJECT_NAME} LANGUAGES CXX)
add_executable(${PROJECT_NAME} ${SOURCES})

########### DEFINE FUNCTION FOR DLL COPIES
function(copy_dlls target_name dlls_root)
    if (MSVC)
        file(GLOB DLLS ${dlls_root}/*.dll)
        foreach (DLL ${DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${DLL} $<TARGET_FILE_DIR:${target_name}>)
        endforeach ()
    endif ()
endfunction()

########### LINK OPEN_CV
message("Searching for OpenCV.")
find_package(OpenCV REQUIRED)
message("OpenCV successfully found.")

copy_dlls(${PROJECT_NAME} "${OpenCV_DIR}${OpenCV_SUBDIR}/bin")
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
message("OpenCV successfully linked.")


########### HANDLE ACCELERATORS
IF (${ACCELERATE_TENSOR_RT})
    ########### LINK TENSOR RT FOR INFERENCING
    message("Searching for CUDAToolkit.")
    find_package(CUDAToolkit REQUIRED)
    message("CUDAToolkit successfully found.")

    list(APPEND PLUGINS "nvinfer_10")
    list(APPEND PLUGINS "nvinfer_plugin_10")
    list(APPEND PLUGINS "nvonnxparser_10")

    if (NOT DEFINED TENSOR_RT_ROOT)
        message(SEND_ERROR "\"TENSOR_RT_ROOT\" is required")
        return()
    endif ()
    message("TensorRT root found at \"${TENSOR_RT_ROOT}\".")
    message("Searching for TensorRT libraries.")


    copy_dlls(${PROJECT_NAME} "${TENSOR_RT_ROOT}/lib")
    foreach (libName ${PLUGINS})
        find_library(${libName}_lib HINTS ${TENSOR_RT_ROOT} NAMES ${libName} PATH_SUFFIXES lib REQUIRED)
        message(" ------ ${libName} (TensorRT part) successfully found.")
        list(APPEND PLUGIN_LIBS "${${libName}_lib}")
    endforeach ()
    message("TensorRT libraries successfully found.")

    target_link_libraries(${PROJECT_NAME} CUDA::cudart ${PLUGIN_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${TENSOR_RT_ROOT}/include src/include)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ACCELERATE_TENSOR_RT=${ACCELERATE_TENSOR_RT})
ELSEIF (${ACCELERATE_FRUGALLY_DEEP})
    ########### LINK FRUGALLY DEEP FOR INFERENCING
    target_include_directories(${PROJECT_NAME}
            PRIVATE
            libs/FunctionalPlus/include
            libs/json/single_include
            libs/eigen
            libs/frugally-deep/include
            src/include
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE ACCELERATE_FRUGALLY_DEEP=${ACCELERATE_FRUGALLY_DEEP})
ELSEIF (${ACCELERATE_ONNX_RUNTIME_CUDA} OR ${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU})
    ########### LINK ONNX RUNTIME WITH CUDA FOR INFERENCING
    if (${ACCELERATE_ONNX_RUNTIME_CUDA})
        list(APPEND PLUGINS "onnxruntime_providers_cuda")
        message("Searching for CUDAToolkit.")
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${PROJECT_NAME} CUDA::cudart)
        message("CUDAToolkit successfully found.")
    endif ()
    if (${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU} OR ${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU})
        ### TODO Add
        ### list(APPEND PLUGINS "onnxruntime_providers_openvino")
    endif ()
    if (NOT DEFINED ONNX_RUNTIME_ROOT)
        message(SEND_ERROR "\"ONNX_RUNTIME_ROOT\" is required")
        return()
    endif ()
    message("Onnx root found at \"${ONNX_RUNTIME_ROOT}\".")

    list(APPEND PLUGINS "onnxruntime")
    list(APPEND PLUGINS "test_execution_provider")
    list(APPEND PLUGINS "onnxruntime_providers_shared")
    list(APPEND PLUGINS "custom_op_local_function")
    list(APPEND PLUGINS "custom_op_library")
    list(APPEND PLUGINS "custom_op_invalid_library")
    list(APPEND PLUGINS "custom_op_get_const_input_test_library")
    message("Searching for Onnx Runtime libraries.")

    foreach (libName ${PLUGINS})
        add_custom_command(
                TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${ONNX_RUNTIME_ROOT}/build/Windows/RelWithDebInfo/RelWithDebInfo/${libName}.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>)
        find_library(${libName}_lib
                NAMES ${libName}
                HINTS ${ONNX_RUNTIME_ROOT}/build/Windows/RelWithDebInfo/RelWithDebInfo
                PATH_SUFFIXES lib
                REQUIRED
        )
        message(" ------ ${libName} (Onnx Runtime part) successfully found.")
        list(APPEND PLUGIN_LIBS "${${libName}_lib}")
    endforeach ()
    message("Onnx Runtime libraries successfully found.")
    target_link_libraries(${PROJECT_NAME} ${PLUGIN_LIBS})
    target_include_directories(${PROJECT_NAME}
            PRIVATE
            ${ONNX_RUNTIME_ROOT}/include
            ${ONNX_RUNTIME_ROOT}/include/onnxruntime/core/session
            src/include
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
            ACCELERATE_ONNX_RUNTIME_CUDA=${ACCELERATE_ONNX_RUNTIME_CUDA}
            ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU=${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU}
            ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU=${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU}
    )
ENDIF ()