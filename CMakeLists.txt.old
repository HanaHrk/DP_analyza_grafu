cmake_minimum_required(VERSION 3.23.5)
set(CMAKE_CXX_STANDARD 17)

set(OpenVINO_DIR "C:/Users/hanah/Downloads/openvino_2024.6.0")

########### GLOBAL VARIABLES - CAN BE CHANGED
set(OpenCV_SUBDIR "/x64/vc16")

########### ACCELERATOR SECTION
OPTION(ACCELERATE_ONNX_RUNTIME_CUDA "Accelerate inference using Onnx-Runtime with CUDA." OFF)
OPTION(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU "Accelerate inference using Onnx-Runtime with OpenVINO on CPU." OFF)
OPTION(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU "Accelerate inference using Onnx-Runtime with OpenVINO on GPU." OFF)
OPTION(ACCELERATE_TENSOR_RT "Accelerate inference using TensorRT." OFF)
OPTION(ACCELERATE_FRUGALLY_DEEP "Accelerate inference using Frugally Deep." OFF)
OPTION(PRINT_STATISTICS "Print thread statistics of program." OFF)

SET(${ACCELERATE_ONNX_RUNTIME_CUDA} OFF CACHE BOOL "Accelerate inference using Onnx-Runtime with CUDA.")
SET(${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU} OFF CACHE BOOL "Accelerate inference using Onnx-Runtime with OpenVINO on CPU.")
SET(${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU} OFF CACHE BOOL "Accelerate inference using Onnx-Runtime with OpenVINO on GPU.")
SET(${ACCELERATE_TENSOR_RT} OFF CACHE BOOL "Accelerate inference using TensorRT.")
SET(${ACCELERATE_FRUGALLY_DEEP} OFF CACHE BOOL "Accelerate inference using Frugally Deep.")
SET(${PRINT_STATISTICS} OFF CACHE BOOL "Print thread statistics of program.")

if (${ACCELERATE_ONNX_RUNTIME_CUDA})
    list(APPEND COMPILE_OPTIONS ACCELERATE_ONNX_RUNTIME_CUDA)
    message("Program compiled for inference upon OnnxRuntime running with CUDAExecutionProvider (GPU).")
endif ()
if (${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU})
    list(APPEND COMPILE_OPTIONS ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU)
    message("Program compiled for inference upon OnnxRuntime running with OpenVINOExecutionProvider (CPU).")
endif ()
if (${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU})
    list(APPEND COMPILE_OPTIONS ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU)
    message("Program compiled for inference upon OnnxRuntime running with OpenVINOExecutionProvider (GPU).")
endif ()
if (${ACCELERATE_TENSOR_RT})
    list(APPEND COMPILE_OPTIONS ACCELERATE_TENSOR_RT)
    message("Program compiled for inference upon TensorRT (GPU).")
endif ()
if (${ACCELERATE_FRUGALLY_DEEP})
    list(APPEND COMPILE_OPTIONS ACCELERATE_FRUGALLY_DEEP)
    message("Program compiled for inference upon FrugallyDeep (CPU).")
endif ()
if (${PRINT_STATISTICS})
    list(APPEND COMPILE_OPTIONS PRINT_STATISTICS)
    message("Program will print statistics information.")
endif ()

########### METADATA
set(PROJECT_NAME InferenceTest)
set(SOURCES
        src/FileUtils.cpp
        src/include/FileUtils.h
        src/ImageUtils.cpp
        src/TensorRTInference.cpp
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
        src/include/OnnxOpenVinoGpuInference.h
        src/include/ThreadUtils.h
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
    target_compile_definitions(${PROJECT_NAME} PRIVATE ACCELERATE_TENSOR_RT)
ENDIF () # TENSOR_RT
IF (${ACCELERATE_FRUGALLY_DEEP})
    ########### LINK FRUGALLY DEEP FOR INFERENCING
    target_include_directories(${PROJECT_NAME}
            PRIVATE
            libs/FunctionalPlus/include
            libs/json/single_include
            libs/eigen
            libs/frugally-deep/include
            src/include
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE ACCELERATE_FRUGALLY_DEEP)
ENDIF () # FRUGALLY DEEP

IF (${ACCELERATE_ONNX_RUNTIME_CUDA} OR ${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU} OR ${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU})
    ########### LINK ONNX RUNTIME WITH CUDA FOR INFERENCING
    if (${ACCELERATE_ONNX_RUNTIME_CUDA})
        list(APPEND PLUGINS "onnxruntime_providers_cuda")
        message("Searching for CUDAToolkit.")
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${PROJECT_NAME} CUDA::cudart)
        message("CUDAToolkit successfully found.")
    endif ()
    if (${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU} OR ${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU})
        find_package(Threads REQUIRED)
        add_definitions(-DUSE_OPENVINO)
        find_package(OpenVINO REQUIRED COMPONENTS Runtime HINTS "${OpenVINO_DIR}/runtime/cmake")

        list(APPEND PLUGINS "onnxruntime_providers_openvino")
        list(APPEND PLUGINS "custom_op_openvino_wrapper_library")
        if (NOT DEFINED OpenVINO_DIR)
            message(SEND_ERROR "\"OpenVINO_DIR\" is required")
            return()
        endif ()

        if (${CMAKE_BUILD_TYPE} MATCHES "Release")
            list(APPEND OPENVINO_LIBS "openvino")
            list(APPEND OPENVINO_LIBS "openvino_c")
            list(APPEND OPENVINO_LIBS "openvino_onnx_frontend")
            list(APPEND OPENVINO_LIBS "openvino_paddle_frontend")
            list(APPEND OPENVINO_LIBS "openvino_pytorch_frontend")
            list(APPEND OPENVINO_LIBS "openvino_tensorflow_frontend")
            list(APPEND OPENVINO_LIBS "openvino_tensorflow_lite_frontend")

            list(APPEND OPENVINO_DLLS "openvino_auto_batch_plugin")
            list(APPEND OPENVINO_DLLS "openvino_auto_plugin")
            list(APPEND OPENVINO_DLLS "openvino_c")
            list(APPEND OPENVINO_DLLS "openvino_hetero_plugin")
            list(APPEND OPENVINO_DLLS "openvino_intel_cpu_plugin")
            list(APPEND OPENVINO_DLLS "openvino_intel_gpu_plugin")
            list(APPEND OPENVINO_DLLS "openvino_intel_npu_plugin")
            list(APPEND OPENVINO_DLLS "openvino_ir_frontend")
            list(APPEND OPENVINO_DLLS "openvino_onnx_frontend")
            list(APPEND OPENVINO_DLLS "openvino_paddle_frontend")
            list(APPEND OPENVINO_DLLS "openvino_pytorch_frontend")
            list(APPEND OPENVINO_DLLS "openvino_tensorflow_frontend")
            list(APPEND OPENVINO_DLLS "openvino_tensorflow_lite_frontend")
        elseif (${CMAKE_BUILD_TYPE} MATCHES "Debug")
            list(APPEND OPENVINO_LIBS "openvinod")
            list(APPEND OPENVINO_LIBS "openvino_cd")
            list(APPEND OPENVINO_LIBS "openvino_onnx_frontendd")
            list(APPEND OPENVINO_LIBS "openvino_paddle_frontendd")
            list(APPEND OPENVINO_LIBS "openvino_pytorch_frontendd")
            list(APPEND OPENVINO_LIBS "openvino_tensorflow_frontendd")
            list(APPEND OPENVINO_LIBS "openvino_tensorflow_lite_frontendd")

            list(APPEND OPENVINO_DLLS "openvinod")
            list(APPEND OPENVINO_DLLS "openvino_auto_batch_plugind")
            list(APPEND OPENVINO_DLLS "openvino_auto_plugind")
            list(APPEND OPENVINO_DLLS "openvino_cd")
            list(APPEND OPENVINO_DLLS "openvino_hetero_plugind")
            list(APPEND OPENVINO_DLLS "openvino_intel_cpu_plugind")
            list(APPEND OPENVINO_DLLS "openvino_intel_gpu_plugind")
            list(APPEND OPENVINO_DLLS "openvino_intel_npu_plugind")
            list(APPEND OPENVINO_DLLS "openvino_ir_frontendd")
            list(APPEND OPENVINO_DLLS "openvino_onnx_frontendd")
            list(APPEND OPENVINO_DLLS "openvino_paddle_frontendd")
            list(APPEND OPENVINO_DLLS "openvino_pytorch_frontendd")
            list(APPEND OPENVINO_DLLS "openvino_tensorflow_frontendd")
            list(APPEND OPENVINO_DLLS "openvino_tensorflow_lite_frontendd")
        endif ()
        foreach (libName ${OPENVINO_LIBS})
            find_library(${libName}_lib HINTS ${OpenVINO_DIR}/../lib/intel64/${CMAKE_BUILD_TYPE} NAMES ${libName} PATH_SUFFIXES lib REQUIRED)
            message(" ------ ${libName} (OpenVINO part) successfully found.")
            list(APPEND PLUGIN_LIBS "${${libName}_lib}")
        endforeach ()
        foreach (libName ${OPENVINO_LIBS})
            add_custom_command(
                    TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                    ${OpenVINO_DIR}/../bin/intel64/${CMAKE_BUILD_TYPE}/${libName}.dll
                    $<TARGET_FILE_DIR:${PROJECT_NAME}>)
            message(" ------ ${libName} (OpenVINO Runtime part) successfully found.")
        endforeach ()
        target_link_libraries(${PROJECT_NAME} ${PLUGIN_LIBS} Threads::Threads openvino::runtime)
    endif ()

    if (${ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU})
        find_package(OpenCL REQUIRED)
        target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
        message("OpenCL library found.")
    endif ()
    if (NOT DEFINED OnnxRuntime_DIR)
        message(SEND_ERROR "\"OnnxRuntime_DIR\" is required")
        return()
    endif ()
    message("Onnx root found at \"${OnnxRuntime_DIR}\".")

    list(APPEND PLUGINS "onnxruntime")
    list(APPEND PLUGINS "onnxruntime_providers_shared")
    list(APPEND PLUGINS "test_execution_provider")
    list(APPEND PLUGINS "custom_op_local_function")
    list(APPEND PLUGINS "custom_op_library")
    list(APPEND PLUGINS "custom_op_invalid_library")
    list(APPEND PLUGINS "custom_op_get_const_input_test_library")
    message("Searching for Onnx Runtime libraries.")

    file(GLOB LIBS "${OnnxRuntime_DIR}/build/Windows/RelWithDebInfo/RelWithDebInfo/*.lib")
    file(GLOB DLLS "${OnnxRuntime_DIR}/build/Windows/RelWithDebInfo/RelWithDebInfo/*.dll")
    foreach (libName ${LIBS})
        get_filename_component(LIB ${libName} NAME_WE)
        find_library(${libName}_lib
                NAMES ${LIB}
                HINTS ${OnnxRuntime_DIR}/build/Windows/RelWithDebInfo/RelWithDebInfo
                PATH_SUFFIXES lib
                REQUIRED
        )
        message(" ------ ${libName} (Onnx Runtime part) successfully found.")
        list(APPEND PLUGIN_LIBS "${${libName}_lib}")
    endforeach ()

        foreach (libName ${DLLS})
        add_custom_command(
                TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${libName}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endforeach ()
    message("Onnx Runtime libraries successfully found.")
    target_link_libraries(${PROJECT_NAME} ${PLUGIN_LIBS})
    target_include_directories(${PROJECT_NAME}
            PRIVATE
            ${OnnxRuntime_DIR}/include
            ${OnnxRuntime_DIR}/include/onnxruntime/core/session
            ${OpenVINO_DIR}/../include
            src/include
    )
ENDIF () # ONNX RUNTIME

target_compile_definitions(${PROJECT_NAME} PRIVATE
        ${COMPILE_OPTIONS}
)