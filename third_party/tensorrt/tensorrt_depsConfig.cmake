cmake_minimum_required(VERSION 3.27)

message("--- CUDA Searching Start ---")
find_package(CUDAToolkit REQUIRED)
message("--- CUDA Searching End ---")



message("--- TensorRT Searching Start ---")
if (NOT DEFINED TensorRT_HOME)
    message(SEND_ERROR "TensorRT_HOME is required")
    return(1)
endif ()

set(TensorRT_INCLUDE_DIRS "${TensorRT_HOME}/include")
set(TensorRT_LIB_DIRS "${TensorRT_HOME}/lib")

# Copy DLLs if using Windows
if (WIN32)
    function(copy_dlls from to)
        file(GLOB dlls ${from}/*.dll)
        foreach (dll ${dlls})
            file(COPY "${dll}" DESTINATION ${to})
        endforeach ()
    endfunction()
    copy_dlls("${TensorRT_LIB_DIRS}" "${CMAKE_CURRENT_BINARY_DIR}")
endif ()

# Define all required TensorRT libraries
set(TENSORRT_LIBRARIES_NAMES
        "nvinfer_10"
        "nvinfer_plugin_10"
        "nvinfer_vc_plugin_10"
        "nvinfer_lean_10"
        "nvinfer_dispatch_10"
        "nvonnxparser_10"
)
# Link all required TensorRT libraries
foreach (tensorRTLib ${TENSORRT_LIBRARIES_NAMES})
    find_library(${tensorRTLib}_lib HINTS "${TensorRT_LIB_DIRS}" NAMES ${tensorRTLib} PATH_SUFFIXES lib REQUIRED)
    message(" ------ ${tensorRTLib} successfully found.")
    list(APPEND TENSORRT_LIBRARIES "${${tensorRTLib}_lib}")
endforeach ()

add_library(tensorrt_deps INTERFACE)
target_link_libraries(tensorrt_deps INTERFACE CUDA::cudart ${TENSORRT_LIBRARIES} opencv_deps)
target_include_directories(tensorrt_deps INTERFACE ${TensorRT_HOME}/include ${CMAKE_CURRENT_LIST_DIR}/include)


message("--- TensorRT Searching End ---")