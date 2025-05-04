
message("--- LibTorch Engine Start ---")

# Add LibTorch dependencies
find_package(LibTorch HINTS ${CMAKE_SOURCE_DIR}/third_party/libtorch REQUIRED)

add_library(LibTorchInferenceEngine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/LibTorchEngineSequential.cpp
)

target_include_directories(LibTorchInferenceEngine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(LibTorchInferenceEngine PUBLIC LibTorch AbstractInferenceEngine)

if (MSVC)
    copy_dlls(LibTorchInferenceEngine ${LibTorch_BINARIES})
endif ()

message("--- LibTorch Engine End ---")
