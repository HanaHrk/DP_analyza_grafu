
message("--- LibTorch Engine Start ---")

# Add LibTorch dependencies
find_package(libtorch_deps HINTS ${CMAKE_SOURCE_DIR}/third_party/libtorch REQUIRED)

add_library(libtorch_engine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/LibTorchEngineSequential.cpp
)

target_include_directories(libtorch_engine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(libtorch_engine PUBLIC libtorch_deps abstract_engine)


message("--- LibTorch Engine End ---")
