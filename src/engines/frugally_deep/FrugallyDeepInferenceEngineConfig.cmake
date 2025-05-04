cmake_minimum_required(VERSION 3.10)

message("--- Frugally Deep Engine Start ---")

# Add Frugally Deep dependencies
find_package(FrugallyDeep HINTS ${CMAKE_SOURCE_DIR}/third_party/frugally_deep REQUIRED)

# Create engine library
add_library(FrugallyDeepInferenceEngine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/FrugallyDeepEngineSequential.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FrugallyDeepEngine.cpp
)
target_include_directories(FrugallyDeepInferenceEngine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(FrugallyDeepInferenceEngine PUBLIC FrugallyDeep AbstractInferenceEngine)

message("--- Frugally Deep Engine End ---")