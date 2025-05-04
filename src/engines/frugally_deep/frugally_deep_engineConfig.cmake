cmake_minimum_required(VERSION 3.10)

message("--- Frugally Deep Engine Start ---")

# Add Frugally Deep dependencies
find_package(frugally_deep_deps HINTS ${CMAKE_SOURCE_DIR}/third_party/frugally_deep REQUIRED)

# Create engine library
add_library(frugally_deep_engine SHARED
        ${CMAKE_CURRENT_LIST_DIR}/FrugallyDeepEngineSequential.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FrugallyDeepEngine.cpp
)
target_include_directories(frugally_deep_engine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(frugally_deep_engine PUBLIC frugally_deep_deps abstract_engine)

message("--- Frugally Deep Engine End ---")