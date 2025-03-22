cmake_minimum_required(VERSION 3.10)

message("--- Frugally Deep Searching Start ---")

# Create interface library to consolidate all dependencies
add_library(frugally_deep_deps INTERFACE IMPORTED)

# Link all required libraries
target_include_directories(frugally_deep_deps
        INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/FunctionalPlus/include
        ${CMAKE_CURRENT_LIST_DIR}/json/single_include
        ${CMAKE_CURRENT_LIST_DIR}/eigen
        ${CMAKE_CURRENT_LIST_DIR}/frugally-deep/include
)

message("--- Frugally Deep Searching End ---")