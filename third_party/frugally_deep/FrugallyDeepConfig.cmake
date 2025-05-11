message("----- Frugally Deep Searching Start")

# Create an interface library for Frugally Deep
add_library(FrugallyDeep INTERFACE IMPORTED)

# Set include directories for the FrugallyDeep library
# These directories contain necessary header files for:
# - FunctionalPlus: A functional programming library
# - JSON: JSON for Modern C++
# - Eigen: Linear algebra library
# - frugally-deep: Main library headers
target_include_directories(FrugallyDeep
        INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/FunctionalPlus/include
        ${CMAKE_CURRENT_LIST_DIR}/json/single_include
        ${CMAKE_CURRENT_LIST_DIR}/eigen
        ${CMAKE_CURRENT_LIST_DIR}/frugally-deep/include
)

message("----- Frugally Deep Searching End")
