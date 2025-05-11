# Start of Frugally Deep Engine configuration
message("----- Frugally Deep Engine Start")

# Find and include the FrugallyDeep package from the third-party directory
find_package(FrugallyDeep HINTS ${CMAKE_SOURCE_DIR}/third_party/frugally_deep REQUIRED)

# Create a static library for FrugallyDeep inference engine
add_library(FrugallyDeepInferenceEngine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/FrugallyDeepEngineSequential.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FrugallyDeepEngine.cpp
)
# Add include directories for public headers
target_include_directories(FrugallyDeepInferenceEngine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
# Link against required libraries
target_link_libraries(FrugallyDeepInferenceEngine PUBLIC FrugallyDeep AbstractInferenceEngine)

# End of Frugally Deep Engine configuration
message("----- Frugally Deep Engine End")
