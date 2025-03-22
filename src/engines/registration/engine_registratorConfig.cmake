cmake_minimum_required(VERSION 3.10)

message("--- Engine Registrator Start ---")

# Create engine library
add_library(engine_registrator STATIC
        ${CMAKE_CURRENT_LIST_DIR}/EngineFactory.cpp
)

# Add Abstract Engine dependencies
find_package(abstract_engine HINTS ${CMAKE_SOURCE_DIR}/src/engines/abstract REQUIRED)

# Add subdirectories conditionally
if (USE_FRUGALLY_DEEP)
    # Add frugally_deep engine
    find_package(frugally_deep_engine HINTS ${CMAKE_SOURCE_DIR}/src/engines/frugally_deep REQUIRED)
    list(APPEND COMPILE_OPTIONS -DUSE_FRUGALLY_DEEP=1)
endif ()

target_compile_definitions(engine_registrator PRIVATE ${COMPILE_OPTIONS})
target_include_directories(engine_registrator PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(engine_registrator PUBLIC opencv_deps frugally_deep_engine abstract_engine)

message("--- Engine Registrator End ---")