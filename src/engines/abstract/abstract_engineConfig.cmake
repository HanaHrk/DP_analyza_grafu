cmake_minimum_required(VERSION 3.10)

message("--- Abstract Engine Start ---")

# Create engine library
add_library(abstract_engine INTERFACE)
target_link_libraries(abstract_engine INTERFACE opencv_deps)

# Add include files
target_include_directories(abstract_engine INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

message("--- Abstract Engine End ---")
