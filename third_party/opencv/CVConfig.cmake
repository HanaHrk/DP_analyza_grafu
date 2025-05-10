cmake_minimum_required(VERSION 3.10)

message("--- OpenCV Searching Start ---")


# Create interface library to consolidate all dependencies
add_library(CV INTERFACE)

if (DEFINED CV_HOME)
    set(OpenCV_DIR ${CV_HOME})
endif ()
find_package(OpenCV REQUIRED)
target_link_libraries(CV INTERFACE ${OpenCV_LIBS})

message("--- OpenCV Searching End ---")
