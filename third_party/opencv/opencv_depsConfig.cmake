cmake_minimum_required(VERSION 3.10)

message("--- OpenCV Searching Start ---")


# Create interface library to consolidate all dependencies
add_library(opencv_deps INTERFACE)

# Path suffix for Windows distribution of OpenCV
set(OpenCV_WINDOWS_SUFFIX "build")
# Hints for OpenCV location
set(OpenCV_HINTS
        "C:/Libs/opencv/${OpenCV_WINDOWS_SUFFIX}"
        "C:/Program Files/opencv/${OpenCV_WINDOWS_SUFFIX}"
)


find_package(OpenCV HINTS ${OpenCV_HINTS} REQUIRED)
target_link_libraries(opencv_deps INTERFACE ${OpenCV_LIBS})

message("--- Frugally Deep Searching End ---")
