message("----- OpenCV Searching Start")

# Create an interface library for OpenCV
add_library(CV INTERFACE)

# If CV_HOME is defined, use it as the OpenCV directory
if (DEFINED CV_HOME)
    set(OpenCV_DIR ${CV_HOME})
endif ()

# Find OpenCV package (required)
find_package(OpenCV REQUIRED)

# Link OpenCV libraries to the CV interface
target_link_libraries(CV INTERFACE ${OpenCV_LIBS})

message("----- OpenCV Searching End")