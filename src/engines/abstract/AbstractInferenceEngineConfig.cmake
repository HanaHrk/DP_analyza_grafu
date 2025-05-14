message("----- Abstract Engine Start")

# Function to copy DLL files to target directory for MSVC builds
function(copy_dlls target_name dlls_root)
    if (MSVC)
        # Find all DLL files in the specified root directory
        file(GLOB DLLS ${dlls_root}/*.dll)
        # For each DLL file found
        foreach (DLL ${DLLS})
            # Add post-build command to copy DLL to target directory
            add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${DLL} $<TARGET_FILE_DIR:${target_name}>)
        endforeach ()
    endif ()
endfunction()

# Log start of Abstract Engine configuration
message("--- Abstract Engine Start ---")

# Create interface library for abstract inference engine
add_library(AbstractInferenceEngine INTERFACE)

# Add include directories for the interface
target_include_directories(AbstractInferenceEngine INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

# Log end of Abstract Engine configuration
message("----- Abstract Engine End")
