cmake_minimum_required(VERSION 3.10)

########### Define functions for DLLs copy.
function(copy_dlls target_name dlls_root)
    if (MSVC)
        file(GLOB DLLS ${dlls_root}/*.dll)
        foreach (DLL ${DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${DLL} $<TARGET_FILE_DIR:${target_name}>)
        endforeach ()
    endif ()
endfunction()


message("--- Abstract Engine Start ---")

# Create engine library
add_library(AbstractInferenceEngine INTERFACE)
target_link_libraries(AbstractInferenceEngine INTERFACE CV)

# Add include files
target_include_directories(AbstractInferenceEngine INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

message("--- Abstract Engine End ---")
