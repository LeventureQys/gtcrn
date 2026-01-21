cmake_minimum_required(VERSION 3.12)
project(GTCRN_C VERSION 1.0.0 LANGUAGES C)

# C Standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# For MSVC multi-config: Remove /RTC1 from Debug flags to allow optimization
if(MSVC)
    # Remove /RTC1 from default Debug flags
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" CACHE STRING "Debug C flags" FORCE)
endif()

# Compiler flags
if(MSVC)
    # MSVC specific flags
    add_compile_options(/W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    # UTF-8 encoding support for source files and execution
    add_compile_options(/utf-8)
    # Disable specific warnings
    add_compile_options(/wd4244 /wd4267)  # conversion warnings
    add_compile_options(/wd4819)  # encoding warnings

    # Add optimization for all configurations (including Debug)
    add_compile_options(/O2 /fp:fast)

    # Add debug info for Debug and RelWithDebInfo
    add_compile_options($<$<CONFIG:Debug>:/Zi>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:/Zi>)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
    # Optimization flags only for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-O3 -ffast-math)
        if(NOT CMAKE_CROSSCOMPILING)
            add_compile_options(-march=native)
        endif()
    endif()
endif()

# Option for double precision
option(GTCRN_USE_DOUBLE "Use double precision instead of float" OFF)
if(GTCRN_USE_DOUBLE)
    add_definitions(-DGTCRN_USE_DOUBLE)
endif()

# Option for debug output
option(GTCRN_STREAM_DEBUG_ENABLE "Enable streaming debug output" OFF)
if(GTCRN_STREAM_DEBUG_ENABLE)
    add_definitions(-DGTCRN_STREAM_DEBUG=1)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library source files
set(GTCRN_SOURCES
    src/gtcrn_math.c
    src/gtcrn_layers.c
    src/gtcrn_fft.c
    src/gtcrn_model.c
    src/gtcrn_forward.c
    src/gtcrn_stream.c
)

# Static library
add_library(gtcrn STATIC ${GTCRN_SOURCES})
target_include_directories(gtcrn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link math library on Unix
if(UNIX)
    target_link_libraries(gtcrn m)
endif()

# ============================================================================
# Demo Executables
# ============================================================================

# Offline demo: processes entire audio file at once
add_executable(gtcrn_demo demo/main.c demo/wav_io.c)
target_link_libraries(gtcrn_demo gtcrn)
if(UNIX)
    target_link_libraries(gtcrn_demo m)
endif()

# Streaming demo: processes audio frame-by-frame (real-time capable)
add_executable(gtcrn_stream_demo demo/main_stream.c demo/wav_io.c)
target_link_libraries(gtcrn_stream_demo gtcrn)
if(UNIX)
    target_link_libraries(gtcrn_stream_demo m)
endif()

# ============================================================================
# Debug Executables (optional, for development)
# ============================================================================
option(GTCRN_BUILD_DEBUG_TOOLS "Build debug/verification tools" OFF)

if(GTCRN_BUILD_DEBUG_TOOLS)
    # Test STFT->ISTFT roundtrip
    add_executable(test_stft_istft_roundtrip demo/test_stft_istft_roundtrip.c)
    target_link_libraries(test_stft_istft_roundtrip gtcrn)
    if(UNIX)
        target_link_libraries(test_stft_istft_roundtrip m)
    endif()

    # Test FFT roundtrip
    add_executable(test_fft_roundtrip demo/test_fft_roundtrip.c)
    target_link_libraries(test_fft_roundtrip gtcrn)
    if(UNIX)
        target_link_libraries(test_fft_roundtrip m)
    endif()

    # Test ISTFT standalone
    add_executable(test_istft_standalone demo/test_istft_standalone.c demo/wav_io.c)
    target_link_libraries(test_istft_standalone gtcrn)
    if(UNIX)
        target_link_libraries(test_istft_standalone m)
    endif()

    # Test STFT match
    add_executable(test_stft_match demo/test_stft_match.c demo/wav_io.c)
    target_link_libraries(test_stft_match gtcrn)
    if(UNIX)
        target_link_libraries(test_stft_match m)
    endif()

    # Test stream debug
    add_executable(test_stream_debug demo/test_stream_debug.c demo/wav_io.c)
    target_link_libraries(test_stream_debug gtcrn)
    if(UNIX)
        target_link_libraries(test_stream_debug m)
    endif()

    # Test stream vs complete
    add_executable(test_stream_vs_complete demo/test_stream_vs_complete.c demo/wav_io.c)
    target_link_libraries(test_stream_vs_complete gtcrn)
    if(UNIX)
        target_link_libraries(test_stream_vs_complete m)
    endif()

    # Test frame 0 debug
    add_executable(test_frame0_debug demo/test_frame0_debug.c demo/wav_io.c)
    target_link_libraries(test_frame0_debug gtcrn)
    if(UNIX)
        target_link_libraries(test_frame0_debug m)
    endif()

    # Test stream spectrum
    add_executable(test_stream_spectrum demo/test_stream_spectrum.c demo/wav_io.c)
    target_link_libraries(test_stream_spectrum gtcrn)
    if(UNIX)
        target_link_libraries(test_stream_spectrum m)
    endif()

    # Test intermediate outputs
    add_executable(test_intermediate demo/test_intermediate.c demo/wav_io.c)
    target_link_libraries(test_intermediate gtcrn)
    if(UNIX)
        target_link_libraries(test_intermediate m)
    endif()

    # Debug forward executable
    add_executable(debug_forward demo/debug_forward.c)
    target_link_libraries(debug_forward gtcrn)
    if(UNIX)
        target_link_libraries(debug_forward m)
    endif()

    # Debug EnConv0 executable
    add_executable(debug_enconv0 demo/debug_enconv0.c)
    target_link_libraries(debug_enconv0 gtcrn)
    if(UNIX)
        target_link_libraries(debug_enconv0 m)
    endif()

    # Debug full forward executable
    add_executable(debug_full_forward demo/debug_full_forward.c)
    target_link_libraries(debug_full_forward gtcrn)
    if(UNIX)
        target_link_libraries(debug_full_forward m)
    endif()

    # Debug GTConvBlock executable
    add_executable(debug_gtconv demo/debug_gtconv.c)
    target_link_libraries(debug_gtconv gtcrn)
    if(UNIX)
        target_link_libraries(debug_gtconv m)
    endif()

    # Debug TRA executable
    add_executable(debug_tra demo/debug_tra.c)
    target_link_libraries(debug_tra gtcrn)
    if(UNIX)
        target_link_libraries(debug_tra m)
    endif()

    # Test save spectrum (for Python comparison)
    add_executable(test_save_spectrum demo/test_save_spectrum.c demo/wav_io.c)
    target_link_libraries(test_save_spectrum gtcrn)
    if(UNIX)
        target_link_libraries(test_save_spectrum m)
    endif()

    # Test STFT frame 0
    add_executable(test_stft_frame0 demo/test_stft_frame0.c demo/wav_io.c)
    target_link_libraries(test_stft_frame0 gtcrn)
    if(UNIX)
        target_link_libraries(test_stft_frame0 m)
    endif()

    # Debug EnGT2
    add_executable(debug_engt2 demo/debug_engt2.c demo/wav_io.c)
    target_link_libraries(debug_engt2 gtcrn)
    if(UNIX)
        target_link_libraries(debug_engt2 m)
    endif()
endif()

# Install targets
install(TARGETS gtcrn
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(TARGETS gtcrn_demo gtcrn_stream_demo
    RUNTIME DESTINATION bin
)

# Print configuration
message(STATUS "")
message(STATUS "GTCRN C Library Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  Double precision: ${GTCRN_USE_DOUBLE}")
message(STATUS "  Build debug tools: ${GTCRN_BUILD_DEBUG_TOOLS}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  gtcrn            - Static library")
message(STATUS "  gtcrn_demo       - Offline processing demo")
message(STATUS "  gtcrn_stream_demo - Streaming (real-time) demo")
message(STATUS "")
