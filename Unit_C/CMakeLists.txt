# ==============================================================================
# GTCRN C Implementation - CMake Build Configuration
# ==============================================================================
#
# 使用方法:
#   mkdir build && cd build
#   cmake ..
#   cmake --build .
#
# 构建选项:
#   -DCMAKE_BUILD_TYPE=Release/Debug  # 构建类型
#   -DSAMPLE_RATE=16000/48000         # 采样率 (默认 48000)
#   -DBUILD_TESTS=ON/OFF              # 是否构建测试 (默认 ON)
#   -DBUILD_EXAMPLES=ON/OFF           # 是否构建示例 (默认 ON)
#
# ==============================================================================

cmake_minimum_required(VERSION 3.10)
project(gtcrn_c VERSION 1.0.0 LANGUAGES C)

# ==============================================================================
# 目录定义
# ==============================================================================

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)
set(SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/scripts)

# ==============================================================================
# 构建选项
# ==============================================================================

option(BUILD_TESTS "Build test executables" ON)
option(BUILD_EXAMPLES "Build example executables" ON)
option(BUILD_SHARED_LIBS "Build shared library" OFF)
set(SAMPLE_RATE "48000" CACHE STRING "Sample rate (16000 or 48000)")

# 验证采样率
if(NOT SAMPLE_RATE STREQUAL "16000" AND NOT SAMPLE_RATE STREQUAL "48000")
    message(FATAL_ERROR "SAMPLE_RATE must be 16000 or 48000")
endif()

message(STATUS "GTCRN C Implementation")
message(STATUS "  Sample Rate: ${SAMPLE_RATE} Hz")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")

# ==============================================================================
# 编译器设置
# ==============================================================================

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 编译选项
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -DDEBUG")

    # 平台特定优化
    if(NOT CMAKE_CROSSCOMPILING)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-march=native" HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
        endif()
    endif()
elseif(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /fp:fast")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Od /Zi /DDEBUG")
endif()

# ==============================================================================
# 源文件定义
# ==============================================================================

# 神经网络层源文件
set(NN_LAYER_SOURCES
    ${SRC_DIR}/conv2d.c
    ${SRC_DIR}/batchnorm2d.c
    ${SRC_DIR}/nn_layers.c
    ${SRC_DIR}/layernorm.c
    ${SRC_DIR}/GRU.c
)

# 模型核心源文件
set(MODEL_SOURCES
    ${SRC_DIR}/gtcrn_model.c
    ${SRC_DIR}/gtcrn_modules.c
)

# 流式处理源文件 (根据采样率选择)
if(SAMPLE_RATE STREQUAL "16000")
    set(STREAMING_SOURCES
        ${SRC_DIR}/gtcrn_streaming_16k.c
        ${SRC_DIR}/gtcrn_streaming_optimized_16k.c
        ${SRC_DIR}/gtcrn_streaming_impl.c
        ${SRC_DIR}/stream_conv.c
    )
else()
    set(STREAMING_SOURCES
        ${SRC_DIR}/gtcrn_streaming.c
        ${SRC_DIR}/gtcrn_streaming_optimized.c
        ${SRC_DIR}/gtcrn_streaming_impl.c
        ${SRC_DIR}/stream_conv.c
    )
endif()

# 信号处理源文件 (根据采样率选择)
if(SAMPLE_RATE STREQUAL "16000")
    set(SIGNAL_SOURCES ${SRC_DIR}/stft_16k.c)
    add_definitions(-DSAMPLE_RATE_16K)
else()
    set(SIGNAL_SOURCES ${SRC_DIR}/stft.c)
    add_definitions(-DSAMPLE_RATE_48K)
endif()

# 工具源文件
set(UTIL_SOURCES
    ${SRC_DIR}/weight_loader.c
    ${SRC_DIR}/gtconvblock_forward_complete.c
    ${SRC_DIR}/GRU_bidirectional_complete.c
)

# 完整的库源文件
set(GTCRN_SOURCES
    ${NN_LAYER_SOURCES}
    ${MODEL_SOURCES}
    ${STREAMING_SOURCES}
    ${SIGNAL_SOURCES}
    ${UTIL_SOURCES}
)

# 头文件
set(GTCRN_HEADERS
    ${INCLUDE_DIR}/conv2d.h
    ${INCLUDE_DIR}/batchnorm2d.h
    ${INCLUDE_DIR}/nn_layers.h
    ${INCLUDE_DIR}/layernorm.h
    ${INCLUDE_DIR}/GRU.h
    ${INCLUDE_DIR}/gtcrn_model.h
    ${INCLUDE_DIR}/gtcrn_modules.h
    ${INCLUDE_DIR}/gtcrn_streaming.h
    ${INCLUDE_DIR}/stream_conv.h
    ${INCLUDE_DIR}/weight_loader.h
)

if(SAMPLE_RATE STREQUAL "16000")
    list(APPEND GTCRN_HEADERS ${INCLUDE_DIR}/stft_16k.h ${INCLUDE_DIR}/gtcrn_streaming_16k.h)
else()
    list(APPEND GTCRN_HEADERS ${INCLUDE_DIR}/stft.h)
endif()

# ==============================================================================
# 库目标
# ==============================================================================

# 静态库
add_library(gtcrn_static STATIC ${GTCRN_SOURCES})
target_include_directories(gtcrn_static PUBLIC ${INCLUDE_DIR})

# 链接数学库 (仅在非 MSVC 编译器上需要)
if(NOT MSVC)
    target_link_libraries(gtcrn_static PUBLIC m)
endif()

set_target_properties(gtcrn_static PROPERTIES OUTPUT_NAME gtcrn)

# 共享库 (可选)
if(BUILD_SHARED_LIBS)
    add_library(gtcrn_shared SHARED ${GTCRN_SOURCES})
    target_include_directories(gtcrn_shared PUBLIC ${INCLUDE_DIR})

    # 链接数学库 (仅在非 MSVC 编译器上需要)
    if(NOT MSVC)
        target_link_libraries(gtcrn_shared PUBLIC m)
    endif()

    set_target_properties(gtcrn_shared PROPERTIES OUTPUT_NAME gtcrn)
endif()

# 创建别名目标
add_library(gtcrn::gtcrn ALIAS gtcrn_static)

# ==============================================================================
# 可执行目标
# ==============================================================================

# 主程序 - 实时降噪
if(BUILD_EXAMPLES)
    if(SAMPLE_RATE STREQUAL "16000")
        add_executable(denoise ${EXAMPLES_DIR}/example_realtime_denoise_16k.c)
    else()
        add_executable(denoise ${EXAMPLES_DIR}/example_realtime_denoise.c)
    endif()
    target_link_libraries(denoise PRIVATE gtcrn_static)

    # 完整模块示例
    add_executable(example_complete ${EXAMPLES_DIR}/example_use_complete_modules.c)
    target_link_libraries(example_complete PRIVATE gtcrn_static)

    # ConvTranspose2d 示例
    add_executable(conv_transpose2d_example ${EXAMPLES_DIR}/conv_transpose2d_example.c)
    target_link_libraries(conv_transpose2d_example PRIVATE gtcrn_static)

    # ConvTranspose2d 可视化
    add_executable(conv_transpose2d_visual ${EXAMPLES_DIR}/conv_transpose2d_visual.c)
    target_link_libraries(conv_transpose2d_visual PRIVATE gtcrn_static)

    # BatchNorm 融合可视化
    add_executable(fusion_visualization ${EXAMPLES_DIR}/fusion_visualization.c)
    target_link_libraries(fusion_visualization PRIVATE gtcrn_static)
endif()

# ==============================================================================
# 测试目标
# ==============================================================================

if(BUILD_TESTS)
    enable_testing()

    # GRU 测试
    add_executable(test_gru ${TESTS_DIR}/test_gru.c)
    target_link_libraries(test_gru PRIVATE gtcrn_static)
    add_test(NAME test_gru COMMAND test_gru)

    # Conv2D 测试
    add_executable(test_conv2d ${TESTS_DIR}/test_conv2d.c)
    target_link_libraries(test_conv2d PRIVATE gtcrn_static)
    add_test(NAME test_conv2d COMMAND test_conv2d)

    # GTCRN 模型测试
    add_executable(test_gtcrn_model ${TESTS_DIR}/test_gtcrn_model.c)
    target_link_libraries(test_gtcrn_model PRIVATE gtcrn_static)
    add_test(NAME test_gtcrn_model COMMAND test_gtcrn_model)

    # GTCRN 模块测试
    add_executable(test_gtcrn_modules ${TESTS_DIR}/test_gtcrn_modules.c)
    target_link_libraries(test_gtcrn_modules PRIVATE gtcrn_static)
    add_test(NAME test_gtcrn_modules COMMAND test_gtcrn_modules)

    # BatchNorm 测试
    add_executable(test_batchnorm ${TESTS_DIR}/test_batchnorm_fusion.c)
    target_link_libraries(test_batchnorm PRIVATE gtcrn_static)
    add_test(NAME test_batchnorm COMMAND test_batchnorm)

    # LayerNorm 测试
    add_executable(test_layernorm ${TESTS_DIR}/test_layernorm.c)
    target_link_libraries(test_layernorm PRIVATE gtcrn_static)
    add_test(NAME test_layernorm COMMAND test_layernorm)

    # NN Layers 测试
    add_executable(test_nn_layers ${TESTS_DIR}/test_nn_layers.c)
    target_link_libraries(test_nn_layers PRIVATE gtcrn_static)
    add_test(NAME test_nn_layers COMMAND test_nn_layers)

    # DPGRNN 测试
    add_executable(test_dpgrnn ${TESTS_DIR}/test_dpgrnn.c)
    target_link_libraries(test_dpgrnn PRIVATE gtcrn_static)
    add_test(NAME test_dpgrnn COMMAND test_dpgrnn)

    # TRA 流式测试
    add_executable(test_tra_stream ${TESTS_DIR}/test_tra_stream.c)
    target_link_libraries(test_tra_stream PRIVATE gtcrn_static)
    add_test(NAME test_tra_stream COMMAND test_tra_stream)
endif()

# ==============================================================================
# 安装配置
# ==============================================================================

include(GNUInstallDirs)

# 安装库
install(TARGETS gtcrn_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_SHARED_LIBS)
    install(TARGETS gtcrn_shared
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# 安装头文件
install(FILES ${GTCRN_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtcrn
)

# 安装可执行文件
if(BUILD_EXAMPLES)
    install(TARGETS denoise
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# ==============================================================================
# CPack 配置 (用于生成安装包)
# ==============================================================================

set(CPACK_PACKAGE_NAME "gtcrn")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GTCRN Speech Enhancement - C Implementation")
set(CPACK_PACKAGE_VENDOR "GTCRN Project")

include(CPack)

# ==============================================================================
# 自定义目标
# ==============================================================================

# 格式化代码 (如果有 clang-format)
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${GTCRN_SOURCES} ${GTCRN_HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting source code..."
    )
endif()

# 清理所有构建产物
add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMENT "Removing build directory..."
)

# ==============================================================================
# 构建信息摘要
# ==============================================================================

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C Flags: ${CMAKE_C_FLAGS}")
message(STATUS "  C Flags (Release): ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "  C Flags (Debug): ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "")
