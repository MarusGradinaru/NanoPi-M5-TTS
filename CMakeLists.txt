cmake_minimum_required(VERSION 3.15)
project(rk3576-tts-demo LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_RKNN "Enable RKNN for accelerated inference" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The parent folder where this project is located (NanoPi-M5-TTS)
get_filename_component(TTS_PARENT_DIR "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

set(ORT_ROOT "${TTS_PARENT_DIR}/onnxruntime-linux-aarch64-1.14.1" CACHE STRING "Path to ONNX Runtime root directory")
include_directories(${ORT_ROOT}/include)
link_directories(${ORT_ROOT}/lib)

set(PIPER_PHONEMIZE_ROOT "${TTS_PARENT_DIR}/piper-phonemize-install" CACHE STRING "Path to piper-phonemize root directory")
include_directories(${PIPER_PHONEMIZE_ROOT}/include)
link_directories(${PIPER_PHONEMIZE_ROOT}/lib)

find_package(fmt REQUIRED)

add_library(piper
    piper/piper.cpp
    piper/text-processor.cpp)

if (USE_RKNN)
    target_compile_definitions(piper PRIVATE USE_RKNN)
    target_sources(piper PRIVATE piper/rknn-inferer.cpp)
    target_link_libraries(piper PRIVATE rknnrt)
endif()

target_precompile_headers(piper PRIVATE piper/pch.hpp)
target_link_libraries(piper
    PRIVATE
    onnxruntime
    espeak-ng
    piper_phonemize
    pthread
    fmt::fmt
    dl)
target_include_directories(piper
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/piper)

add_executable(rk3576-tts-demo
    src/main.cpp)
target_link_libraries(rk3576-tts-demo PRIVATE piper)
