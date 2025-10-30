cmake_minimum_required(VERSION 3.16)
project(CuedSpeechDecoder VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build tests" OFF)

# Find dependencies
find_package(PkgConfig REQUIRED)

# Flashlight-text
find_package(flashlight-text CONFIG)
if(NOT flashlight-text_FOUND)
    message(STATUS "flashlight-text not found via CMake, trying manual search...")
    find_path(FLASHLIGHT_TEXT_INCLUDE_DIR 
        NAMES flashlight/lib/text/decoder/Decoder.h
        PATHS /usr/local/include /usr/include
    )
    find_library(FLASHLIGHT_TEXT_LIBRARY 
        NAMES flashlight-text
        PATHS /usr/local/lib /usr/lib
    )
    if(FLASHLIGHT_TEXT_INCLUDE_DIR AND FLASHLIGHT_TEXT_LIBRARY)
        set(FLASHLIGHT_TEXT_FOUND TRUE)
        set(FLASHLIGHT_TEXT_INCLUDE_DIRS ${FLASHLIGHT_TEXT_INCLUDE_DIR})
        set(FLASHLIGHT_TEXT_LIBRARIES ${FLASHLIGHT_TEXT_LIBRARY})
    else()
        message(FATAL_ERROR "flashlight-text not found. Please install from https://github.com/flashlight/text")
    endif()
endif()

# KenLM
find_path(KENLM_INCLUDE_DIR 
    NAMES kenlm/lm/model.hh
    PATHS /usr/local/include /usr/include
)
find_library(KENLM_LIBRARY 
    NAMES kenlm
    PATHS /usr/local/lib /usr/lib
)
if(NOT KENLM_INCLUDE_DIR OR NOT KENLM_LIBRARY)
    message(FATAL_ERROR "KenLM not found. Please install from https://github.com/kpu/kenlm")
endif()

set(KENLM_INCLUDE_DIRS ${KENLM_INCLUDE_DIR})
set(KENLM_LIBRARIES ${KENLM_LIBRARY})

# TensorFlow Lite
find_path(TFLITE_INCLUDE_DIR
    NAMES tensorflow/lite/interpreter.h
    PATHS /usr/local/include /usr/include
)
find_library(TFLITE_LIBRARY
    NAMES tensorflowlite tensorflow-lite
    PATHS /usr/local/lib /usr/lib
)
if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY)
    message(FATAL_ERROR "TensorFlow Lite not found. Please install the TFLite C++ library and set TFLITE_INCLUDE_DIR/TFLITE_LIBRARY if needed.")
endif()

set(TFLITE_INCLUDE_DIRS ${TFLITE_INCLUDE_DIR})
set(TFLITE_LIBRARIES ${TFLITE_LIBRARY})

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(OPENCV_LIBRARIES ${OpenCV_LIBS})

message(STATUS "Found flashlight-text: ${FLASHLIGHT_TEXT_LIBRARIES}")
message(STATUS "Found KenLM: ${KENLM_LIBRARIES}")
message(STATUS "Found TensorFlow Lite: ${TFLITE_LIBRARIES}")
message(STATUS "Found OpenCV: ${OPENCV_LIBRARIES}")
# Include directories
target_include_directories(cued_speech_decoder
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${FLASHLIGHT_TEXT_INCLUDE_DIRS}
        ${KENLM_INCLUDE_DIRS}
        ${TFLITE_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cued_speech_decoder
    PRIVATE
        ${FLASHLIGHT_TEXT_LIBRARIES}
        ${KENLM_LIBRARIES}
        ${TFLITE_LIBRARIES}
        ${OPENCV_LIBRARIES}
)

add_executable(demo_decode demo_decode.cpp)
target_link_libraries(demo_decode PRIVATE cued_speech_decoder)

# Compiler flags
if(MSVC)
    target_compile_options(cued_speech_decoder PRIVATE /W4)
else()
    target_compile_options(cued_speech_decoder PRIVATE -Wall -Wextra -pedantic)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS cued_speech_decoder
    EXPORT CuedSpeechDecoderTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cued_speech
)

# Export targets
install(EXPORT CuedSpeechDecoderTargets
    FILE CuedSpeechDecoderTargets.cmake
    NAMESPACE CuedSpeech::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CuedSpeechDecoder
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CuedSpeechDecoderConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CuedSpeechDecoderConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CuedSpeechDecoder
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CuedSpeechDecoderConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CuedSpeechDecoderConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CuedSpeechDecoder
)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "CuedSpeechDecoder Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  flashlight-text: ${FLASHLIGHT_TEXT_LIBRARIES}")
message(STATUS "  KenLM: ${KENLM_LIBRARIES}")
message(STATUS "  TensorFlow Lite: ${TFLITE_LIBRARIES}")
message(STATUS "  OpenCV: ${OPENCV_LIBRARIES}")
message(STATUS "")

