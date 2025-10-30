cmake_minimum_required(VERSION 3.16)
project(CuedSpeechDecoder VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build tests" OFF)

# Detect $HOME/local as a convenient default prefix
if(NOT DEFINED HOME_LOCAL_PREFIX)
  if(DEFINED ENV{HOME})
    set(HOME_LOCAL_PREFIX "$ENV{HOME}/local")
  else()
    set(HOME_LOCAL_PREFIX "")
  endif()
endif()

# Helper macro to prepend a path to CMAKE_PREFIX_PATH if it exists
macro(_append_prefix_if_exists path)
  if(EXISTS "${path}")
    list(APPEND CMAKE_PREFIX_PATH "${path}")
  endif()
endmacro()

# Add common local prefixes to CMAKE_PREFIX_PATH to help find_package
if(HOME_LOCAL_PREFIX)
  _append_prefix_if_exists("${HOME_LOCAL_PREFIX}")
  _append_prefix_if_exists("${HOME_LOCAL_PREFIX}/flashlight-text")
  _append_prefix_if_exists("${HOME_LOCAL_PREFIX}/kenlm")
  _append_prefix_if_exists("${HOME_LOCAL_PREFIX}/opencv")
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

# flashlight-text (prefer imported targets)
find_package(flashlight-text CONFIG QUIET)
if(NOT flashlight-text_FOUND)
  # Try explicit hint to the common install path
  if(HOME_LOCAL_PREFIX)
    set(flashlight-text_DIR "${HOME_LOCAL_PREFIX}/flashlight-text/share/flashlight-text/cmake" CACHE PATH "")
  endif()
  find_package(flashlight-text CONFIG QUIET)
endif()

# KenLM (via flashlight-text imported target or fallback)
set(USE_IMPORTED_KENLM OFF)
if(flashlight-text_FOUND)
  set(USE_IMPORTED_KENLM ON)
endif()

# TensorFlow Lite (headers+lib)
# Allow override via -DTFLITE_INCLUDE_DIR / -DTFLITE_LIBRARY
if(NOT TFLITE_INCLUDE_DIR AND HOME_LOCAL_PREFIX)
  set(TFLITE_INCLUDE_DIR "${HOME_LOCAL_PREFIX}/tflite/include" CACHE PATH "")
endif()
if(NOT TFLITE_LIBRARY AND HOME_LOCAL_PREFIX)
  # Prefer static; users can override with their shared lib
  if(EXISTS "${HOME_LOCAL_PREFIX}/tflite/lib/libtensorflow-lite.a")
    set(TFLITE_LIBRARY "${HOME_LOCAL_PREFIX}/tflite/lib/libtensorflow-lite.a" CACHE FILEPATH "")
  elseif(EXISTS "${HOME_LOCAL_PREFIX}/tflite/lib64/libtensorflow-lite.a")
    set(TFLITE_LIBRARY "${HOME_LOCAL_PREFIX}/tflite/lib64/libtensorflow-lite.a" CACHE FILEPATH "")
  elseif(EXISTS "${HOME_LOCAL_PREFIX}/tflite/lib/libtensorflowlite.so")
    set(TFLITE_LIBRARY "${HOME_LOCAL_PREFIX}/tflite/lib/libtensorflowlite.so" CACHE FILEPATH "")
  elseif(EXISTS "${HOME_LOCAL_PREFIX}/tflite/lib64/libtensorflowlite.so")
    set(TFLITE_LIBRARY "${HOME_LOCAL_PREFIX}/tflite/lib64/libtensorflowlite.so" CACHE FILEPATH "")
  endif()
endif()
if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY)
  message(FATAL_ERROR "TensorFlow Lite not found. Set TFLITE_INCLUDE_DIR and TFLITE_LIBRARY (e.g., $HOME/local/tflite/include and $HOME/local/tflite/lib/libtensorflow-lite.a)")
endif()
set(TFLITE_INCLUDE_DIRS ${TFLITE_INCLUDE_DIR})
set(TFLITE_LIBRARIES ${TFLITE_LIBRARY})
get_filename_component(TFLITE_LIB_DIR "${TFLITE_LIBRARY}" DIRECTORY)

set(_tflite_support_libs)
foreach(_dep IN ITEMS flatbuffers farmhash pthreadpool cpuinfo fft2d_fftsg fft2d_fftsg2d)
  find_library(TFLITE_${_dep}_LIB
    NAMES ${_dep}
    PATHS "${TFLITE_LIB_DIR}"
    NO_DEFAULT_PATH)
  if(TFLITE_${_dep}_LIB)
    list(APPEND _tflite_support_libs "${TFLITE_${_dep}_LIB}")
  endif()
endforeach()

file(GLOB TFLITE_RUY_LIBS "${TFLITE_LIB_DIR}/libruy*.a")
list(SORT TFLITE_RUY_LIBS)
list(APPEND _tflite_support_libs ${TFLITE_RUY_LIBS})
list(REMOVE_DUPLICATES _tflite_support_libs)
set(TFLITE_SUPPORT_LIBS ${_tflite_support_libs} CACHE STRING "Additional TensorFlow Lite static dependencies" FORCE)

# OpenCV (prefer package config)
find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
if(NOT OpenCV_FOUND)
  if(HOME_LOCAL_PREFIX)
    # Typical OpenCV config dir
    set(OpenCV_DIR "${HOME_LOCAL_PREFIX}/opencv/lib/cmake/opencv4" CACHE PATH "")
  endif()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
endif()
set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(OPENCV_LIBRARIES ${OpenCV_LIBS})

find_package(Threads REQUIRED)
find_library(MATH_LIBRARY m)

# KenLM fallback (only if not using imported targets)
if(NOT USE_IMPORTED_KENLM)
  if(NOT KENLM_INCLUDE_DIR AND HOME_LOCAL_PREFIX)
    set(KENLM_INCLUDE_DIR "${HOME_LOCAL_PREFIX}/kenlm/include" CACHE PATH "")
  endif()
  if(NOT KENLM_LIBRARY AND HOME_LOCAL_PREFIX)
    if(EXISTS "${HOME_LOCAL_PREFIX}/kenlm/lib/libkenlm.a")
      set(KENLM_LIBRARY "${HOME_LOCAL_PREFIX}/kenlm/lib/libkenlm.a" CACHE FILEPATH "")
    elseif(EXISTS "${HOME_LOCAL_PREFIX}/kenlm/lib/libkenlm.so")
      set(KENLM_LIBRARY "${HOME_LOCAL_PREFIX}/kenlm/lib/libkenlm.so" CACHE FILEPATH "")
    endif()
  endif()
  if(NOT KENLM_INCLUDE_DIR OR NOT KENLM_LIBRARY)
    message(FATAL_ERROR "KenLM not found. Set KENLM_INCLUDE_DIR and KENLM_LIBRARY (e.g., $HOME/local/kenlm/include and $HOME/local/kenlm/lib/libkenlm.a)")
  endif()
  set(KENLM_INCLUDE_DIRS ${KENLM_INCLUDE_DIR})
  set(KENLM_LIBRARIES ${KENLM_LIBRARY})
endif()

# Ensure KenLM headers are available even when provided via flashlight-text
if(NOT KENLM_INCLUDE_DIRS)
  find_path(KENLM_INCLUDE_DIRS
    NAMES kenlm/lm/model.hh
    HINTS
      ${HOME_LOCAL_PREFIX}/kenlm/include
      ${HOME_LOCAL_PREFIX}/kenlm
      ${HOME_LOCAL_PREFIX}/flashlight-text/include
      ${HOME_LOCAL_PREFIX}
    PATH_SUFFIXES include
  )
endif()

message(STATUS "flashlight-text: ${flashlight-text_FOUND}")
message(STATUS "TFLite: ${TFLITE_LIBRARIES}")
message(STATUS "OpenCV: ${OPENCV_LIBRARIES}")
if(USE_IMPORTED_KENLM)
  message(STATUS "KenLM via flashlight-text imported target")
else()
  message(STATUS "KenLM (manual): ${KENLM_LIBRARIES}")
endif()

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

set(DECODER_SOURCES
    decoder.cpp
    decoder_c_api.cpp
)

set(DECODER_HEADERS
    decoder.h
    decoder_c_api.h
)

if(BUILD_SHARED_LIBS)
  add_library(cued_speech_decoder SHARED ${DECODER_SOURCES})
  set_target_properties(cued_speech_decoder PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${DECODER_HEADERS}"
  )
else()
  add_library(cued_speech_decoder STATIC ${DECODER_SOURCES})
  set_target_properties(cued_speech_decoder PROPERTIES
    PUBLIC_HEADER "${DECODER_HEADERS}"
  )
endif()

# Includes
if(flashlight-text_FOUND)
  target_include_directories(cued_speech_decoder
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>
      ${KENLM_INCLUDE_DIRS}
    PRIVATE
      ${TFLITE_INCLUDE_DIRS}
      ${OPENCV_INCLUDE_DIRS}
  )
else()
  target_include_directories(cued_speech_decoder
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>
      ${KENLM_INCLUDE_DIRS}
    PRIVATE
      ${TFLITE_INCLUDE_DIRS}
      ${OPENCV_INCLUDE_DIRS}
  )
endif()

# Links
if(flashlight-text_FOUND)
  target_link_libraries(cued_speech_decoder
    PUBLIC
      flashlight::flashlight-text
      flashlight::flashlight-text-kenlm
      ${OPENCV_LIBRARIES}
  )
else()
  target_link_libraries(cued_speech_decoder
    PUBLIC
      ${KENLM_LIBRARIES}
      ${OPENCV_LIBRARIES}
  )
endif()

set(_tflite_link_libs ${TFLITE_LIBRARIES})
if(TFLITE_SUPPORT_LIBS)
  list(APPEND _tflite_link_libs ${TFLITE_SUPPORT_LIBS})
endif()

if(TFLITE_LIBRARY MATCHES "\\.a$" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel" )
  target_link_libraries(cued_speech_decoder
    PUBLIC
      "-Wl,--start-group"
      ${_tflite_link_libs}
      "-Wl,--end-group"
      Threads::Threads
  )
else()
  target_link_libraries(cued_speech_decoder
    PUBLIC
      ${_tflite_link_libs}
      Threads::Threads
  )
endif()

if(CMAKE_DL_LIBS)
  target_link_libraries(cued_speech_decoder PUBLIC ${CMAKE_DL_LIBS})
endif()

if(MATH_LIBRARY)
  target_link_libraries(cued_speech_decoder PUBLIC ${MATH_LIBRARY})
endif()

target_compile_definitions(cued_speech_decoder
  PUBLIC
    KENLM_MAX_ORDER=6
)

# Warnings
if(MSVC)
  target_compile_options(cued_speech_decoder PRIVATE /W4)
else()
  target_compile_options(cued_speech_decoder PRIVATE -Wall -Wextra -pedantic)
endif()

# Demo executable
add_executable(demo_decode demo_decode.cpp)
target_link_libraries(demo_decode PRIVATE cued_speech_decoder)

# Install
include(GNUInstallDirs)
install(TARGETS cued_speech_decoder
  EXPORT CuedSpeechDecoderTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cued_speech
)

# Export package config
include(CMakePackageConfigHelpers)
install(EXPORT CuedSpeechDecoderTargets
  FILE CuedSpeechDecoderTargets.cmake
  NAMESPACE CuedSpeech::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CuedSpeechDecoder
)
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

# Summary
message(STATUS "")
message(STATUS "CuedSpeechDecoder Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        