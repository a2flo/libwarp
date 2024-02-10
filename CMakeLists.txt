# VS/vcpkg/clang/CMake project for libwarp
cmake_minimum_required(VERSION 3.11)
if (WIN32)
	project(libwarp)
else ()
	project(warp)
endif ()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

## compile flags
add_compile_options("$<$<CONFIG:DEBUG>:-DLIBWARP_DEBUG>")

## build output
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

## source files
include_directories("include/")
add_library(${PROJECT_NAME} SHARED
	src/libwarp.cpp
	src/libwarp.mm
	src/libwarp_internal.hpp
	src/build_version.hpp
	include/libwarp/libwarp.h
	include/libwarp/warp_kernels.hpp
)

# include libfloor base configuration
set(LIBFLOOR_USER 1)
set(LIBFLOOR_LIBRARY 1)
if (WIN32)
	include($ENV{ProgramW6432}/floor/include/floor/libfloor.cmake)
else ()
	include(/opt/floor/include/floor/libfloor.cmake)
endif (WIN32)
