# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
cmake_minimum_required(VERSION 2.6.0)

project(Replica)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(THIRD_PARTY "${CMAKE_CURRENT_LIST_DIR}/3rdparty/")

find_package(Eigen3 NO_MODULE)
if(NOT ${Eigen3_FOUND})
    message(STATUS "Using 3rd party Eigen3")
    set(Eigen_INCLUDE_DIRS "${THIRD_PARTY}/eigen")
endif()

# use system Pangolin if available
find_package(Pangolin)
if(NOT ${Pangolin_FOUND})
    message(STATUS "Using 3rd party Pangolin")
    set(Pangolin_INCLUDE_DIRS "")
    # Add all component include directories
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_core/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_display/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_opengl/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_image/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_windowing/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_vars/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_geometry/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/components/pango_glgeometry/include")
    list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/build/include")
    # Update library paths for new Pangolin structure
    set(Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_display.so")
    list(APPEND Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_opengl.so")
    list(APPEND Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_core.so")
    list(APPEND Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_image.so")
    list(APPEND Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_windowing.so")
    list(APPEND Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_vars.so")
    list(APPEND Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_geometry.so")
    list(APPEND Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/libpango_glgeometry.so")
endif()

find_package(dl REQUIRED)

add_subdirectory("./ReplicaSDK")
