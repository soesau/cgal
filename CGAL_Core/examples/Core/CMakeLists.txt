cmake_minimum_required(VERSION 3.12...3.31)
project(Core_Examples)

# CGAL and its components
find_package(CGAL REQUIRED COMPONENTS Core)

if(NOT CGAL_Core_FOUND)
  message("NOTICE: This project requires the CGAL_Core library, and will not be compiled.")
  return()
endif()

create_single_source_cgal_program("zero-one.cpp")
create_single_source_cgal_program("delaunay.cpp")
