# Standalone benchmark comparing the storage back-ends of
# Triangulation_data_structure_3 (pointer/Compact_container vs. Indexed_storage).
#
# Configure it against any CGAL source tree, e.g.
#   cmake -S . -B build-index -DCGAL_DIR=<cgal-tree> -DCMAKE_BUILD_TYPE=Release -DTDS_STORAGE_INDEX=ON
# The driver compare_storage_benchmark.py does this for all configurations.

cmake_minimum_required(VERSION 3.12...3.31)
project(TDS_3_storage_benchmark CXX)

find_package(CGAL REQUIRED)

option(TDS_STORAGE_INDEX "Use CGAL::Index_tag (Indexed_storage) instead of CGAL::Handle_tag" OFF)

add_executable(storage_benchmark storage_benchmark.cpp)
target_link_libraries(storage_benchmark PRIVATE CGAL::CGAL)
target_compile_features(storage_benchmark PRIVATE cxx_std_17)
if(TDS_STORAGE_INDEX)
  target_compile_definitions(storage_benchmark PRIVATE TDS_STORAGE_INDEX=1)
endif()
