project(Surface_mesh_topology_Benchmarks)

cmake_minimum_required(VERSION 3.12...3.31)

find_package(CGAL REQUIRED)

# add_compile_definitions(CGAL_TRACE_PATH_TESTS)
# add_compile_definitions(CGAL_TRACE_CMAP_TOOLS)

set(SOURCE_FILES quadrangulation_computation_benchmarks.cpp
                 path_homotopy_with_schema.cpp)

foreach(cppfile ${SOURCE_FILES})
  create_single_source_cgal_program("${cppfile}")
endforeach()
