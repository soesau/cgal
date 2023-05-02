#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#ifdef USE_POLYHEDRON
#include <CGAL/Polyhedron_3.h>
#else
#include <CGAL/Surface_mesh.h>
#endif
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Polygon_mesh.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
//#include "include/utils.h"

#include <CGAL/Polygon_mesh_processing/compute_normal.h>

#include <CGAL/Shape_regularization/Shape_optimization.h>

// Typedefs.
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Vector_3 = typename Kernel::Vector_3;

#ifdef USE_POLYHEDRON
using Polygon_mesh = CGAL::Polyhedron_3<Kernel>;
#else
using Polygon_mesh = CGAL::Surface_mesh<Point_3>;
#endif

using Neighbor_query = CGAL::Shape_detection::Polygon_mesh::One_ring_neighbor_query<Polygon_mesh>;
using Region_type = CGAL::Shape_detection::Polygon_mesh::Least_squares_plane_fit_region<Kernel, Polygon_mesh>;
using Sorting = CGAL::Shape_detection::Polygon_mesh::Least_squares_plane_fit_sorting<Kernel, Polygon_mesh, Neighbor_query>;
using Region_growing = CGAL::Shape_detection::Region_growing<Neighbor_query, Region_type>;

int main(int argc, char* argv[]) {
  // Default parameter values for the data file building.off.
  FT max_distance = FT(1.0);
  FT max_angle = FT(10);
  std::size_t min_region_size = 50;
  FT fidelity = 1.0;
  FT simplicity = 1.0;
  FT completeness = 1.0;

  // Load data either from a local folder or a user-provided file.
  bool is_default_input = argc > 1 ? false : true;
  std::string filename = is_default_input ? CGAL::data_file_path("meshes/building.off") : argv[1];
  if (false) {
    is_default_input = false;
    filename = "C:/Data/Bentley/graz_subset.off";
    min_region_size = 1;
    max_distance = 0.6;
    completeness = 100.0;
    simplicity = 20.0;
  }
  if (true) {
    is_default_input = false;
    filename = CGAL::data_file_path("meshes/bunny00.off");
    max_distance = 0.08;
    fidelity = 1;
    min_region_size = 1;
  }
  if (false) {
    is_default_input = false;
    //filename = "c:/Data/subset.off";//"c:/Data/homepage_bracket.stl";
    filename = "c:/Data/homepage_bracket.stl";
    max_distance = 0.0625; // 0.05
    min_region_size = 1; // 1
    simplicity = 1000.0; // 100.0
    completeness = 100.0; // 100.0
  }
  if (false) {
    is_default_input = false;
    filename = "c:/Data/subset.off";
    max_distance = 0.0625; // 0.05
    min_region_size = 1; // 1
    simplicity = 1000.0; // 100.0
    completeness = 100000.0; // 100.0
  }
  if (false) {
    is_default_input = false;
    //filename = "c:/Data/subset.off";//"c:/Data/homepage_bracket.stl";
    filename = "c:/Data/sphere.off";
    max_distance = 0.0625; // 0.05
    FT max_angle = FT(40);
    min_region_size = 1; // 1
    simplicity = 1000.0; // 100.0
    completeness = 100.0; // 100.0
  }
  if (false) {
    is_default_input = false;
    //filename = "c:/Data/subset.off";//"c:/Data/homepage_bracket.stl";
    filename = "c:/Data/sphere2.off";
    max_distance = 0.0625; // 0.05
    FT max_angle = FT(40);
    min_region_size = 1; // 1
    simplicity = 1000.0; // 100.0
    completeness = 100.0; // 100.0
  }
  std::ifstream in(filename);
  CGAL::IO::set_ascii_mode(in);

  Polygon_mesh polygon_mesh;
  if (!CGAL::IO::read_polygon_mesh(filename, polygon_mesh)) {
    std::cerr << "ERROR: cannot read the input file!" << std::endl;
    return EXIT_FAILURE;
  }
  const auto& face_range = faces(polygon_mesh);
  std::cout << "* number of input faces: " << face_range.size() << std::endl;
  std::cout << "* max_distance: " << max_distance << std::endl;
  std::cout << "* min_region_size: " << min_region_size << std::endl;
  assert(!is_default_input || face_range.size() == 32245);
  /*

    auto fnormals = polygon_mesh.add_property_map<typename Polygon_mesh::Face_index, Vector_3>("f:normals", CGAL::NULL_VECTOR).first;

    CGAL::Polygon_mesh_processing::compute_face_normals(polygon_mesh, fnormals);*/

    // Create instances of the classes Neighbor_query and Region_type.
  Neighbor_query neighbor_query(polygon_mesh);

  Region_type region_type(
    polygon_mesh,
    CGAL::parameters::
    maximum_distance(max_distance).
    maximum_angle(max_angle).
    minimum_region_size(min_region_size));

  // Sort face indices.
  Sorting sorting(
    polygon_mesh, neighbor_query);
  sorting.sort();

  // Create an instance of the region growing class.
  Region_growing region_growing(
    face_range, sorting.ordered(), neighbor_query, region_type);

  // Run the algorithm.
  std::vector<typename Region_growing::Primitive_and_region> regions;
  region_growing.detect(std::back_inserter(regions));
  std::cout << "* number of found planes: " << regions.size() << std::endl;

  const Region_growing::Region_map& map = region_growing.region_map();

  for (std::size_t i = 0; i < regions.size(); i++)
    for (auto& item : regions[i].second) {
      if (i != get(map, item)) {
        std::cout << "Region map incorrect" << std::endl;
      }
    }

  std::vector<typename Region_growing::Item> unassigned;
  region_growing.unassigned_items(face_range, std::back_inserter(unassigned));
  std::cout << "* number of unassigned faces: " << unassigned.size() << std::endl;

  for (auto& item : unassigned) {
    if (std::size_t(-1) != get(map, item)) {
      std::cout << "Region map for unassigned incorrect" << std::endl;
    }
  }
  utils::save_polygon_mesh_regions(polygon_mesh, regions, "before.ply");

  // Unassigned are invalidated after optimization
  optimize_shapes(polygon_mesh, map, regions, fidelity, simplicity, completeness, CGAL::parameters::maximum_distance(max_distance).minimum_region_size(min_region_size).geom_traits(Kernel()));

  // Save regions to a file.
  utils::save_polygon_mesh_regions(polygon_mesh, regions, "after.ply");

  return EXIT_SUCCESS;
}
