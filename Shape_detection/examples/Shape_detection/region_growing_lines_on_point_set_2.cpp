#include <CGAL/Simple_cartesian.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Point_set.h>
#include <CGAL/grid_simplify_point_set.h>

#include "include/utils.h"

#include <boost/range/irange.hpp>

// Typedefs.
using Kernel   = CGAL::Simple_cartesian<double>;
using FT       = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Point_2  = typename Kernel::Point_2;
using Vector_2 = typename Kernel::Vector_2;

using Point_with_normal = std::pair<Point_2, Vector_2>;
using Point_set_2       = std::vector<Point_with_normal>;

// we use Compose_property_map as the property maps are expected to operate on the item type
// std::size_t passed as parameter to Sphere_neighbor_query and Least_squares_line_fit_region
using Point_map      = CGAL::Compose_property_map<CGAL::Random_access_property_map<Point_set_2>,
                                                  CGAL::First_of_pair_property_map<Point_with_normal> >;
using Normal_map     = CGAL::Compose_property_map<CGAL::Random_access_property_map<Point_set_2>,
                                                  CGAL::Second_of_pair_property_map<Point_with_normal> >;

using Neighbor_query = CGAL::Shape_detection::Point_set::Sphere_neighbor_query<Kernel, std::size_t, Point_map>;
using Region_type    = CGAL::Shape_detection::Point_set::Least_squares_line_fit_region<Kernel, std::size_t, Point_map, Normal_map>;
using Region_growing = CGAL::Shape_detection::Region_growing<Neighbor_query, Region_type>;

int main(int argc, char *argv[]) {

  // Load xyz data either from a local folder or a user-provided file.
  const bool is_default_input = argc > 1 ? false : true;
  std::string fn;
  //fn = "C:/data/ebp bund/0019 - Kostheim-1_utm zone32 6stellig_mbes";
  //fn = "C:/data/ebp bund/0005 - Muendung-Mainz-2-BB-utm zone32 6stellig_mbes";
  fn = "C:/data/ebp bund/0003 - Muendung-Mainz-9-SB-utm zone32 6stellig_mbes";
  //std::ifstream in(is_default_input ? "C:/data/ebp bund/.pwn" : argv[1]);
  //std::ifstream in(is_default_input ? "C:/data/ebp bund/.pwn" : argv[1]);
  std::ifstream in(is_default_input ? fn + ".pwn" : argv[1]);

  std::cout << fn << std::endl;

  CGAL::IO::set_ascii_mode(in);
  if (!in) {
    std::cerr << "ERROR: cannot read the input file!" << std::endl;
    return EXIT_FAILURE;
  }

  FT a, b, c, d, e, f;
  Point_set_2 ps;
  std::size_t count = 0;
  while (in >> a >> b >> c >> d >> e >> f) {
    count++;
    if (CGAL::abs(f) > 0.8)
      continue;
    FT l = CGAL::sqrt(d * d + e * e);
    if (l == 0)
      l = 1;
    ps.push_back(
      std::make_pair(Point_2(a, b), Vector_2(d/l, e/l)));
  }
  in.close();
  std::cout << "* number of input points: " << count << " filtered to " << ps.size() << " via normal direction" << std::endl;

  // Filter outliers
  /*typedef CGAL::Search_traits_adapter<std::size_t, Point_map, CGAL::Search_traits_2<Kernel>> Traits;
  typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_circle;


  typedef CGAL::Kd_tree<Traits> Tree;
  Tree tree(boost::counting_iterator<std::size_t>(0),
    boost::counting_iterator<std::size_t>(ps.size()), Tree::Splitter(), Traits(Point_map(ps)));

  Point_set_2 ps2;
  for (Point_with_normal &pwn : ps) {
    std::vector<std::size_t> result;

    tree.search(std::back_inserter(result), Fuzzy_circle(pwn.first, 0.02, 0, Traits(Point_map(ps))));
    if (result.size() > 5)
      ps2.push_back(pwn);
  }

  std::cout << ps2.size() << " after density filtering" << std::endl;

  std::ofstream fout("filtered.pwn");
  for (const Point_with_normal &p : ps2)
    fout << p.first << " 0 " << p.second << " 0" << std::endl;

  fout.close();*/

  std::vector<std::pair<Point_3, Vector_2>> pts3d;
  pts3d.reserve(ps.size());
  for (const Point_with_normal &p : ps) {
    pts3d.push_back(std::make_pair(Point_3(p.first.x(), p.first.y(), 0), p.second));
  }

  auto it = CGAL::grid_simplify_point_set(pts3d, 0.2, CGAL::parameters::min_points_per_cell(20).point_map(CGAL::First_of_pair_property_map<std::pair<Point_3, Vector_2>>()));

  pts3d.resize(it - pts3d.begin());

  std::cout << pts3d.size() << " after grid simplification instead" << std::endl;
  std::ofstream fout2(fn + "_simplified.pwn");
  for (const auto& p : pts3d)
    fout2 << p.first << " 0 " << p.second << " 0" << std::endl;

  fout2.close();

  ps.resize(pts3d.size());
  for (std::size_t i = 0;i<pts3d.size();i++) {
    ps[i].first = Point_2(pts3d[i].first.x(), pts3d[i].first.y());
    ps[i].second = pts3d[i].second;
  }

  Point_map point_map(CGAL::make_random_access_property_map(ps));
  Normal_map normal_map(CGAL::make_random_access_property_map(ps));

  // Default parameter values for the data file buildings_outline.xyz.
  const FT          sphere_radius   = FT(0.6);
  const FT          max_distance    = 0.4;
  const std::size_t min_region_size = 100;

  // Create instances of the classes Neighbor_query and Region_type.
  Neighbor_query neighbor_query(
    boost::irange<std::size_t>(0,ps.size()),
    CGAL::parameters::sphere_radius(sphere_radius)
                     .point_map(point_map));

  Region_type region_type(
    CGAL::parameters::
    maximum_distance(max_distance).
    cosine_of_maximum_angle(0).
    minimum_region_size(min_region_size).
    normal_map(normal_map).
    point_map(point_map));

  // Create an instance of the region growing class.
  Region_growing region_growing(
    boost::irange<std::size_t>(0,ps.size()), neighbor_query, region_type);

  // Run the algorithm.
  std::vector<typename Region_growing::Primitive_and_region> regions;
  region_growing.detect(std::back_inserter(regions));
  std::cout << "* number of found lines: " << regions.size() << std::endl;
  assert(!is_default_input || regions.size() == 72);

  // Save regions to a file.
  const std::string fullpath = (argc > 2 ? argv[2] : fn + "_lines.ply");
  utils::save_point_regions_2<Kernel, std::vector<typename Region_growing::Primitive_and_region>, Point_map>(
    regions, fullpath, point_map);
  return EXIT_SUCCESS;
}
