#include <CGAL/Simple_cartesian.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Point_set.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Optimal_transportation_reconstruction_2.h>

#include "include/utils.h"
#include <CGAL/IO/read_points.h>
#include <boost/range/irange.hpp>
#include <CGAL/Timer.h>

// Typedefs.
using Kernel   = CGAL::Simple_cartesian<double>;
using FT       = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Point_2  = typename Kernel::Point_2;
using Vector_2 = typename Kernel::Vector_2;
using Vector_3 = typename Kernel::Vector_3;
using Segment_2 = typename Kernel::Segment_2;
using Line_2 = typename Kernel::Line_2;
using Ray_2 = typename Kernel::Ray_2;

using Point_with_normal = std::pair<Point_2, Vector_2>;
using Point_set_2 = std::vector<Point_with_normal>;
using Points_3 = std::vector<std::pair<Point_3, Vector_3>>;

// we use Compose_property_map as the property maps are expected to operate on the item type
// std::size_t passed as parameter to Sphere_neighbor_query and Least_squares_line_fit_region
using Point_map_2      = CGAL::Compose_property_map<CGAL::Random_access_property_map<Point_set_2>,
                                                  CGAL::First_of_pair_property_map<Point_with_normal> >;
using Normal_map_2     = CGAL::Compose_property_map<CGAL::Random_access_property_map<Point_set_2>,
                                                  CGAL::Second_of_pair_property_map<Point_with_normal> >;

using Neighbor_query_lines = CGAL::Shape_detection::Point_set::Sphere_neighbor_query<Kernel, std::size_t, Point_map_2>;
using Line_region    = CGAL::Shape_detection::Point_set::Least_squares_line_fit_region<Kernel, std::size_t, Point_map_2, Normal_map_2>;
using RG_lines = CGAL::Shape_detection::Region_growing<Neighbor_query_lines, Line_region>;

//using Point_map_3 = CGAL::Compose_property_map<CGAL::Random_access_property_map<Points_3>, CGAL::First_of_pair_property_map<std::pair<Point_3, Vector_3>> >;
//using Normal_map_3 = CGAL::Compose_property_map<CGAL::Random_access_property_map<Points_3>, CGAL::Second_of_pair_property_map<std::pair<Point_3, Vector_3>> >;

using Item = typename Points_3::const_iterator;
using Deref_map = CGAL::Dereference_property_map<const std::pair<Point_3, Vector_3>, Item>;
using Point_map_3 = CGAL::Compose_property_map<Deref_map, CGAL::First_of_pair_property_map<std::pair<Point_3, Vector_3>> >;
using Normal_map_3 = CGAL::Compose_property_map<Deref_map, CGAL::Second_of_pair_property_map<std::pair<Point_3, Vector_3>> >;

using Plane_region = CGAL::Shape_detection::Point_set::Least_squares_plane_fit_region<Kernel, Item, Point_map_3, Normal_map_3>;
using Neighbor_query_3 = CGAL::Shape_detection::Point_set::Sphere_neighbor_query<Kernel, Item, Point_map_3>;
using Sorting = CGAL::Shape_detection::Point_set::Least_squares_plane_fit_sorting<Kernel, Item, Neighbor_query_3, Point_map_3, Normal_map_3>;
using RG_planes = CGAL::Shape_detection::Region_growing<Neighbor_query_3, Plane_region>;
using Point_inserter = utils::Insert_point_colored_by_region_index<Item, Points_3, Point_map_3, Kernel::Plane_3>;

void detect_walls(std::vector<std::pair<Point_3, Vector_3>> &input, FT eps, std::size_t min_region_size, const Vector_2 &wall_dir, const std::string &filename) {
  Plane_region region_type(
    CGAL::parameters::
    maximum_distance(eps).
    cosine_of_maximum_angle(0).
    minimum_region_size(min_region_size));

  FT l = wall_dir * wall_dir;
  l = CGAL::sqrt(l);

  region_type.add_filter([=](const Kernel::Plane_3 &p) -> bool {return ::abs(Vector_3(wall_dir.x()/l, wall_dir.y()/l, 0) * p.orthogonal_vector()) < 0.15;});

  // Create instances of the classes Neighbor_query and Region_type.
  Neighbor_query_3 neighbor_query(input, CGAL::parameters::sphere_radius(0.1));

  Sorting sorting(input, neighbor_query);
  sorting.sort();

  std::vector<typename RG_planes::Primitive_and_region> regions;
  RG_planes region_growing(input, sorting.ordered(), neighbor_query, region_type);
  region_growing.detect(std::back_inserter(regions));

  std::cout << regions.size() << " detected segments" << std::endl;

  utils::save_point_regions_3<Kernel, std::vector<typename RG_planes::Primitive_and_region>, CGAL::First_of_pair_property_map<std::pair<Point_3, Vector_3>>>(regions, filename, CGAL::First_of_pair_property_map<std::pair<Point_3, Vector_3>>());
}

template<typename Item, typename PointMap>
Segment_2 get_segment(const Line_2 &l, std::vector<Item> points, PointMap pmap) {
  FT minp = (std::numeric_limits<float>::max)();
  FT maxp = -minp;

  for (const Item &i : points) {
    Point_2 p = get(pmap, i);
    FT proj = (p - l.point()) * l.to_vector();
    minp = (std::min<FT>)(minp, proj);
    maxp = (std::max<FT>)(maxp, proj);
  }

  return Segment_2(l.point(minp), l.point(maxp));
}

std::list<Segment_2>::const_iterator find_closest(const Segment_2 &s, const std::list<Segment_2> &segments, bool source, FT &dist) {
  dist = (std::numeric_limits<FT>::max)();
  std::list<Segment_2>::const_iterator best = segments.end();

  Point_2 p = (source) ? s.source() : s.target();

  for (auto it = segments.begin();it!=segments.end();it++) {
    FT d = CGAL::squared_distance(p, *it);
    if (d < dist) {
      dist = d;
      best = it;
    }
  }

  return best;
}

template<typename Line_Regions, typename PointMap>
void extract_polyline(const std::vector<Line_Regions> &lines, std::vector<std::vector<Point_2>> &polylines, PointMap pmap) {
  if (lines.empty()) {
    polylines.clear();
    return;
  }

  std::list<Segment_2> segments;

  for (const auto &l : lines)
    segments.push_back(get_segment(l.first, l.second, pmap));

  static int cnt = 0;

  for (const Segment_2 &s : segments) {
    std::ofstream fout(std::to_string(cnt++) + ".polylines.txt");
    fout << "2 " << s.source() << " 0 " << s.target() << " 0" << std::endl;
    fout.close();
  }

  return;

  // How to connect segments? In current data, just connecting the closest ends will do the job.

  if (segments.size() == 1) {
    polylines.resize(1);
    polylines[0].clear();
    polylines[0].push_back(segments.front().source());
    polylines[0].push_back(segments.front().target());
    return;
  }

  while (!segments.empty()) {
    std::vector<Point_2> polyline;
    Segment_2 s = segments.front();
    polyline.push_back(s.source());
    polyline.push_back(s.target());
    segments.pop_front();

    FT dist;
    std::list<Segment_2>::const_iterator it = find_closest(s, segments, false, dist);

    // When the gap is too big, start next polyline.
    if (dist > 1.5)
      continue;

    if (CGAL::do_intersect(s, *it)) {
      auto res = CGAL::intersection(s, *it);
      polyline.pop_back();
      if (std::get_if<Segment_2>(&*res) != nullptr) {
        // Segments are collinear and overlap -> merge
        Vector_2 dir = s.target() - s.source();
        FT ps = (it->target() - s.source()) * dir;
        FT pt = (it->source() - s.source()) * dir;

        polyline.pop_back();

        if (ps > pt)
          polyline.push_back(it->source());
        else
          polyline.push_back(it->target());
      }
      else {
        const Point_2* p = std::get_if<Point_2>(&*res);
        polyline.pop_back();
        polyline.push_back(*p);

        // Which part of the segment is to keep? Let's take the longer one for now.
        if (CGAL::squared_distance(*p, it->source()) > CGAL::squared_distance(*p, it->target()))
          polyline.push_back(it->source());
        else
          polyline.push_back(it->target());
      }
    }
    else if (dist < 1.5) {// are they close enough to connect? (wall segment size is roughly 1m, so include tolerance for one missing segment)
      // Snapping by prolonging towards intersection? does not work if segments are (almost) parallel
      auto res = CGAL::intersection(Ray_2(s.source(), s.target()), it->supporting_line());
      if (res) {
        if (std::get_if<Ray_2>(&*res) != nullptr) {
          Vector_2 dir = s.target() - s.source();
          FT ps = (it->target() - s.source()) * dir;
          FT pt = (it->source() - s.source()) * dir;

          polyline.pop_back();

          if (ps > pt)
            polyline.push_back(it->source());
          else
            polyline.push_back(it->target());
        }
        else {
          const Point_2 *p = std::get_if<Point_2>(&*res);
          // too far away?
          if (CGAL::squared_distance(polyline.back(), *p) > (dist * dist)) {

          }
          polyline.pop_back();
          polyline.push_back(*p);
          Vector_2 dir = it->target() - *p;
          if (CGAL::abs((it->source() - *p) * dir) > 1.0)
            polyline.push_back(it->source());
          else
            polyline.push_back(it->target());
        }
      }
      else; //
    }

    segments.erase(it);

    s = Segment_2(polyline[polyline.size() - 2], polyline.back());
    polylines.emplace_back(polyline);
  }
}

template<typename Points>
void optimal_transport(const Points& points, const std::string &filename, std::size_t target) {
  using Otr_2 = CGAL::Optimal_transportation_reconstruction_2<Kernel>;
  Otr_2 otr2(points);
  //otr2.run_under_wasserstein_tolerance(1.25);
  otr2.run_until(target);

  std::vector<Point_2> vertices;
  std::vector<size_t> isolated_vertices;
  std::vector<std::pair<size_t, size_t> > edges;

  otr2.indexed_output(
    std::back_inserter(vertices),
    std::back_inserter(isolated_vertices),
    std::back_inserter(edges));

  std::ofstream output_obj(filename);
  {
//     std::vector<Point_2>::iterator vit;
//     for (vit = vertices.begin(); vit != vertices.end(); vit++) {
//       output_obj << "2 " << vit->x() << " " << vit->y() << " " << 0 << "\n";
//     }

    std::vector<std::pair<size_t, size_t>>::iterator eit;
    for (eit = edges.begin(); eit != edges.end(); eit++) {
      const Point_2& u = vertices[eit->first];
      const Point_2& v = vertices[eit->second];
      output_obj << "2 " << u.x() << " " << u.y() << " 0 " << v.x() << " " << v.y() << " 0" << std::endl;
    }
  }
  output_obj.close();
}


int main(int argc, char *argv[]) {

  // Load xyz data either from a local folder or a user-provided file.
  const bool is_default_input = argc > 1 ? false : true;
  std::string fn;
  //fn = "C:/data/ebp bund/0019 - Kostheim-1_utm zone32 6stellig_mbes";
  //fn = "C:/data/ebp bund/0005 - Muendung-Mainz-2-BB-utm zone32 6stellig_mbes";
  fn = "C:/data/ebp bund/0003 - Muendung-Mainz-9-SB-utm zone32 6stellig_mbes";
  //std::ifstream in(is_default_input ? "C:/data/ebp bund/.pwn" : argv[1]);
  //std::ifstream in(is_default_input ? "C:/data/ebp bund/.pwn" : argv[1]);

  std::vector<std::pair<Point_3, Vector_3>> pts3d;

/*
  if (!CGAL::IO::read_points(fn + "wall-1.xyz", std::back_inserter(pts3d), CGAL::parameters::point_map(CGAL::First_of_pair_property_map<std::pair<Point_3, Vector_3>>()).normal_map(CGAL::Second_of_pair_property_map<std::pair<Point_3, Vector_3>>()))) {
    std::cout << "loading input file failed" << std::endl;
    return 0;
  }

  detect_walls(pts3d, 0.03, 1200, fn + "wall-1-segments.ply");
  return -1;*/

  CGAL::Timer timer;
  timer.start();

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
    pts3d.push_back(std::make_pair(Point_3(a, b, c), Vector_3(d, e, f)));
  }
  in.close();
  std::cout << timer.time() << " s for importing data" << std::endl;
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
  timer.reset();
  const std::size_t min_points_per_cell = 20;
  const double grid_size = 0.1;

  CGAL::internal::Epsilon_point_set_3<Point_3, CGAL::Identity_property_map<Point_3>, CGAL::Tag_true> point_set(grid_size, CGAL::Identity_property_map<Point_3>(), min_points_per_cell);
  if (true) {
    for (const auto& [p, n] : pts3d)
      point_set.insert(Point_3(p.x(), p.y(), 0));

    std::vector<std::pair<Point_3, Vector_3>> pts3dfiltered;

    for (std::size_t i = 0; i < pts3d.size(); i++) {
      const Point_3& p = pts3d[i].first;
      auto it = point_set.find(Point_3(p.x(), p.y(), 0));
      assert(it != point_set.end());
      if (it->second >= min_points_per_cell) {
        it->second = 0;
        pts3dfiltered.push_back(pts3d[i]);
        const Vector_3& n = pts3d[i].second;
        FT l = CGAL::sqrt(n.x() * n.x() + n.y() * n.y());
        if (l == 0)
          l = 1;
        ps.push_back(
          std::make_pair(Point_2(p.x(), p.y()), Vector_2(n.x() / l, n.y() / l)));
      }
    }

//     std::cout << pts3d.size() << std::endl;
//     std::cout << pts3dfiltered.size() << std::endl;
//     std::cout << ps.size() << std::endl;
//
//     std::ofstream fout2(fn + "_my_filter.pwn");
//     for (const auto& p : pts3dfiltered)
//       fout2 << p.first << " " << p.second << std::endl;
//
//     fout2.close();
  }
  else {
    auto it = CGAL::grid_simplify_point_set(pts3d, grid_size, CGAL::parameters::min_points_per_cell(min_points_per_cell).point_map(CGAL::First_of_pair_property_map<std::pair<Point_3, Vector_3>>()));

    pts3d.resize(it - pts3d.begin());

    std::cout << pts3d.size() << " after grid simplification instead" << std::endl;
    std::ofstream fout2(fn + "_simplified.pwn");
    for (const auto& p : pts3d)
      fout2 << p.first << " 0 " << p.second << " 0" << std::endl;

    fout2.close();

    ps.resize(pts3d.size());
    for (std::size_t i = 0; i < pts3d.size(); i++) {
      ps[i].first = Point_2(pts3d[i].first.x(), pts3d[i].first.y());

      const Vector_3& n = pts3d[i].second;
      FT l = CGAL::sqrt(n.x() * n.x() + n.y() * n.y());
      if (l == 0)
        l = 1;

      ps[i].second = Vector_2(n.x() / l, n.y() / l);
    }
  }

  // Line detection

  // Default parameter values for the data file buildings_outline.xyz.
  const FT          sphere_radius   = FT(0.2);
  const FT          max_distance    = 0.4;
  const std::size_t min_region_size = 100;

  Point_map_2 point_map(CGAL::make_random_access_property_map(ps));
  Normal_map_2 normal_map(CGAL::make_random_access_property_map(ps));

  // Create instances of the classes Neighbor_query and Region_type.
  Neighbor_query_lines neighbor_query(
    boost::irange<std::size_t>(0,ps.size()),
    CGAL::parameters::sphere_radius(sphere_radius)
                     .point_map(point_map));

  Line_region region_type(
    CGAL::parameters::
    maximum_distance(max_distance).
    cosine_of_maximum_angle(0).
    minimum_region_size(min_region_size).
    normal_map(normal_map).
    point_map(point_map));

  // Create an instance of the region growing class.
  RG_lines rg_lines(
    boost::irange<std::size_t>(0,ps.size()), neighbor_query, region_type);

  // Run the algorithm.
  std::vector<typename RG_lines::Primitive_and_region> regions;
  rg_lines.detect(std::back_inserter(regions));
  std::cout << "* number of found lines: " << regions.size() << std::endl;
  assert(!is_default_input || regions.size() == 72);

  std::vector<Point_2> pts2d;
  for (const auto &r : regions)
    for (auto &i : r.second)
      pts2d.push_back(get(point_map, i));

  std::ofstream fout("pts2d.xyz");
  for (const Point_2 &p : pts2d)
    fout << p.x() << " " << p.y() << " 0\n";
  fout.close();

  optimal_transport(pts2d, "out-2.polylines.txt", 2);
  optimal_transport(pts2d, "out-3.polylines.txt", 3);
  optimal_transport(pts2d, "out-4.polylines.txt", 4);
  optimal_transport(pts2d, "out-5.polylines.txt", 5);
  optimal_transport(pts2d, "out-10.polylines.txt", 10);
  optimal_transport(pts2d, "out-30.polylines.txt", 30);
  optimal_transport(pts2d, "out-half.polylines.txt", pts2d.size() >> 1);

  // Save regions to a file.
  const std::string fullpath = (argc > 2 ? argv[2] : fn + "_lines.ply");
  utils::save_point_regions_2<Kernel, std::vector<typename RG_lines::Primitive_and_region>, Point_map_2>(regions, fullpath, point_map);
  std::vector<std::vector<Point_2> > polylines;
  extract_polyline<typename RG_lines::Primitive_and_region, Point_map_2>(regions, polylines, point_map);
  return 0;

  point_set.clear();

  // Backmapping to 3D
  std::size_t idx = 0;
  for (const RG_lines::Primitive_and_region &r : regions) {
    for(const auto &i : r.second) {
      const Point_2 &p = get(point_map, i);
      point_set[Point_3(p.x(), p.y(), 0)] = idx;
    }
    idx++;
  }

  std::vector<std::vector<std::pair<Point_3, Vector_3> > > line_regions_3d;
  line_regions_3d.resize(regions.size());

  for (std::size_t i = 0; i < pts3d.size(); i++) {
    const Point_3& p = pts3d[i].first;
    auto it = point_set.find(Point_3(p.x(), p.y(), 0));
    if (it == point_set.end())
      continue;
    assert (it->second < line_regions_3d.size());
    line_regions_3d[it->second].push_back(pts3d[i]);
  }

//   idx = 0;
//   for (const auto& r : line_regions_3d) {
//     std::ofstream fout(fn + "wall-" + std::to_string(idx) + ".xyz");
//     for (const auto& p : r)
//       fout << p.first << " " << p.second << std::endl;
//
//     fout.close();
//
//     idx++;
//   }

  for (std::size_t i = 0;i<(std::min<std::size_t>)(2, line_regions_3d.size());i++)
    detect_walls(line_regions_3d[i], 0.03, 1200, regions[i].first.to_vector(), fn + "wall-" + std::to_string(i) + "-segments.ply");

  std::cout << timer.time() << " s for processing" << std::endl;

  return EXIT_SUCCESS;
}
