#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Segment_set.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/IO/WKT.h>

#include <typeinfo>

//#include <boost/geometry/strategies/agnostic/simplify_douglas_peucker.hpp>

#include <boost/geometry/algorithms/simplify.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include "include/utils.h"

// Typedefs.
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;

using Point_2 = typename Kernel::Point_2;
using Point_3 = typename Kernel::Point_3;
using Polygon_2 = CGAL::Polygon_2<Kernel>;
using Vector_2 = typename Kernel::Vector_2;
using Segment_2 = typename Kernel::Segment_2;
using Segment_3 = typename Kernel::Segment_3;
using FT = typename Kernel::FT;

using Segment_range = std::vector<Segment_2>;
using Item = typename Segment_range::const_iterator;

using Segment_map = CGAL::Dereference_property_map<const Segment_2, Item>;

using Region_type = CGAL::Shape_detection::Segment_set::Least_squares_line_fit_region<Kernel, Item, Segment_map>;

struct Neighbor_query {
  Neighbor_query(Segment_range &range) : begin(range.begin()), end(range.end()) {}
  Item begin, end;

  std::map<Item, std::vector<Item> > m_neighbors;

  void operator()(
    Item query, std::vector<Item>& neighbors) const {
    assert(query != end);
    if (query == begin)
      neighbors.push_back(end - 1);
    else neighbors.push_back(query - 1);

    if (query + 1 == end)
      neighbors.push_back(begin);
    else
      neighbors.push_back(query + 1);

    assert(neighbors[0] != end);
    assert(neighbors[1] != end);
  }
};
using Region_growing = CGAL::Shape_detection::Region_growing<Neighbor_query, Region_type>;

std::string files[] = {
"20250512_staircase_holes_0",
"20250605_diagonal_staircase_0",
"20250605_diagonal_staircase_1",
"20250605_diagonal_staircase_2",
"20250605_diagonal_staircase_3",
"20250605_diagonal_staircase_4",
"20250605_diagonal_staircase_5",
"20250605_diagonal_staircase_6",
"20250605_smoothing_issues_0",
"case2_0",
"case2_1",
"case2_10",
"case2_11",
"case2_12",
"case2_13",
"case2_14",
"case2_15",
"case2_16",
"case2_17",
"case2_18",
"case2_19",
"case2_2",
"case2_20",
"case2_21",
"case2_22",
"case2_23",
"case2_24",
"case2_25",
"case2_3",
"case2_4",
"case2_5",
"case2_6",
"case2_7",
"case2_8",
"case2_9"
};

std::vector<double> eps = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };

template<typename Primitive_and_region_range, typename SegmentMap>
std::vector<Segment_2> get_segments(const Primitive_and_region_range &regions, SegmentMap map) {
  std::vector<Segment_2> segments;
  for (const auto &r : regions) {
    FT low = (std::numeric_limits<double>::max)();
    FT high = -low;
    typename Primitive_and_region_range::value_type::first_type p = r.first;
    Point_2 origin = p.point(0);
    Vector_2 dir = p.to_vector();

    for (const auto &i : r.second) {
      Segment_2 seg = get(map, i);
      FT proj = (seg.source() - origin) * dir;
      low = (std::min)(low, proj);
      high = (std::max)(high, proj);

      proj = (seg.target() - origin) * dir;
      low = (std::min)(low, proj);
      high = (std::max)(high, proj);
    }

    if (!r.second.empty())
      segments.push_back(Segment_2(origin + low * dir, origin + high * dir));
  }
  return segments;
}

void simplify(const Polygon_2 &in, const std::string &filename) {
  typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> point_t;
  typedef boost::geometry::model::ring<point_t, false, true> ring_t;

  ring_t polygon;

  for (Segment_2 s : in.edges())
    polygon.push_back(point_t(s.source().x(), s.source().y()));

  polygon.push_back(point_t(in.edges().begin()->source().x(), in.edges().begin()->source().y()));

  for (double e : eps) {
    ring_t simplified;

    boost::geometry::simplify(polygon, simplified, e);
    std::cout << e << " " << int(simplified.size()) << std::endl;

    std::ofstream fout(filename + "_dp_" + std::to_string(e) + "_" + std::to_string(simplified.size()) + ".polylines.txt", CGAL::IO::ASCII);
    fout << int(simplified.size());

    for (const point_t& p : simplified)
      fout << " " << boost::geometry::get<0>(p) << " " << boost::geometry::get<1>(p) << " 0";
    fout << std::endl;

    fout.close();
  }
}

void detect(const Polygon_2 &in, const std::string& filename) {
  Segment_range segments;
  segments.reserve(in.edges().size());

  for (Segment_2 s : in.edges())
    segments.push_back(s);

  std::cout << segments.size() << " segments in polygon" << std::endl;

  std::ofstream out(filename + ".polylines.txt");
  for (const Segment_2& s : segments)
    out << "2 " << s.source().x() << " " << s.source().y() << " 0 " << s.target().x() << " " << s.target().y() << " 0" << std::endl;
  out.close();

  Neighbor_query neighbor_query(segments);
  const FT          cos_max_angle = FT(0);
  const std::size_t min_region_size = 1;

  using Sorting = CGAL::Shape_detection::Segment_set::Least_squares_line_fit_sorting<Kernel, Item, Neighbor_query, Segment_map>;

  Sorting sorting(
    segments, neighbor_query);
  sorting.sort();

  for (double e : eps) {
    Region_type region_type(
      CGAL::parameters::
      maximum_distance(e).
      cosine_of_maximum_angle(cos_max_angle).
      minimum_region_size(min_region_size));

    std::vector< std::pair< typename Region_type::Primitive, std::vector<Item> > > regions;
    Region_growing region_growing(
      segments, neighbor_query, region_type);
    region_growing.detect(std::back_inserter(regions));

    std::cout << regions.size() << " segments detected" << std::endl;

    std::vector<Segment_2> region_segments = get_segments(regions, Segment_map());

    std::ofstream fout(filename + "_sd_" + std::to_string(e) + "_" + std::to_string(region_segments.size()) + ".polylines.txt", CGAL::IO::ASCII);
    for (const Segment_2& s : region_segments)
      fout << "2  " << s.source().x() << " " << s.source().y() << " 0 " << s.target().x() << " " << s.target().y() << " 0" << std::endl;

    fout.close();
  }
}

void detect_staircases(const Polygon_2 &in, ) {

}

int main(int argc, char *argv[]) {
  for (std::string filename : files) {
    Polygon_2 p;
    std::string path = "data/" + filename + ".txt";
    std::ifstream file(path, CGAL::IO::ASCII);

    std::cout << path << std::endl;

    if (!CGAL::IO::read_polygon_WKT(file, p)) {
      std::cout << "Error: File could not be found or read!" << std::endl;
      continue;
    }

    simplify(p, "results/" + filename);
    detect(p, "results/" + filename);
  }

  return EXIT_SUCCESS;
}
