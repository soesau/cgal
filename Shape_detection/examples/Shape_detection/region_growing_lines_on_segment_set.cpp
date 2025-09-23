#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Segment_set.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/IO/WKT.h>
#include <CGAL/Frechet_distance.h>

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

std::vector<double> eps = { 0.2, 0.3, 0.4, 0.5 };

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

void export_WKT(const std::vector<Point_2> &segment, const std::string &filename) {
  if (segment.empty())
    return;

  std::ofstream fout(filename + ".wkt.txt", CGAL::IO::ASCII);

  fout << "POLYGON((" << segment[0].x() << ", " << segment[0].y();

  for (std::size_t i = 1; i < segment.size(); i++)
    fout << ", " << segment[i].x() << ", " << segment[i].y();

  fout << "))" << std::endl;

  fout.close();
}

void detect(const Polygon_2 &in, const std::string& filename) {
  Segment_range segments;
  segments.reserve(in.edges().size());

  for (Segment_2 s : in.edges())
    segments.push_back(s);

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

FT smallest_edge_length(const Polygon_2 &in) {
  FT min_length = (std::numeric_limits<double>::max)();
  for (Segment_2 s : in.edges())
    min_length = (std::min)(min_length, s.squared_length());
  return CGAL::sqrt(min_length);
}

FT manhattan_length(const Segment_2 &s) {
  return CGAL::abs(s.source().x() - s.target().x()) + CGAL::abs(s.source().y() - s.target().y());
}

void smooth(std::vector<Point_2> &polyline) {
  Vector_2 prev = polyline[0] - CGAL::ORIGIN;

  for (std::size_t i = 1; i <= polyline.size() - 2; ++i)
  {
    Vector_2 curr = polyline[i] - CGAL::ORIGIN;
    Vector_2 next = polyline[i + 1] - CGAL::ORIGIN;

    polyline[i] = CGAL::ORIGIN + (prev + 2 * curr + next) / 4;
    prev = curr;
  }
}

void preserve_long_segments(const Polygon_2 &in, const std::string &fn, bool smoothing) {
  typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> point_t;
  typedef boost::geometry::model::ring<point_t, false, true> ring_t;
  typedef boost::geometry::model::linestring<point_t> polyline_t;

  std::string filename = fn;
  if (smoothing)
    filename += "_lss_";
  else
    filename += "_ls_";

  FT edge_length = smallest_edge_length(in);

  bool first = false;
  std::size_t idx = 0;

  static int counter = 0;


  for (double e : eps) {
    Polygon_2::Edge_const_circulator start = in.edges_circulator();
    Polygon_2::Edge_const_circulator curr = start;
    std::vector<Point_2> out;
    // Search first long edge.
    std::size_t start_idx = 0;
    bool found = false;
    do {
      if (manhattan_length(*curr) > 4) {
        found = true;
        break;
      }
      start_idx++;
    } while (++curr != start);

    if (!found) { // use DP for full polygon
      ring_t polygon;
      polygon.reserve(in.vertices().size() + 1);
      for (const Point_2& p : in.vertices())
        polygon.push_back(point_t(p.x(), p.y()));

      polygon.push_back(point_t(in.vertices_begin()->x(), in.vertices_begin()->y()));

      ring_t simplified;
      boost::geometry::simplify(polygon, simplified, e);

      out.reserve(simplified.size());

      for (const point_t& p : simplified)
        out.push_back(Point_2(boost::geometry::get<0>(p), boost::geometry::get<1>(p)));
    }
    else {
      start = curr;
      std::vector<Point_2> segment;
      out.reserve(in.vertices().size());
      segment.reserve(in.vertices().size());
      do {
        if (manhattan_length(*curr) > 6) {
          if (segment.empty())
            out.push_back(curr->source());
          else { // end of small segment, simplify it
            segment.push_back(curr->source());

            if (smoothing)
              smooth(segment);

            if (first) {
              //export_WKT(segment, filename + std::to_string(idx++) + "_" + std::to_string(segment.size() - 1));
              std::ofstream fout(filename + std::to_string(idx++) + "_" + std::to_string(segment.size() - 1) + ".polylines.txt", CGAL::IO::ASCII);

              fout << "2 " << segment[0].x() << " " << segment[0].y() << " 0 ";
              for (std::size_t i = 1; i < segment.size() - 1; i++)
                fout << segment[i].x() << " " << segment[i].y() << " 0\n2 " << segment[i].x() << " " << segment[i].y() << " 0 ";

              fout << segment.back().x() << " " << segment.back().y() << " 0" << std::endl;
            }

            polyline_t polygon;
            polygon.reserve(segment.size());
            for (const Point_2& p : segment)
              polygon.push_back(point_t(p.x(), p.y()));

            polyline_t simplified;
            boost::geometry::simplify(polygon, simplified, e);

            for (const point_t& p : simplified)
              out.push_back(Point_2(boost::geometry::get<0>(p), boost::geometry::get<1>(p)));

            segment.clear();
          }
        }
        else
          segment.push_back(curr->source());
      } while (++curr != start);

      // Still a segment to simplify?
      if (!segment.empty()) {
        segment.push_back(curr->source());

        if (smoothing)
          smooth(segment);

        if (first) {
          std::ofstream fout(filename + std::to_string(idx) + "_" + std::to_string(segment.size() - 1) + ".polylines.txt", CGAL::IO::ASCII);

          fout << "2 " << segment[0].x() << " " << segment[0].y() << " 0 ";
          for (std::size_t i = 1; i < segment.size() - 1; i++)
            fout << segment[i].x() << " " << segment[i].y() << " 0\n2 " << segment[i].x() << " " << segment[i].y() << " 0 ";

          fout << segment.back().x() << " " << segment.back().y() << " 0" << std::endl;
        }

        polyline_t polygon;
        polygon.reserve(segment.size());
        for (const Point_2& p : segment)
          polygon.push_back(point_t(p.x(), p.y()));

        polyline_t simplified;
        boost::geometry::simplify(polygon, simplified, e);

        for (const point_t& p : simplified)
          out.push_back(Point_2(boost::geometry::get<0>(p), boost::geometry::get<1>(p)));
      }
      else out.push_back(curr->source());
    }

    std::vector<Point_2> poly;
    poly.reserve(in.vertices().size());
    if (start_idx == 0)
      for (const Point_2& p : in.vertices())
        poly.push_back(p);
    else {
      for (auto it = in.vertices_begin() + start_idx;it != in.vertices_end();it++)
        poly.push_back(*it);

      for (auto it = in.vertices_begin(); it != in.vertices_begin() + start_idx; it++)
        poly.push_back(*it);
    }
    poly.push_back(*(in.vertices_begin() + start_idx));

    if (counter == 24) {
      std::string out1, out2;
      out1 = (smoothing) ? "poly1.smoothed_" : "poly1_";
      out2 = (smoothing) ? "poly2.smoothed_" : "poly2_";

      out1 += std::to_string(counter);
      out2 += std::to_string(counter);

      std::ofstream fout(out1 + ".polylines.txt", CGAL::IO::ASCII);

      fout << "2 " << poly[0].x() << " " << poly[0].y() << " 0 ";
      for (std::size_t i = 1; i < poly.size() - 1; i++)
        fout << poly[i].x() << " " << poly[i].y() << " 0\n2 " << poly[i].x() << " " << poly[i].y() << " 0 ";

      fout << poly.back().x() << " " << poly.back().y() << " 0" << std::endl;

      fout.close();

      std::ofstream fout2(out2 + ".polylines.txt", CGAL::IO::ASCII);

      fout2 << "2 " << out[0].x() << " " << out[0].y() << " 0 ";
      for (std::size_t i = 1; i < out.size() - 1; i++)
        fout2 << out[i].x() << " " << out[i].y() << " 0\n2 " << out[i].x() << " " << out[i].y() << " 0 ";

      fout2 << out.back().x() << " " << out.back().y() << " 0" << std::endl;

      fout2.close();
    }

    counter++;

    std::pair<double, double> res = CGAL::bounded_error_Frechet_distance(poly, out, 0.000001);

    //export_WKT(out, filename + std::to_string(idx++) + "_" + std::to_string(out.size() - 1));
    std::ofstream fout(filename + std::to_string(e) + "_" + std::to_string(res.second) + "_" + std::to_string(out.size() - 1) + ".polylines.txt", CGAL::IO::ASCII);

    fout << "2 " << out[0].x() << " " << out[0].y() << " 0 ";
    for (std::size_t i = 1; i < out.size() - 1; i++)
      fout << out[i].x() << " " << out[i].y() << " 0\n2 " << out[i].x() << " " << out[i].y() << " 0 ";

    fout << out.back().x() << " " << out.back().y() << " 0" << std::endl;
    fout.close();
    first = false;
  }
}

int main(int argc, char *argv[]) {
  for (const std::string &filename : files) {
    Polygon_2 p;
    std::string path = "data/" + filename + ".txt";
    std::ifstream file(path, CGAL::IO::ASCII);

    std::cout << path << std::endl;

    if (!CGAL::IO::read_polygon_WKT(file, p)) {
      std::cout << "Error: File could not be found or read!" << std::endl;
      continue;
    }

    std::string resultpath = "results/" + filename;

    Segment_range segments;
    segments.reserve(p.edges().size());

    for (Segment_2 s : p.edges())
      segments.push_back(s);

    std::cout << segments.size() << " segments in polygon" << std::endl;

    std::ofstream out(resultpath + ".polylines.txt");
    for (const Segment_2& s : segments)
      out << "2 " << s.source().x() << " " << s.source().y() << " 0 " << s.target().x() << " " << s.target().y() << " 0" << std::endl;
    out.close();

    preserve_long_segments(p, resultpath, false);
    preserve_long_segments(p, resultpath, true);

    //simplify(p, resultpath);
    //detect(p, resultpath);
  }

  return EXIT_SUCCESS;
}
