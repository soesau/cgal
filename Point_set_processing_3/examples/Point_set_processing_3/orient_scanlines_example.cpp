
#define CGAL_SCANORIENT_DUMP_RANDOM_SCANLINES

#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/read_las_points.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/scanline_orient_normals.h>

using Kernel = CGAL::Simple_cartesian<double>;
using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Point_with_info = std::tuple<Point_3, Vector_3, float, short, double, float>;
using Point_map = CGAL::Nth_of_tuple_property_map<0, Point_with_info>;
using Normal_map = CGAL::Nth_of_tuple_property_map<1, Point_with_info>;
using Scan_angle_map = CGAL::Nth_of_tuple_property_map<2, Point_with_info>;
using Scanline_id_map = CGAL::Nth_of_tuple_property_map<3, Point_with_info>;
using GPStime_map = CGAL::Nth_of_tuple_property_map<4, Point_with_info>;
/*

void dump(const char* filename, const std::vector<Point_with_info>& points)
{
  std::ofstream ofile(filename, std::ios::binary);
  CGAL::IO::set_binary_mode(ofile);
  CGAL::IO::write_PLY
  (ofile, points,
    CGAL::parameters::point_map(Point_map()).
    normal_map(Normal_map()));

}*/

template<class PointRange>
void dump(const char* filename, PointRange points)
{
  std::ofstream ofile(filename, std::ios::binary);
  CGAL::IO::set_binary_mode(ofile);
  CGAL::IO::write_PLY
  (ofile, points,
    CGAL::parameters::point_map(Point_map()).
    normal_map(Normal_map()));

}

int main (int argc, char** argv)
{
  std::string fname(argc > 1 ? argv[1] : "../data/urban.las");
  fname = "C:/Data/ESRI/CGAL/completeCircle_RGBN_20201025_101808_022_101752_0_6601_4.las";

  std::vector<Point_with_info> points;

  std::cerr << "Reading input file " << fname << std::endl;
  std::ifstream ifile (fname, std::ios::binary);
  if (!ifile ||
      !CGAL::IO::read_LAS_with_properties
      (ifile, std::back_inserter (points),
       CGAL::IO::make_las_point_reader (Point_map()),
       std::make_pair (Scan_angle_map(),
                       CGAL::IO::LAS_property::Scan_angle()),
       std::make_pair (Scanline_id_map(),
                       CGAL::IO::LAS_property::Scan_direction_flag()),
       std::make_pair (GPStime_map(),
                       CGAL::IO::LAS_property::GPS_time())))
  {
    std::cerr << "Can't read " << fname << std::endl;
    return EXIT_FAILURE;
  }

  /*std::cerr << "Estimating normals" << std::endl;
  CGAL::jet_estimate_normals<CGAL::Parallel_if_available_tag>
    (points, 12,
     CGAL::parameters::point_map (Point_map()).
     normal_map (Normal_map()));*/

  std::cerr << "Orienting normals using scan angle and direction flag" << std::endl;
  CGAL::circular_scanline_orient_normals
    (points,
     CGAL::parameters::point_map (Point_map()).
     normal_map (Normal_map()).
     scan_angle_map (Scan_angle_map()).
     scanline_id_map (Scanline_id_map()).
     gpstime_id_map(GPStime_map()));
  dump("out_angle_and_flag.ply", points);

  return EXIT_SUCCESS;

  std::cerr << "Orienting normals using scan direction flag only" << std::endl;
  CGAL::scanline_orient_normals
    (points,
     CGAL::parameters::point_map (Point_map()).
     normal_map (Normal_map()).
      scanline_id_map(Scanline_id_map()).
      gpstime_id_map(GPStime_map()));
  dump("out_flag.ply", points);

  std::cerr << "Orienting normals using scan angle only" << std::endl;
  CGAL::scanline_orient_normals
    (points,
     CGAL::parameters::point_map (Point_map()).
     normal_map (Normal_map()).
     scan_angle_map(Scan_angle_map()).
     gpstime_id_map(GPStime_map()));
  dump("out_angle.ply", points);

  std::cerr << "Orienting normals using no additional info" << std::endl;
  CGAL::scanline_orient_normals
    (points,
     CGAL::parameters::point_map (Point_map()).
     normal_map(Normal_map()).
     gpstime_id_map(GPStime_map()));
  dump("out_nothing.ply", points);

  return EXIT_SUCCESS;
}
