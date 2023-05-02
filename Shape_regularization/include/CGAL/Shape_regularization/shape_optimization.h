#ifndef CGAL_SHAPE_OPTIMIZATION_H
#define CGAL_SHAPE_OPTIMIZATION_H

#include <utility>
#include <iomanip>
#include <unordered_map>
#include <boost/graph/graph_traits.hpp>
#include <CGAL/Default_diagonalize_traits.h>
#include <CGAL/Timer.h>

#include "../../../examples/Shape_regularization/include/utils.h"

const bool prune_merges = true;

namespace CGAL {
namespace internal {

// assemble covariance matrix from a triangle set
template < typename InputIterator, typename InputIterator2, typename K >
void assemble_covariance_matrix_3(InputIterator first, InputIterator beyond,
  InputIterator2 first_remove, InputIterator2 beyond_remove,
  typename Eigen_diagonalize_traits<typename K::FT, 3>::Covariance_matrix& covariance, // covariance matrix
  const typename K::Point_3& c, // centroid
  const K&,                    // kernel
  const typename K::Triangle_3*,// used for indirection
  const CGAL::Dimension_tag<2>&,
  const Eigen_diagonalize_traits<typename K::FT, 3>&)
{
  typedef typename K::FT          FT;
  typedef typename K::Triangle_3  Triangle;
  typedef typename Eigen::Matrix<FT, 3, 3> Matrix;

  // assemble covariance matrix as a semi-definite matrix.
  // Matrix numbering:
  // 0 1 2
  //   3 4
  //     5
  //Final combined covariance matrix for all triangles and their combined mass
  FT mass = 0.0;

  // assemble 2nd order moment about the origin.
  Matrix moment;
  moment << FT(1.0 / 12.0), FT(1.0 / 24.0), FT(1.0 / 24.0),
    FT(1.0 / 24.0), FT(1.0 / 12.0), FT(1.0 / 24.0),
    FT(1.0 / 24.0), FT(1.0 / 24.0), FT(1.0 / 12.0);

  for (InputIterator it = first; it != beyond; it++) {
    // Now for each triangle, construct the 2nd order moment about the origin.
    // assemble the transformation matrix.
    const Triangle& t = *it;

    // defined for convenience.
    Matrix transformation;
    transformation << t[0].x(), t[1].x(), t[2].x(),
      t[0].y(), t[1].y(), t[2].y(),
      t[0].z(), t[1].z(), t[2].z();

    FT area = CGAL::approximate_sqrt(t.squared_area());

    // skip zero measure primitives
    if (area == (FT)0.0)
      continue;

    // Find the 2nd order moment for the triangle wrt to the origin by an affine transformation.

    // Transform the standard 2nd order moment using the transformation matrix
    transformation = 2 * area * transformation * moment * transformation.transpose();

    // and add to covariance matrix
    covariance[0] += transformation(0, 0);
    covariance[1] += transformation(1, 0);
    covariance[2] += transformation(2, 0);
    covariance[3] += transformation(1, 1);
    covariance[4] += transformation(2, 1);
    covariance[5] += transformation(2, 2);

    mass += area;
  }

  for (InputIterator it = first_remove; it != beyond_remove; it++) {
    // Now for each triangle, construct the 2nd order moment about the origin.
    // assemble the transformation matrix.
    const Triangle& t = *it;

    // defined for convenience.
    Matrix transformation;
    transformation << t[0].x(), t[1].x(), t[2].x(),
      t[0].y(), t[1].y(), t[2].y(),
      t[0].z(), t[1].z(), t[2].z();

    FT area = CGAL::approximate_sqrt(t.squared_area());

    // skip zero measure primitives
    if (area == (FT)0.0)
      continue;

    // Find the 2nd order moment for the triangle wrt to the origin by an affine transformation.

    // Transform the standard 2nd order moment using the transformation matrix
    transformation = 2 * area * transformation * moment * transformation.transpose();

    // and add to covariance matrix
    covariance[0] -= transformation(0, 0);
    covariance[1] -= transformation(1, 0);
    covariance[2] -= transformation(2, 0);
    covariance[3] -= transformation(1, 1);
    covariance[4] -= transformation(2, 1);
    covariance[5] -= transformation(2, 2);

    mass -= area;
  }

  CGAL_assertion_msg(mass != FT(0), "Can't compute PCA of null measure.");

  // Translate the 2nd order moment calculated about the origin to
  // the center of mass to get the covariance.
  covariance[0] += -mass * (c.x() * c.x());
  covariance[1] += -mass * (c.x() * c.y());
  covariance[2] += -mass * (c.z() * c.x());
  covariance[3] += -mass * (c.y() * c.y());
  covariance[4] += -mass * (c.z() * c.y());
  covariance[5] += -mass * (c.z() * c.z());

}

// centroid for 3D triangle set with 2D tag
template < typename InputIterator, typename InputIterator2, typename K >
typename K::Point_3 centroid(InputIterator first, InputIterator beyond,
  InputIterator2 first_remove, InputIterator2 beyond_remove,
  const K&,
  const typename K::Triangle_3*,
  CGAL::Dimension_tag<2>)
{
  typedef typename K::FT       FT;
  typedef typename K::Vector_3 Vector;
  typedef typename K::Point_3  Point;
  typedef typename K::Triangle_3 Triangle;

  CGAL_precondition(first != beyond);

  Vector v = NULL_VECTOR;
  FT sum_areas = 0;
  for (InputIterator it = first; it != beyond; it++) {
    const Triangle& triangle = *it;
    FT unsigned_area = CGAL::approximate_sqrt(triangle.squared_area());
    Point c = K().construct_centroid_3_object()(triangle[0], triangle[1], triangle[2]);
    v = v + unsigned_area * (c - ORIGIN);
    sum_areas += unsigned_area;
  }

  for (InputIterator it = first_remove; it != beyond_remove; it++) {
    const Triangle& triangle = *it;
    FT unsigned_area = CGAL::approximate_sqrt(triangle.squared_area());
    Point c = K().construct_centroid_3_object()(triangle[0], triangle[1], triangle[2]);
    v = v - unsigned_area * (c - ORIGIN);
    sum_areas -= unsigned_area;
  }
  CGAL_assertion(sum_areas != 0.0);
  return ORIGIN + v / sum_areas;
} // end centroid of a 3D triangle set with 2D tag


// fits a plane to a 3D triangle set
template < typename InputIterator, typename InputIterator2, typename K, typename DiagonalizeTraits >
typename K::FT linear_least_squares_fitting_3(InputIterator first, InputIterator beyond,
  InputIterator2 first_remove, InputIterator2 beyond_remove,
  typename K::Plane_3& plane,   // best fit plane
  typename K::Point_3& c,       // centroid
  const typename K::Triangle_3*,  // used for indirection
  const K& k,                   // kernel
  const CGAL::Dimension_tag<2>& tag,
  const DiagonalizeTraits& diagonalize_traits)
{
  typedef typename K::Triangle_3  Triangle;

  // precondition: at least one element in the container.
  CGAL_precondition(first != beyond);

  // compute centroid
  c = centroid(first, beyond, first_remove, beyond_remove, K(), (Triangle*) nullptr, tag);

  // assemble covariance matrix
  typename DiagonalizeTraits::Covariance_matrix covariance = { { 0., 0., 0., 0., 0., 0. } };
  assemble_covariance_matrix_3(first, beyond, first_remove, beyond_remove, covariance, c, k, (Triangle*) nullptr, tag, diagonalize_traits);

  // compute fitting plane
  return fitting_plane_3(covariance, c, plane, k, diagonalize_traits);
} // end linear_least_squares_fitting_triangles_3


template<typename Kernel, typename Mesh, typename RegionMap, typename Primitive_and_region>
class Shape_optimization {
public:
  using FT = typename Kernel::FT;

private:
  using Face = typename boost::graph_traits<Mesh>::face_descriptor;
  using Vertex = typename boost::graph_traits<Mesh>::vertex_descriptor;
  using Halfedge = typename boost::graph_traits<Mesh>::halfedge_descriptor;
  using Primitive = typename Primitive_and_region::first_type;
  using Point_3 = typename Kernel::Point_3;
  using Triangle = typename Kernel::Triangle_3;
  using VPMap = typename boost::property_map<Mesh, vertex_point_t>::type;
  using Region_index_map = typename boost::property_map<Mesh, CGAL::dynamic_face_property_t<std::size_t> >::const_type;
  using Triangle_map = typename boost::property_map<Mesh, CGAL::dynamic_face_property_t<Triangle> >::const_type;

  enum Op {
    EXCLUSION,
    INCLUSION,
    MERGE,
    SPLIT,
    TRANSFER
  };

  struct Operation {
    Op op;
    FT dEnergy;
    std::size_t a, b;
    std::set<std::size_t> affected_regions;
    Operation(Op op, std::size_t a, FT dEnergy) : op(op), dEnergy(dEnergy), a(a), affected_regions{ a } {}
    Operation(Op op, std::size_t a, std::size_t b, FT dEnergy) : op(op), dEnergy(dEnergy), a(a), b(b), affected_regions{ a, b } {}
    Operation(Op op, std::set<std::size_t> regions, FT dEnergy) : op(op), dEnergy(dEnergy), affected_regions(regions) {}
  };

  struct Operation_order {
    bool operator()(const Operation& a, const Operation& b) {
      return a.dEnergy > b.dEnergy;
    }
  };

public:
  Shape_optimization(const Mesh& mesh, RegionMap& region_map, std::vector<Primitive_and_region>& input, FT epsilon, std::size_t min_region_size, FT w1 = 1.0, FT w2 = 1.0, FT w3 = 1.0)
    : mesh(mesh), region_map(region_map), input(input), epsilon(epsilon), min_region_size(min_region_size), vpmap(get(boost::vertex_point, mesh)), w1(w1), w2(w2), w3(w3), m_no_exclusion(false), m_no_split(false) {
    CGAL::Timer t;
    t.reset();
    t.start();
    regions.resize(input.size());
    primitives.resize(input.size());
    sum_distances.resize(input.size(), 0);
    double t1 = t.time();

    total_fidelity = 0;
    num_inliers = 0;
    centers.resize(input.size());
    for (size_t i = 0; i < input.size(); i++) {
      regions[i].insert(input[i].second.begin(), input[i].second.end());
      std::size_t new_outliers = 0;
      sum_distances[i] = fit(regions[i], primitives[i], centers[i], new_outliers, true, region_map);
      if (new_outliers)
        std::cout << i << ". region has epsilon violations in " << new_outliers << " cases" << std::endl;
      num_inliers += regions[i].size();
      total_fidelity += sum_distances[i];
    }
    double t2 = t.time();

    total_fidelity /= num_inliers;

    rms = get(CGAL::dynamic_face_property_t<std::size_t>(), mesh);
    trimap = get(CGAL::dynamic_face_property_t<Triangle>(), mesh);
    double t3 = t.time();
    std::cout << "allocating: " << (t1 * 1000) << std::endl;
    std::cout << "converting: " << ((t2 - t1) * 1000) << std::endl;
    std::cout << "regionmap: " << ((t3 - t2) * 1000) << std::endl;
  }

  void optimize(bool no_exclusion = false, bool no_split = false) {
    m_no_split = no_split;
    m_no_exclusion = no_exclusion;

    for (double& d : time_per_operation)
      d = 0;

    for (std::size_t& op : operation_count)
      op = 0;

    for (std::size_t i = 0; i < regions.size(); i++)
      eps_check(i);

    std::cout << "visualize the two regions for the merge and think about quick reject, angle alone is not sufficient." << std::endl;
    std::cout << "tbb parallelization will also help" << std::endl;
    std::cout << "Do not update queue if a very large region absorbs a tiny region" << std::endl;
    // But if this happens a lot, a refit may be useful. However, it should be done anyway during transfer

    std::cout << "reduce one region gains: " << w2 * (-1.0 / FT(mesh.num_faces() / min_region_size)) << std::endl;

    num_inliers = 0;
    for (std::size_t r = 0; r < regions.size(); r++) {
      if (!single_component(regions[r]))
        std::cout << r << " is not a connected region" << std::endl;
      num_inliers += regions[r].size();
    }

    std::cout << std::setprecision(10);

    CGAL::Timer t;
    t.reset();
    t.start();

    verify();

    fidelity(true);

    std::cout << "energy: " << energy() << " fidelity: " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;

    std::cout << "Removing epsilon violating inliers may cause recursion while refitting" << std::endl;
    std::cout << "Do a quick rejection in merge based on the primitive normals? May preserve small unwanted primitives?" << std::endl;
    /*t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(31, 115, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(108, 3212, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(108, 3707, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(109, 1911, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(97, 961, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(97, 1055, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(97, 1145, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(97, 1185, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(97, 1338, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(97, 1395, true);
    t.stop();
    std::cout << t.time() << std::endl;
    t.reset();
    t.start();
    merge(97, 1601, true);
    t.stop();
    std::cout << t.time() << std::endl;
    exit(2);*/

    perform_alternating_optimization();

    verify();

    input.resize(num_regions());
    std::size_t j = 0;
    for (std::size_t i = 0; i < regions.size(); i++) {
      if (regions[i].size() > 0) {
        input[j].first = primitives[i];
        input[j].second.clear();
        std::copy(regions[i].begin(), regions[i].end(), std::back_inserter(input[j].second));
        j++;
      }
    }

    t.stop();
    std::cout << "Processing time: " << t.time() * 1000 << "ms" << std::endl;

    std::cout << num_regions() << " regions" << std::endl;
    std::cout << (mesh.num_faces() - num_inliers) << " outliers" << std::endl;
    std::cout << "energy: " << energy() << " fidelity: " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;

    for (std::size_t i = 0; i < regions.size(); i++)
      eps_check(i);
  }

private:
  // Basic operations
  template<typename Primitive>
  FT dist(Face face, const Primitive& primitive) const {

    Halfedge h = mesh.halfedge(face);

    FT dist = 0;

    do {
      Point_3 v = get(vpmap, mesh.target(h));
      if (CGAL::abs(dist) < CGAL::abs((v - CGAL::ORIGIN) * primitive.orthogonal_vector() + primitive.d()))
        dist = (v - CGAL::ORIGIN) * primitive.orthogonal_vector() + primitive.d();
      h = mesh.next(h);
    } while (h != mesh.halfedge(face));

    return dist;
  }

  Point_3 center(Face face) {
    Halfedge h = mesh.halfedge(face);
    Point_3 c(0, 0, 0);

    do {
      c = c + (1.0/3.0) * (get(vpmap, mesh.target(h)) - CGAL::ORIGIN);
      h = mesh.next(h);
    } while (h != mesh.halfedge(face));

    return c;
  }

  template<typename FaceRange, typename Primitive, typename RegionMap = Region_index_map>
  FT fit(FaceRange& region, Primitive& p, Point_3 &centroid, std::size_t& new_outliers, bool remove_epsilon_violations = false, RegionMap map = Region_index_map()) {
    std::vector<Triangle> tris;
    tris.reserve(region.size());
    for (auto f : region) {
      Halfedge h = mesh.halfedge(f);
      Point_3 a = get(vpmap, mesh.target(h));
      Point_3 b = get(vpmap, mesh.target(mesh.next(h)));
      Point_3 c = get(vpmap, mesh.target(mesh.next(mesh.next(h))));
      tris.push_back(Triangle(a, b, c));
    }

    linear_least_squares_fitting_3(tris.begin(), tris.end(), p, centroid, CGAL::Dimension_tag<2>());

    if (remove_epsilon_violations)
      return dist_sum(region, p, new_outliers, map);
    else return dist_sum_const(region, p, new_outliers);
  }

  template<typename FaceRange, typename Primitive, typename FaceRange2 = std::vector<Face>, typename FaceRange3 = std::vector<Face>>
  FT fit2(const FaceRange& region, Primitive& p, std::size_t& new_outliers, const FaceRange2& add = std::vector<Face>(), const FaceRange3& remove = std::vector<Face>()) {
    std::vector<Triangle> tris, tris_remove;
    tris.reserve(region.size() + add.size());

    for (auto f : region) {
      Halfedge h = mesh.halfedge(f);
      Point_3 a = get(vpmap, mesh.target(h));
      Point_3 b = get(vpmap, mesh.target(mesh.next(h)));
      Point_3 c = get(vpmap, mesh.target(mesh.next(mesh.next(h))));
      tris.push_back(Triangle(a, b, c));
    }

    for (auto f : add) {
      Halfedge h = mesh.halfedge(f);
      Point_3 a = get(vpmap, mesh.target(h));
      Point_3 b = get(vpmap, mesh.target(mesh.next(h)));
      Point_3 c = get(vpmap, mesh.target(mesh.next(mesh.next(h))));
      tris.push_back(Triangle(a, b, c));
    }

    tris_remove.reserve(remove.size());
    for (auto f : remove) {
      Halfedge h = mesh.halfedge(f);
      Point_3 a = get(vpmap, mesh.target(h));
      Point_3 b = get(vpmap, mesh.target(mesh.next(h)));
      Point_3 c = get(vpmap, mesh.target(mesh.next(mesh.next(h))));
      tris_remove.push_back(Triangle(a, b, c));
    }

    linear_least_squares_fitting_3(tris.begin(), tris.end(), tris_remove.begin(), tris_remove.end(), p, Point_3(), &Triangle(), Kernel(), CGAL::Dimension_tag<2>(), CGAL::Eigen_diagonalize_traits<typename Kernel::FT, 3>());

    return dist_sum_const(region, p, new_outliers) + dist_sum_const(add, p, new_outliers) - dist_sum_const(remove, p);
  }

  template<typename FaceRange, typename Primitive, typename FaceRange2 = std::vector<Face>, typename FaceRange3 = std::vector<Face>>
  FT fit3(const FaceRange& region, Primitive& p, const FaceRange2& add = std::vector<Face>(), const FaceRange3& remove = std::vector<Face>()) {
    std::vector<Triangle> tris, tris_remove;
    tris.reserve(region.size() + add.size());

    for (auto f : region) {
      tris.push_back(get(trimap, f));
    }

    for (auto f : add) {
      tris.push_back(get(trimap, f));
    }

    tris_remove.reserve(remove.size());
    for (auto f : remove) {
      tris_remove.push_back(get(trimap, f));
    }

    linear_least_squares_fitting_3(tris.begin(), tris.end(), tris_remove.begin(), tris_remove.end(), p, Point_3(), &Triangle(), Kernel(), CGAL::Dimension_tag<2>(), CGAL::Eigen_diagonalize_traits<typename Kernel::FT, 3>());

    FT res = dist_sum(region, p) + dist_sum(add, p) - dist_sum(remove, p);
    return res;
  }

  // Energy calculations
  FT completeness() {
    return 1.0 - (num_inliers / FT(mesh.num_faces()));
  }

  FT simplicity() {
    return num_regions() / FT(mesh.num_faces() / min_region_size);
  }

  std::size_t num_regions() const {
    std::size_t num = 0;
    for (const auto &r : regions) {
      if (r.size() != 0)
        num++;
    }

    return num;
  }

  template<typename FaceRange, typename Primitive>
  FT dist_sum(FaceRange& region, const Primitive& p, std::size_t& new_outliers, bool remove_epsilon_violations = false) {
    FT region_fidelity = 0;
    std::vector<typename FaceRange::value_type> to_be_deleted;
    for (auto f : region) {
      FT d = CGAL::abs(dist(f, p));
      if (d > epsilon) {
        new_outliers++;
        if (remove_epsilon_violations) {
          put(region_map, f, std::size_t(-1));
          int check = get(region_map, f);
          to_be_deleted.push_back(f);
          num_inliers--;
        }
      }
      else
        region_fidelity += d / epsilon;
    }

    for (auto f : to_be_deleted)
      region.erase(f);

    return region_fidelity;
  }

  template<typename FaceRange, typename Primitive>
  FT dist_sum(FaceRange& region, const Primitive& p) {
    FT region_fidelity = 0;
    for (auto f : region) {
      FT d = CGAL::abs(dist(f, p));
      region_fidelity += d / epsilon;
    }

    return region_fidelity;
  }

  template<typename FaceRange, typename Primitive, typename RegionMap>
  FT dist_sum(FaceRange& region, const Primitive& p, std::size_t& new_outliers, RegionMap map) {
    FT region_fidelity = 0;
    std::vector<typename FaceRange::value_type> to_be_deleted;
    for (auto f : region) {
      FT d = CGAL::abs(dist(f, p));
      if (d > epsilon) {
        put(map, f, std::size_t(-1));
        to_be_deleted.push_back(f);
        new_outliers++;
      }
      else region_fidelity += d / epsilon;
    }

    for (auto f : to_be_deleted)
      region.erase(f);

    return region_fidelity;
  }

  template<typename FaceRange, typename Primitive>
  FT dist_sum_const(const FaceRange& region, const Primitive& p) const  {
    FT region_fidelity = 0;
    for (auto f : region) {
      FT d = CGAL::abs(dist(f, p));
      region_fidelity += d / epsilon;
    }

    return region_fidelity;
  }

  template<typename FaceRange, typename Primitive>
  FT dist_sum_const(const FaceRange& region, const Primitive& p, std::size_t& new_outliers) const {
    FT region_fidelity = 0;
    for (auto f : region) {
      FT d = CGAL::abs(dist(f, p));
      if (d <= epsilon)
        region_fidelity += d / epsilon;
      else new_outliers++;
    }

    return region_fidelity;
  }

  bool eps_check(std::size_t idx) const {
    std::size_t failed = 0;
    for (auto f : regions[idx]) {
      FT d = CGAL::abs(dist(f, primitives[idx]));
      if (d > epsilon)
        failed++;
    }

    if (failed) {
      std::cout << idx << ". region has " << failed << " epsilon violations out of " << regions[idx].size() << std::endl;
      return false;
    }

    return true;
  }

  FT fidelity(bool recalc = false) {
    total_fidelity = 0;

    if (recalc) {
      std::size_t inliers = 0;
      for (auto& r : sum_distances)
        r = 0;

      for (auto i : mesh.faces()) {
        int region_index = get(region_map, i);
        if (region_index == -1)
          continue;

        inliers++;

        FT d = CGAL::abs(dist(i, primitives[region_index]));
        sum_distances[region_index] += d / epsilon;
      }
      CGAL_assertion(inliers == num_inliers);
    }

    for (std::size_t i = 0; i < regions.size();i++) {
      total_fidelity += sum_distances[i];
    }

    total_fidelity /= num_inliers;

    return total_fidelity;
  }

  FT energy() {
    return w1 * fidelity() + w2 * simplicity() + w3 * completeness();
  }

  void save(std::string filename, int highlight = -1) {
    Mesh tmp = mesh;
    input.resize(num_regions());
    std::size_t j = 0;
    for (std::size_t i = 0; i < regions.size(); i++) {
      if (regions[i].size() > 0) {
        input[j].first = primitives[i];
        input[j].second.clear();
        std::copy(regions[i].begin(), regions[i].end(), std::back_inserter(input[j].second));
        j++;
      }
      else highlight--;
    }

    utils::save_polygon_mesh_regions(tmp, input, filename, highlight);
  }

  void perform_operations_loop() {
    const int width_delta = 12;
    std::cout << std::setprecision(8);
    while (true) {

      std::vector<bool> used(regions.size(), false);

      debug = true;// ((std::size_t(iteration / 10) * 10) == iteration);
      //       if (iteration > 1800)
            //debug = (iteration > 1300);

      if (iteration == 2186)
        debug = true;

      FT first = 1;

      std::set<size_t> changed_regions;
      while (true) {
        FT before = energy();
        FT pred;
        if (debug)
          std::cout << std::setw(5) << std::right << iteration;
        iteration++;

        std::set<std::size_t> regions_exclusion;
        FT d_exclusion = exclusion(true, regions_exclusion);

        std::set<std::size_t> regions_inclusion;
        FT d_inclusion = inclusion(true, regions_inclusion);

        std::size_t best_merge = -1;
        if (!merges.empty()) {
          for (std::size_t i = 0; i < merges.size(); i++) {
            if (!used[merges[i].a] && !used[merges[i].b])
              if (best_merge == -1)
                best_merge = i;
              else if (merges[best_merge].dEnergy > merges[i].dEnergy)
                best_merge = i;
          }
        }

        std::size_t best_split = -1;
        if (!splits.empty()) {
          for (std::size_t i = 1; i < splits.size(); i++) {
            if (!used[splits[i].a])
              if (best_split == -1)
                best_split = i;
              else if (splits[best_split].dEnergy > splits[i].dEnergy)
                best_split = i;
          }
        }

        std::string o;

        // Initialize the reference energy to recalculate
        if (first > 0) {
          first = (std::min<FT>)(d_exclusion, d_inclusion);
          if (best_merge != -1)
            first = (std::min<FT>)(first, merges[best_merge].dEnergy);
          if (best_split != -1)
            first = (std::min<FT>)(first, splits[best_split].dEnergy);
        }

        if (d_inclusion >= 0 && d_exclusion >= 0 && best_merge == -1 && best_split == -1)
          break;

        if (best_split == -1 && best_merge == -1) {
          if (d_inclusion <= d_exclusion) {
            if (d_inclusion > first * 0.5)
              break;
            inclusion(false, changed_regions);
            for (auto t : changed_regions)
              used[t] = true;
            pred = d_inclusion;
            if (debug)
              std::cout << "i            " << std::setw(width_delta) << std::left << d_inclusion << " ";
          }
          else {
            if (d_exclusion > first * 0.5)
              break;
            exclusion(false, changed_regions);
            for (auto t : changed_regions)
              used[t] = true;
            pred = d_exclusion;
            if (debug)
              std::cout << "e             " << std::setw(width_delta) << std::left << d_exclusion << " ";
          }
        }
        else if (best_merge != -1 && (best_split == -1 || merges[best_merge].dEnergy < splits[best_split].dEnergy)) {
          if (d_inclusion <= d_exclusion) {
            if (d_inclusion <= merges[best_merge].dEnergy) {
              if (d_inclusion > first * 0.5)
                break;
              inclusion(false, changed_regions);
              for (auto t : changed_regions)
                used[t] = true;
              pred = d_inclusion;
              if (debug)
                std::cout << "i             " << std::setw(width_delta) << std::left << d_inclusion << " ";
            }
            else {
              if (merges[best_merge].dEnergy > first * 0.5)
                break;
              merge(merges[best_merge].a, merges[best_merge].b, false);
              used[merges[best_merge].a] = true;
              used[merges[best_merge].b] = true;
              pred = merges[best_merge].dEnergy;
              changed_regions.insert(merges[best_merge].a);
              changed_regions.insert(merges[best_merge].b);
              if (debug)
                std::cout << "m " << std::setw(5) << merges[best_merge].a << " " << std::setw(5) << merges[best_merge].b << " " << std::setw(width_delta) << std::left << merges[best_merge].dEnergy << " ";
            }
          }
          else {
            if (d_exclusion <= merges[best_merge].dEnergy) {
              if (d_exclusion > first * 0.5)
                break;
              exclusion(false, changed_regions);
              for (auto t : changed_regions)
                used[t] = true;
              pred = d_exclusion;
              if (debug)
                std::cout << "e             " << std::setw(width_delta) << std::left << d_exclusion << " ";
            }
            else {
              if (merges[best_merge].dEnergy > first * 0.5)
                break;
              merge(merges[best_merge].a, merges[best_merge].b, false);
              used[merges[best_merge].a] = true;
              used[merges[best_merge].b] = true;
              pred = merges[best_merge].dEnergy;
              changed_regions.insert(merges[best_merge].a);
              changed_regions.insert(merges[best_merge].b);
              if (debug)
                std::cout << "m " << std::setw(5) << merges[best_merge].a << " " << std::setw(5) << merges[best_merge].b << " " << std::setw(width_delta) << std::left << merges[best_merge].dEnergy << " ";
            }
          }
        }
        else {
          if (d_inclusion <= d_exclusion) {
            if (d_inclusion <= splits[best_split].dEnergy) {
              if (d_inclusion > first * 0.5)
                break;
              inclusion(false, changed_regions);
              for (auto t : changed_regions)
                used[t] = true;
              pred = d_inclusion;
              if (debug)
                std::cout << "i             " << std::setw(width_delta) << std::left << d_inclusion << " ";
            }
            else {
              if (splits[best_split].dEnergy > first * 0.5)
                break;
              splits[best_split].dEnergy = 0;
              split(splits[best_split].a, false);
              used[splits[best_split].a] = true;
              pred = splits[best_split].dEnergy;
              if (debug)
                std::cout << "s " << std::setw(5) << std::right << splits[best_split].a << "     " << std::setw(width_delta) << std::left << splits[best_split].dEnergy << " ";
              changed_regions.insert(splits[best_split].a);
              changed_regions.insert(regions.size() - 1);
            }
          }
          else {
            if (d_exclusion <= splits[best_split].dEnergy) {
              if (d_exclusion > first * 0.5)
                break;
              exclusion(false, changed_regions);
              for (auto t : changed_regions)
                used[t] = true;
              pred = d_exclusion;
              if (debug)
                std::cout << "e             " << std::setw(width_delta) << std::left << d_exclusion << " ";
            }
            else {
              if (splits[best_split].dEnergy > first * 0.5)
                break;
              splits[best_split].dEnergy = 0;
              split(splits[best_split].a, false);
              used[splits[best_split].a] = true;
              pred = splits[best_split].dEnergy;
              if (debug)
                std::cout << "s " << std::setw(5) << std::right << splits[best_split].a << "     " << std::setw(width_delta) << std::left << splits[best_split].dEnergy << " ";
              changed_regions.insert(splits[best_split].a);
              changed_regions.insert(regions.size() - 1);
            }
          }
        }

        if (debug)
          std::cout << std::endl;

        /*
              if ((std::size_t(iteration/100) * 100) == iteration)
                save(std::to_string(iteration) + "-iteration.ply");*/

        for (std::size_t i = 0; i < regions.size(); i++)
          eps_check(i);
        verify();

        FT after = energy();
        if (before < after) {
          std::cout << "ENERGY INCREASED! " << before << " to " << after << std::endl;
          std::cout << "d " << (after - before) << std::endl;
          std::cout << " energy: " << energy() << " fidelity: " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;
          exit(5);
        }
/*
        if (!changed_regions.empty())
          break;*/
      }

      /*
            if (CGAL::abs(before - after + pred) > 0.00001) {
              std::cout << "prediction is off: " << (before - after + pred) << " pred: " << pred << " before: " << before << " after: " << after << std::endl;
            }*/
      if (debug) {
        std::cout << std::setw(width_delta) << std::left << energy() << " \t" << std::flush;
        for (std::size_t r : changed_regions)
          std::cout << std::setw(5) << std::right << r << " ";
        std::cout << std::endl;
      }

      if (changed_regions.empty())
        break;

      update_operations(changed_regions);
    }

    //std::cout << " energy: " << energy() << " fidelity : " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;

/*
      std::cout << "t0: " << (ti[0] / 1000) << std::endl;
      for (std::size_t i = 1; i < 8; i++)
        std::cout << "t" << i << ": " << ((ti[i] - ti[i - 1]) * 1000) << std::endl;


      if (operation_count[0])
        std::cout << "simulating " << operation_count[0] << " exclusions took: " << time_per_operation[0] * 1000 << "ms" << std::endl;

      if (operation_count[1])
        std::cout << "simulating " << operation_count[1] << " inclusion took: " << time_per_operation[1] * 1000 << "ms" << std::endl;

      if (operation_count[2])
        std::cout << "simulating " << operation_count[2] << " splits took: " << time_per_operation[2] * 1000 << "ms" << std::endl;

      if (operation_count[3])
        std::cout << "simulating " << operation_count[3] << " merges took: " << time_per_operation[3] * 1000 << "ms" << std::endl;*/
  }

  void update_operations(const std::set<std::size_t>& changed) {
    CGAL::Timer t;
    for (int i = merges.size() - 1; i != -1; i--)
      if (changed.find(merges[i].a) != changed.end() || changed.find(merges[i].b) != changed.end())
        merges.erase(merges.begin() + i);

    for (int i = splits.size() - 1; i != -1; i--)
      if (changed.find(splits[i].a) != changed.end())
        splits.erase(splits.begin() + i);

    for (std::size_t i : changed) {
      if (regions[i].empty())
        continue;

      t.reset();
      t.start();
      FT d = split(i, true);
      t.stop();

      time_per_operation[2] += t.time();
      operation_count[2]++;

      if (d < 0) {
        splits.push_back(Operation(SPLIT, i, d));
      }

      std::set<std::size_t> neighbors;
      adjacent(i, neighbors);
      for (auto j : neighbors) {
        t.reset();
        t.start();
        d = merge(i, j, true);
        t.stop();

        time_per_operation[3] += t.time();
        operation_count[3]++;

        if (d < 0) {
          merges.push_back(Operation(MERGE, i, j, d));
        }
      }
    }
  }

  void perform_alternating_optimization() {
    merges.clear();
    splits.clear();

    iteration = 0;
    m_pruned = 0;

    CGAL::Timer t;

    // First fill of the operation vectors.
    if (!m_no_split) {
      std::cout << "simulating splits" << std::endl;
      for (std::size_t i = 0; i < regions.size(); i++) {
        if (regions[i].size() == 0)
          continue;

        t.reset();
        t.start();
        FT d = split(i, true);
        t.stop();

        time_per_operation[2] += t.time();
        operation_count[2]++;

        if (d < 0) {
          splits.push_back(Operation(SPLIT, i, d));
        }
      }
      std::cout << "simulating " << operation_count[2] << " splits took: " << time_per_operation[2] * 1000 << "ms" << std::endl;
    }


    std::cout << "simulating merges of " << regions.size() << " regions" << std::endl;

    double maxT = 0;
    std::size_t a, b;

    for (std::size_t i = 0; i < regions.size(); i++) {
      if (regions[i].size() == 0)
        continue;


      std::set<std::size_t> neighbors;
      adjacent(i, neighbors);

      for (auto j : neighbors) {
        if (j < i)
          continue;

        CGAL::Timer t2;
        t2.reset();
        t2.start();
        FT d = merge(i, j, true);
        t2.stop();

        if (t2.time() > maxT) {
          maxT = t2.time();
          a = i;
          b = j;
        }

        time_per_operation[3] += t2.time();
        operation_count[3]++;
        if (d < 0) {
          merges.push_back(Operation(MERGE, i, j, d));
        }
      }
    }

    std::cout << "simulating " << operation_count[3] << " merges took: " << time_per_operation[3] * 1000 << "ms" << std::endl;
    std::cout << m_pruned << " merges pruned" << std::endl;

    std::cout << "longest merge was " << maxT * 1000 << " ms of regions " << a << "(" << regions[a].size() << ") and " << b << "(" << regions[b].size() << ")" << std::endl;

    // Alternating loop. Start with operations.
    FT former = (std::numeric_limits<double>::max)();
    std::size_t loops = 0;
    while (energy() < former) {
      former = energy();
      std::cout << "operations loop" << std::endl;
      perform_operations_loop();
      save(std::to_string(loops) + "-before-transfer-loop.ply");
      std::cout << "global transfer" << std::endl;
      std::cout << "energy: " << energy() << " fidelity: " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;
      full_transfer();
      save(std::to_string(loops) + "-loop.ply");
      std::cout << loops << ". energy: " << energy() << " fidelity: " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;
      loops++;
      std::cout << "end of loop" << std::endl;

    }
  }

  void remove_epsilon_violations() {
    num_inliers = 0;
    std::vector<bool> changed(regions.size(), false);
    for (auto f : mesh.faces()) {
      std::size_t region_index = get(region_map, f);
      if (region_index == std::size_t(-1))
        continue;

      FT d = CGAL::abs(dist(f, primitives[region_index]));
      if (d > epsilon) {
        put(region_map, f, std::size_t(-1));
        changed[region_index] = true;
      }
      else num_inliers++;
    }

    for (std::size_t i = 0; i < regions.size(); i++) {
      if (regions[i].size() < min_region_size) {
        remove_region(i);
      }
      else if (changed[i]) {
        sum_distances[i] = fit(regions[i], primitives[i]);
      }
    }
  }

  void remove_region(std::size_t idx) {
    for (auto f : regions[idx]) {
      std::size_t region_index = get(region_map, f);
      if (region_index == idx)
        put(region_map, f, std::size_t(-1));
    }

    sum_distances[idx] = 0;
    num_inliers -= regions[idx].size();

    regions[idx].clear();
    // Mark primitive as invalid
    primitives[idx] = Primitive(0, 0, 0, 2 * epsilon);

    //verify();
  }

  void verify() {
    std::size_t outliers = 0;
    for (auto f : mesh.faces()) {
      std::size_t region_index = get(region_map, f);
      if (region_index == -1) {
        outliers++;
        continue;
      }

      CGAL_assertion(region_index < regions.size());
      if (regions[region_index].find(f) == regions[region_index].end()) {
        std::cout << "ERROR: region_map does not map region sets ri: " << region_index << " f: " << f << std::endl;
        for (std::size_t i = 0; i < regions.size(); i++) {
          std::cout << i << ". region " << regions[i].size() << std::endl;
        }
        exit(5);
      }
    }

    std::size_t sum = 0;

    std::vector<bool> checked(mesh.num_faces(), false);
    for (std::size_t i = 0; i < regions.size(); i++) {
      if (CGAL::abs(sum_distances[i] - dist_sum(regions[i], primitives[i])) > 0.00001)
        std::cout << "ERROR: sum_distances[" << i << "] deviates from dist_sum" << std::endl;

      sum += regions[i].size();
      for (auto f : regions[i]) {
        if (checked[f]) {
          std::cout << "ERROR: face contained in two region sets" << std::endl;
        }
        checked[f] = true;
        std::size_t region_index = get(region_map, f);
        if (region_index != i)
          std::cout << "ERROR: region set contains face with region_map mismatch" << std::endl;
      }
    }

    if (sum != num_inliers)
      std::cout << "num_inliers is wrong" << std::endl;

    if ((sum + outliers) != mesh.num_faces()) {
      std::cout << "ERROR: sum of region set sizes and outliers does not match num_faces" << std::endl;
    }
  }

  FT exclusion(bool simulate, std::set<std::size_t>& affected_regions) {
    if (m_no_exclusion)
      return 0;
    //verify();
    // Global operator, removes the 10 worst fits
    std::vector<std::pair<FT, std::pair<Face, std::size_t> > > distances;

    FT sFidelity = 0;
    std::size_t inliers = 0;

    for (auto f : mesh.faces()) {
      std::size_t region_index = get(region_map, f);
      if (region_index == -1)
        continue;

      inliers++;

      FT d = CGAL::abs(dist(f, primitives[region_index]));
      sFidelity += d / epsilon;

      distances.push_back(std::make_pair(d, std::make_pair(f, region_index)));
    }

    CGAL_assertion(num_inliers == inliers);

    std::sort(distances.begin(), distances.end(), [](auto& a, auto& b) {return a.first > b.first; });

    FT dFidelity = 0;
    int dInliers = 0;

    // first collect all removed samples per primitive
    std::map<std::size_t, std::vector<Face> > to_be_removed;
    for (std::size_t i = 0; i < (std::min<int>)(10, distances.size()); i++) {
      to_be_removed[distances[i].second.second].push_back(distances[i].second.first);
      affected_regions.insert(distances[i].second.second);
    }

    FT dSimplicity = 0;

    for (auto p : to_be_removed) {
      if (!simulate) {
        // Check if regions drop below min_region_size, if so discard the full region.
        if (regions[p.first].size() - p.second.size() < min_region_size) {
          remove_region(p.first);
        }
        else
        {
          for (auto f : p.second) {
            put(region_map, f, std::size_t(-1));
            regions[p.first].erase(f);
            num_inliers--;
          }
          std::size_t new_outliers = 0;
          sum_distances[p.first] = fit(regions[p.first], primitives[p.first], centers[p.first], new_outliers, true);
          num_inliers -= new_outliers;
        }
      }
      else {
        // Check if region would be dropped by this operation.
        // If so, just the delta of inliers needs to be considered.
        if (regions[p.first].size() - p.second.size() < min_region_size) {
          dInliers += regions[p.first].size();
          dFidelity -= sum_distances[p.first];
          dSimplicity -= 1.0 / FT(mesh.num_faces() / min_region_size);
          continue;
        }
        // Create a temporary set
        std::set<Face> region = regions[p.first];
        for (auto f : p.second)
          region.erase(f);

        // Fit temporary primitive and recalculate fidelity
        Primitive prim;
        Point_3 c;
        std::size_t new_outliers = 0;
        dFidelity += fit(region, prim, c, new_outliers);
        dInliers -= p.second.size();
      }
    }

    //verify();

    if (num_inliers + dInliers == 0)
      return w1 + w2 + w3;

    if (!simulate) {
      total_fidelity = 0;
      for (std::size_t i = 0; i < sum_distances.size(); i++)
        total_fidelity += sum_distances[i];

      return w1 * total_fidelity / num_inliers + w2 * simplicity() + w3 * completeness();
    }
    else
    {
      for (std::size_t i = 0; i < sum_distances.size(); i++) {
        if (to_be_removed.count(i) == 0)
          dFidelity += sum_distances[i];
      }

      dFidelity = dFidelity / (num_inliers + dInliers) - total_fidelity;

      FT dCompleteness = -(dInliers / FT(mesh.num_faces()));

      return w1 * dFidelity + w2 * dSimplicity + w3 * dCompleteness;
    }
  }

  std::vector<Face> adjacent(Face face) {
    std::vector<Face> neighbors;

    Halfedge h = mesh.halfedge(face);

    do {
      Halfedge o = mesh.opposite(h);
      if (!mesh.is_border(o)) {
        neighbors.push_back(mesh.face(o));
      }
      h = mesh.next(h);
    } while (h != mesh.halfedge(face));

    return neighbors;
  }

  std::size_t adjacent(std::size_t region, std::set<std::size_t>& neighbors) {
    neighbors.clear();

    for (auto f : regions[region]) {
      Halfedge h = mesh.halfedge(f);

      do {
        Halfedge o = mesh.opposite(h);
        if (!mesh.is_border(o)) {
          std::size_t neighbor_region = get(region_map, mesh.face(o));
          if (neighbor_region != region && neighbor_region != -1) {
            neighbors.insert(neighbor_region);
          }
        }
        h = mesh.next(h);
      } while (h != mesh.halfedge(f));
    }

    return neighbors.size();
  }

  bool single_component(const std::set<Face>& region) {
    if (region.size() == 0)
      return true;

    std::set<Face> touched;

    std::queue<Face> border;
    border.push(*region.begin());
    touched.insert(border.front());
    while (!border.empty()) {
      std::vector<Face> adj = adjacent(border.front());
      border.pop();
      for (Face f : adj) {
        if (region.find(f) != region.end()) {
          auto p = touched.insert(f);
          if (p.second)
            border.push(f);
        }
      }
    }

    return touched.size() == region.size();
  }

  void border(std::size_t a, std::size_t b, std::set<Face>& border_a, std::set<Face>& border_b) {
    border_a.clear();
    border_b.clear();

    for (auto f : regions[a]) {
      Halfedge h = mesh.halfedge(f);

      do {
        Halfedge o = mesh.opposite(h);
        if (!mesh.is_border(o)) {
          std::size_t neighbor_region = get(region_map, mesh.face(o));
          if (neighbor_region == b) {
            border_a.insert(mesh.face(f));
            border_b.insert(mesh.face(o));
          }
        }
        h = mesh.next(h);
      } while (h != mesh.halfedge(f));
    }
  }

  template<typename Container, typename RegionMap>
  std::size_t border(const Container& region, std::size_t b, std::set<Face>& border_in_b, RegionMap rmap) {
    border_in_b.clear();

    for (auto f : region) {
      Halfedge h = mesh.halfedge(f);

      do {
        Halfedge o = mesh.opposite(h);
        if (!mesh.is_border(o)) {
          std::size_t neighbor_region = get(rmap, mesh.face(o));
          if (neighbor_region == b) {
            border_in_b.insert(mesh.face(o));
          }
        }
        h = mesh.next(h);
      } while (h != mesh.halfedge(f));
    }

    return border_in_b.size();
  }

  FT inclusion(bool simulate, std::set<std::size_t>& affected_regions) {
    if (num_inliers == mesh.num_faces())
      return 0;
    // Global operator, add 10 outliers to best fit adjacent region
    std::vector<std::pair<FT, std::pair<Face, std::size_t> > > distances;

    for (auto f : mesh.faces()) {
      int region_index = get(region_map, f);
      if (region_index != -1)
        continue;

      std::vector<Face> adj = adjacent(f);

      FT best_dist = epsilon;
      std::size_t best_region = -1;

      for (auto n : adj) {

        int neighbor_region = get(region_map, n);
        if (neighbor_region == -1)
          continue;

        FT d = CGAL::abs(dist(f, primitives[neighbor_region]));
        if (d < best_dist) {
          best_dist = d;
          best_region = neighbor_region;
        }
      }

      if (best_region == -1)
        continue;

      distances.push_back(std::make_pair(best_dist, std::make_pair(f, best_region)));
    }

    std::sort(distances.begin(), distances.end(), [](auto& a, auto& b) {return a.first < b.first; });

    FT dFidelity = 0;
    std::size_t dInliers = 0;

    // first collect all to be added samples per primitive
    std::map<std::size_t, std::vector<Face> > to_be_added;
    for (std::size_t i = 0; i < (std::min<int>)(50, distances.size()); i++) {
      to_be_added[distances[i].second.second].push_back(distances[i].second.first);
      affected_regions.insert(distances[i].second.second);
    }

    for (auto p : to_be_added) {
      if (!simulate) {
        for (auto f : p.second) {
          put(region_map, f, p.first);
          regions[p.first].insert(f);
          num_inliers++;
        }
        std::size_t new_outliers = 0;
        sum_distances[p.first] = fit(regions[p.first], primitives[p.first], centers[p.first], new_outliers, true, region_map);
        num_inliers -= new_outliers;
      }
      else {
        // Create a temporary set
        std::vector<Face> region;
        region.reserve(regions[p.first].size() + p.second.size());
        std::copy(regions[p.first].begin(), regions[p.first].end(), std::back_inserter(region));
        for (auto f : p.second)
          region.push_back(f);

        // Fit temporary primitive and recalculate fidelity
        Primitive prim;
        std::size_t new_outliers = 0;
        dFidelity += fit2(region, prim, new_outliers);
        dInliers += p.second.size() - new_outliers;
      }
    }

    //verify();

    if (!simulate) {
      total_fidelity = 0;
      for (std::size_t i = 0; i < sum_distances.size(); i++)
        total_fidelity += sum_distances[i];

      return w1 * total_fidelity / num_inliers + w2 * simplicity() + w3 * completeness();
    }
    else
    {
      for (std::size_t i = 0; i < sum_distances.size(); i++) {
        if (to_be_added.count(i) == 0)
          dFidelity += sum_distances[i];
      }

      dFidelity = dFidelity / (num_inliers + dInliers) - total_fidelity;

      FT dCompleteness = -(dInliers / FT(mesh.num_faces()));

      return w1 * dFidelity + w3 * dCompleteness;
    }
  }

  FT merge(std::size_t a, std::size_t b, bool simulate) {
    CGAL::Timer t;
    // This function does not check for adjacency, although the function is supposed to be called on adjacent regions!
    CGAL_assertion(a < regions.size());
    CGAL_assertion(b < regions.size());
    t.reset();

    t.start();

    bool pruned = false;

    FT angle = primitives[a].orthogonal_vector() * primitives[b].orthogonal_vector();
    FT dist = CGAL::sqrt(CGAL::squared_distance(centers[a], centers[b]));
    FT eps_dist = 2 * (dist + 1.0) * epsilon;
    FT dist_a = CGAL::abs((centers[a] - centers[b]) * primitives[a].orthogonal_vector());
    FT dist_b = CGAL::abs((centers[a] - centers[b]) * primitives[b].orthogonal_vector());
    if (prune_merges)
      if (regions[a].size() > 50 && regions[b].size() > 50) {
        if (dist_a > eps_dist && dist_b > eps_dist) {
/*
          std::cout << "pruning " << a << "(" << regions[a].size() << ") " << b << "(" << regions[b].size() << "): " << std::endl;
          std::cout << "dist: " << dist << std::endl;
          std::cout << "dist_eps: " << ((dist + 1.0) * epsilon) << std::endl;
          std::cout << "dist_a: " << dist_a << std::endl;
          std::cout << "dist_b: " << dist_b << std::endl;*/
          pruned = true;
          m_pruned++;
          //return 0;
        }
      }
    // The longer the distance between the centers, the bigger the primitives
    // If the angle is large the distance between centers needs to be small
    // For large primitives the angle is more reliable.

    // How to check?
    // for small regions it is difficult
    // angle between regions and the vector between centers?
    // distance of centers to other planes?
    // Having the extend of the region would be useful

    // I actually need to consider the change in simplicity and especially the weight

    FT former_f = fidelity();
    FT former_c = completeness();
    FT former_s = simplicity();

    if (!simulate) {
      std::size_t new_outliers = 0;
      regions[a].insert(regions[b].begin(), regions[b].end());

      // Reassign all inliers of b to a
      for (auto f : regions[b]) {
        put(region_map, f, a);
      }
      sum_distances[a] = fit(regions[a], primitives[a], centers[a], new_outliers, true, region_map);
      num_inliers -= new_outliers;
      sum_distances[b] = 0;
      primitives[b] = Primitive(0, 0, 0, 2 * epsilon);
      regions[b].clear();

      return energy();
    }
    else {
      Primitive prim;
      std::size_t new_outliers = 0;
      FT dFidelity = (fit2(regions[a], prim, new_outliers, regions[b]) - sum_distances[a] - sum_distances[b]);
      dFidelity /= (num_inliers - new_outliers);
      double t4 = t.time();
/*
      std::cout << "fidelity time: " << t1 << std::endl;
      std::cout << "completeness time: " << (t2 - t1) << std::endl;
      std::cout << "simplicity time: " << (t3 - t2) << std::endl;
      std::cout << "fit time: " << (t4 - t3) << std::endl;*/

/*
      std::cout << "t0: " << time[0] << std::endl;
      for (int i = 1; i < 8; i++)
        std::cout << "t" << i << ": " << time[i] - time[i - 1] << std::endl;*/

      FT res = w1 * dFidelity + w2 * (-1.0 / FT(mesh.num_faces() / min_region_size)) + w3 * (new_outliers / FT(mesh.num_faces()));
      //if (pruned && res < 0)
        //std::cout << "pruned: " << res << " " << a << "(" << regions[a].size() << ") " << b << "(" << regions[b].size() << "): " << std::endl;
      //if (!pruned && res > 0.0)
        //std::cout << "not pruned: " << res << " " << a << "(" << regions[a].size() << ") " << b << "(" << regions[b].size() << "): " << std::endl;
      return res;
    }
  }

  FT split(std::size_t a, bool simulate) {
    if (m_no_split)
      return 0;

    if (regions[a].size() < 2 * min_region_size)
      return 0;

    if (!single_component(regions[a]))
      std::cout << iteration << ": " << a << " is not a single component!" << std::endl;

    std::size_t start_size = regions[a].size();

/*
    if (debug)
      std::cout << "split start size " << start_size << std::endl;*/

    FT neg = 0, pos = 0;
    Face fneg, fpos;

    for (auto f : regions[a]) {
      FT d = dist(f, primitives[a]);
      if (d < neg) {
        neg = d;
        fneg = f;
        continue;
      }
      if (d > pos) {
        pos = d;
        fpos = f;
      }
    }

    if (fpos == fneg || fpos == Face(-1) || fneg == Face(-1))
      return 0;

    // Split the region spatially into two connected parts
    std::set<Face> rpos, rneg;
    std::queue<Face> border;
    rneg.insert(fneg);
    rpos.insert(fpos);
    std::set<Face> assigned;

    Point_3 cpos = center(fpos);
    Point_3 cneg = center(fneg);

    border.push(fpos);
    while (!border.empty()) {
      Face f = border.front();
      border.pop();
      if (assigned.find(f) != assigned.end())
        continue;

      Point_3 c = center(f);
      FT dpos = CGAL::squared_distance(cpos, c);
      FT dneg = CGAL::squared_distance(cneg, c);

      if (dpos <= dneg) {
        rpos.insert(f);
        assigned.insert(f);
        std::vector<Face> adj = adjacent(f);
        for (auto& n : adj) {
          if (get(region_map, n) == a && assigned.find(n) == assigned.end())
            border.push(n);
        }
      }
    }

    if (assigned.find(fneg) != assigned.end())
      std::cout << "Split "<< a << " region with " << regions[a].size() << " inliers has falsely assigned fneg" << std::endl;

    border.push(fneg);
    while (!border.empty()) {
      Face f = border.front();
      border.pop();
      if (assigned.find(f) != assigned.end())
        continue;

      Point_3 c = center(f);
      FT dpos = CGAL::squared_distance(cpos, c);
      FT dneg = CGAL::squared_distance(cneg, c);

      if (dneg <= dpos) {
        rneg.insert(f);
        assigned.insert(f);
        std::vector<Face> adj = adjacent(f);
        for (auto& n : adj) {
          if (get(region_map, n) == a && assigned.find(n) == assigned.end())
            border.push(n);
        }
      }
    }

/*
    if (debug)
      std::cout << rpos.size() << " rpos, " << rneg.size() << " rneg after first grow" << std::endl;*/

    std::vector<Face> new_outliers;

    std::size_t new_rpos = 0;
    std::size_t new_rneg = 0;

    // Assign left overs
    std::vector<Face> leftovers;

    for (auto f : regions[a]) {
      if (assigned.find(f) == assigned.end())
        leftovers.push_back(f);
    }

    bool changed = true;
    while (changed) {
      changed = false;

      for (auto& f : leftovers) {
        if (f == Face(-1))
          continue;

        std::vector<Face> adj = adjacent(f);
        bool inserted = false;
        for (auto& n : adj) {
          if (rpos.find(n) != rpos.end()) {
            rpos.insert(f);
            changed = true;
            f = Face(-1);
            new_rpos++;
            break;
          }
          else if (rneg.find(n) != rneg.end()) {
            rneg.insert(f);
            changed = true;
            f = Face(-1);
            new_rneg++;
            break;
          }
        }
      }
    }

/*
    if (debug)
      std::cout << rpos.size() << " rpos, " << rneg.size() << " rneg after second grow with " << leftovers.size() << " outliers" << std::endl;*/

    if (!single_component(rpos))
      std::cout << "rpos is not a connected region!" << std::endl;

    if (!single_component(rneg))
      std::cout << "rneg is not a connected region!" << std::endl;

    // If the face does not have a neighbor belonging to one of the split regions, it is labeled as outlier.
    for (auto f : leftovers) {
      if (f == Face(-1))
        continue;
      put(rms, f, std::size_t(-1));
      new_outliers.push_back(f);
    }

    for (auto f : mesh.faces())
      put(rms, f, std::size_t(-1));

    for (auto f : rpos)
      put(rms, f, a);

    for (auto f : rneg)
      put(rms, f, regions.size());

    Primitive primA, primB;
    Point_3 ca, cb;
    std::size_t new_outlier_count;
    fit(rpos, primA, ca, new_outlier_count);
    fit(rneg, primB, cb, new_outlier_count);

    FT dist_sum_a;
    FT dist_sum_b;

    //transfer loop
    int rounds = 15;
    while (rounds > 0) {
      bool good_before = (rpos.size() >= min_region_size && rneg.size() >= min_region_size);
      dist_sum_a = dist_sum(rpos, primA);
      dist_sum_b = dist_sum(rneg, primB);
      std::size_t tmp = 0;
      FT dA = transfer(a, regions.size(), true, rpos, rneg, primA, primB, ca, cb, tmp, rms);
      FT dB = transfer(regions.size(), a, true, rneg, rpos, primB, primA, cb, ca, tmp, rms);

      if (dA >= 0 && dB >= 0)
        break;

      std::set<Face> rpos_tmp = rpos;
      std::set<Face> rneg_tmp = rneg;

      if (dA < dB)
        transfer(a, regions.size(), false, rpos_tmp, rneg_tmp, primA, primB, ca, cb, new_outlier_count, rms);
      else
        transfer(regions.size(), a, false, rneg_tmp, rpos_tmp, primB, primA, cb, ca, new_outlier_count, rms);

/*
      if (rpos_tmp.size() + rneg_tmp.size() + new_outliers.size() != regions[a].size()) {
        std::cout << rounds << " " << rpos_tmp.size() << " + " << rneg_tmp.size() << " + " << new_outliers.size() << " is not " << regions[a].size() << std::endl;
      }*/

      if (rpos_tmp.size() < min_region_size || rneg_tmp.size() < min_region_size && good_before)
        break;

      if (!single_component(rpos_tmp))
        break;

      if (!single_component(rneg_tmp))
        break;


      std::swap(rpos, rpos_tmp);
      std::swap(rneg, rneg_tmp);

      dist_sum_a = dist_sum(rpos, primA);
      dist_sum_b = dist_sum(rneg, primB);

      rounds--;
    }

    if (!simulate) {
      CGAL_assertion(rpos.size() >= min_region_size);
      CGAL_assertion(rneg.size() >= min_region_size);

      // Copy indices from temporary region map into actual region map
      for (auto f : regions[a])
        put(region_map, f, get(rms, f));

      regions[a].clear();
      //std::size_t new_outliers_count = 0;
      std::copy(rpos.begin(), rpos.end(), std::inserter(regions[a], regions[a].begin()));
      sum_distances[a] = fit(regions[a], primitives[a], centers[a], new_outlier_count, true);

      // Add a new region and primitive
      regions.push_back(std::set<Face>());
      std::copy(rneg.begin(), rneg.end(), std::inserter(regions.back(), regions.back().begin()));
      primitives.push_back(Primitive());
      centers.push_back(Point_3());
      sum_distances.push_back(fit(regions.back(), primitives.back(), centers.back(), new_outlier_count, true));

      if (regions[a].size() + regions.back().size() + new_outliers.size() != start_size)
        std::cout << "split error, resulting region sizes " << regions[a].size() << " " << regions.back().size() << " with " << new_outliers.size() << " do not match original size " << start_size << std::endl;

/*
      // Flip region indices from the original region to the new part.
      for (auto f : regions.back())
        put(region_map, f, regions.size() - 1);

      for (auto f : new_outliers) {
        put(region_map, f, std::size_t(-1));
      }*/

      num_inliers -= new_outliers.size();

      return energy();
    }
    else {
      // Don't do the split if one of the regions will be too small.
      if (rpos.size() < min_region_size || rneg.size() < min_region_size)
        return 0;



      FT df = -(sum_distances[a] - dist_sum(rpos, primA) - dist_sum(rneg, primB)) / (num_inliers - new_outliers.size());
      FT ds = (1.0 / FT(mesh.num_faces() / min_region_size));
      FT dc = new_outliers.size() / FT(mesh.num_faces());

      return w1 * df + w2 * ds + w3 * dc;
    }

    //verify();

    return 0;
  }

  std::size_t full_transfer() {
    std::priority_queue<Operation, std::vector<Operation>, Operation_order> queue;

    std::size_t transfers = 0;

    CGAL::Timer t;
    t.reset();
    t.start();

    std::vector<bool> used(regions.size(), true);
    do {
      //verify();
      // Reset vector
      while (!queue.empty()) {
        Operation op = queue.top();
        queue.pop();
        if (!used[op.a] && !used[op.b]) {
          used[op.a] = true;
          used[op.b] = true;
          std::size_t new_outliers = 0;
          transfer(op.a, op.b, new_outliers, false);
          num_inliers -= new_outliers;
          std::cout << transfers << ". energy: " << energy() << " fidelity: " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;
          //verify();
          transfers++;
          //if ((std::size_t(transfers / 100) * 100) == transfers)
          //std::cout << "energy: " << energy() << "fidelity: " << fidelity() << " simplicity : " << simplicity() << " completeness : " << completeness() << std::endl;
        }
      }
      for (std::size_t a = 0; a < regions.size(); a++) {
        std::set<std::size_t> neighbors;
        adjacent(a, neighbors);
        for (auto b : neighbors) {
          std::size_t new_outliers = 0;
          if (used[a] || used[b]) {
            FT d = transfer(a, b, new_outliers, true);
            if (d < 0)
              queue.push(Operation(TRANSFER, a, b, d));
          }
        }
      }
      for (auto& b : used)
        b = false;
    } while (!queue.empty());

    t.stop();
    double t1 = t.time();

    //std::cout << "global transfer time " << (t1 * 1000) << std::endl;

    return transfers;
  }

  template<typename RegionMap>
  FT transfer(std::size_t a, std::size_t b, bool simulate, std::set<Face>& regionA, std::set<Face>& regionB, Primitive& primA, Primitive& primB, Point_3 &centerA, Point_3 &centerB, std::size_t &new_outliers, RegionMap rmap) {
    // Just consider the immediate border
    std::set<Face> faces;
    border(regionA, b, faces, rmap);

    std::set<Face> region_a;
    std::set<Face> region_b;

    FT dist_a, dist_b;
    if (a == sum_distances.size() || b == sum_distances.size()) {
      dist_a = dist_sum(regionA, primA);
      dist_b = dist_sum(regionB, primB);
    }
    else {
      dist_a = sum_distances[a];
      dist_b = sum_distances[b];
    }

    bool changed = false;

    for (auto f : faces) {
      FT dist_a = CGAL::abs(dist(f, primA));
      FT dist_b = CGAL::abs(dist(f, primB));
      if (dist_a < dist_b) {
        changed = true;
        if (!simulate) {
          regionA.insert(f);
          regionB.erase(f);
          put(rmap, f, a);
        }
        else {
          if (region_a.size() == 0) {
            region_a = regionA;
            region_b = regionB;
          }
          region_a.insert(f);
          region_b.erase(f);
        }
      }
    }

    if (!single_component(region_a))
      return 0;

    if (!single_component(region_b))
      return 0;

    if (changed) {
      if (!simulate) {
        if (a == sum_distances.size() || b == sum_distances.size()) {
          fit(regionA, primA, centerA, new_outliers, true, rmap);
          fit(regionB, primB, centerB, new_outliers, true, rmap);
        }
        else {
          sum_distances[a] = fit(regionA, primA, centerA, new_outliers, true, rmap);
          sum_distances[b] = fit(regionB, primB, centerB, new_outliers, true, rmap);
        }

        return 0;
      }
      else {
        Primitive pa, pb;
        Point_3 ca, cb;
        std::size_t new_outliers = 0;
        FT dFidelity = fit(region_a, pa, ca, new_outliers, false) - dist_a - dist_b;
        if (region_b.size() < min_region_size) {
          dFidelity /= ((num_inliers) - region_b.size());

          return w1 * dFidelity + w2 * -1.0 / FT(mesh.num_faces() / min_region_size) + w3 * ((region_b.size() + new_outliers) / FT(mesh.num_faces()));
        }
        else {
          dFidelity += fit(region_b, pb, cb, new_outliers, false);
          dFidelity /= num_inliers - new_outliers;
          return w1 * dFidelity + w3 * new_outliers / FT(mesh.num_faces());
        }
      }
    }

    return 0;
  }

  FT transfer(std::size_t a, std::size_t b, std::size_t &new_outliers, bool simulate) {
    return transfer(a, b, simulate, regions[a], regions[b], primitives[a], primitives[b], centers[a], centers[b], new_outliers, region_map);
  }

private:
  std::priority_queue<Operation, std::vector<Operation>, Operation_order> queue;
  std::vector<Operation> merges, splits;

  std::vector<Primitive_and_region>& input;
  std::vector<Primitive> primitives;
  std::vector<std::set<Face> > regions;

  std::vector<Point_3> centers;
  std::vector<FT> sum_distances;

  bool debug;
  std::size_t iteration;

  const Mesh& mesh;
  RegionMap& region_map;
  Region_index_map rms;
  Triangle_map trimap;
  VPMap vpmap;

  //std::priority_queue
  FT w1, w2, w3;
  std::size_t min_region_size;
  FT epsilon;

  std::size_t num_inliers;
  std::size_t m_num_faces;
  FT total_fidelity;

  double time_per_operation[4];
  std::size_t operation_count[4];

  bool m_no_exclusion;
  bool m_no_split;

  std::size_t m_pruned;
};
}

template<typename Mesh, typename RegionMap, typename Primitive_and_region, typename FT, typename NamedParameters = parameters::Default_named_parameters>
void optimize_shapes(const Mesh& mesh, RegionMap& region_map, std::vector<Primitive_and_region>& regions, FT w1, FT w2, FT w3, const NamedParameters& np = parameters::default_values()) {
  //typedef typename GetGeomTraits<Mesh, NamedParameters>::type::FT FT;
  std::size_t min_region_size = parameters::choose_parameter(parameters::get_parameter(np, internal_np::minimum_region_size), 1);
  FT epsilon = parameters::choose_parameter(parameters::get_parameter(np, internal_np::maximum_distance), 1);
  internal::Shape_optimization<typename GetGeomTraits<Mesh, NamedParameters>::type, Mesh, RegionMap, Primitive_and_region> so(mesh, region_map, regions, epsilon, min_region_size, w1, w2, w3);
  so.optimize(false, false);
}
}

#endif