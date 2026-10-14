// Benchmark and correctness check comparing the storage back-ends of
// Triangulation_data_structure_3:
//
//   MAIN   : pointer-based TDS (Compact_container) of a CGAL tree without
//            <CGAL/TDS_3/Indexed_storage.h>, e.g. cgal/main
//   HANDLE : this branch, TDS with CGAL::Handle_tag   (Compact_container)
//   INDEX  : this branch, TDS with CGAL::Index_tag    (Indexed_storage)
//
// The same source compiles against both trees. On a tree with
// Indexed_storage.h, define TDS_STORAGE_INDEX=1 to select INDEX, otherwise
// HANDLE is used.
//
// Every test prints machine-readable lines:
//   TIME,<config>,<dataset>,<test>,<min_ms>,<median_ms>,<runs>
//   CHECK,<config>,<dataset>,<key>,<value>
//   MEM,<config>,<dataset>,<key>,<bytes>
// CHECK values are independent of the storage and must be identical for all
// configurations (the Delaunay triangulation of points in general position is
// unique). Use compare_storage_benchmark.py to build, run and compare.
//
// Usage: storage_benchmark [--repeat R] [--n N] [--dataset name|file.xyz]...
//        datasets: cube, sphere, or a path to an .xyz file

#define CGAL_NDEBUG 1
#ifndef NDEBUG
#  define NDEBUG 1
#endif

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/Memory_sizer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#if defined(__GLIBC__) && __has_include(<malloc.h>)
#  include <malloc.h>
#  define TDS_BENCH_HEAP_STATS 1
#endif

#if __has_include(<CGAL/TDS_3/Indexed_storage.h>)
#  define TDS_STORAGE_BRANCH 1
#else
#  define TDS_STORAGE_BRANCH 0
#endif

using K = CGAL::Exact_predicates_inexact_constructions_kernel;

#if TDS_STORAGE_BRANCH && TDS_STORAGE_INDEX
using Vb  = CGAL::VertexWithPoint<K>;
using Cb  = CGAL::Cell4Delaunay<K>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Sequential_tag, CGAL::Index_tag>;
static const char* config_name = "INDEX";
#elif TDS_STORAGE_BRANCH
using Vb  = CGAL::Triangulation_vertex_base_3<K>;
using Cb  = CGAL::Delaunay_triangulation_cell_base_3<K>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Sequential_tag, CGAL::Handle_tag>;
static const char* config_name = "HANDLE";
#else
#  if TDS_STORAGE_INDEX
#    error "TDS_STORAGE_INDEX requested, but this CGAL tree has no Indexed_storage"
#  endif
using Vb  = CGAL::Triangulation_vertex_base_3<K>;
using Cb  = CGAL::Delaunay_triangulation_cell_base_3<K>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Sequential_tag>;
static const char* config_name = "MAIN";
#endif

using DT      = CGAL::Delaunay_triangulation_3<K, Tds>;
using Point_3 = K::Point_3;

// ---------------------------------------------------------------------------
// deterministic input (independent of the CGAL version and of libstdc++)

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() { // splitmix64
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); } // [0,1)
  double uniform(double a, double b) { return a + (b - a) * uniform(); }
};

static std::vector<Point_3> make_cube(std::size_t n, std::uint64_t seed) {
  Rng r(seed);
  std::vector<Point_3> pts;
  pts.reserve(n);
  for(std::size_t i = 0; i < n; ++i) {
    double x = r.uniform(-1, 1), y = r.uniform(-1, 1), z = r.uniform(-1, 1);
    pts.emplace_back(x, y, z);
  }
  return pts;
}

static std::vector<Point_3> make_sphere(std::size_t n, std::uint64_t seed) {
  Rng r(seed);
  std::vector<Point_3> pts;
  pts.reserve(n);
  while(pts.size() < n) {
    double x = r.uniform(-1, 1), y = r.uniform(-1, 1), z = r.uniform(-1, 1);
    double l = std::sqrt(x * x + y * y + z * z);
    if(l < 0.1 || l > 1) continue;
    pts.emplace_back(x / l, y / l, z / l);
  }
  return pts;
}

static std::vector<Point_3> read_xyz(const std::string& fname) {
  std::ifstream in(fname);
  if(!in) { std::cerr << "cannot open " << fname << "\n"; std::exit(1); }
  std::vector<Point_3> pts;
  std::string line;
  while(std::getline(in, line)) {
    double x, y, z;
    if(std::sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z) == 3) pts.emplace_back(x, y, z);
  }
  // remove duplicates, so that the vertex count is well defined
  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  Rng r(4711);
  for(std::size_t i = pts.size(); i > 1; --i) std::swap(pts[i - 1], pts[r.next() % i]); // shuffle
  return pts;
}

// ---------------------------------------------------------------------------
// storage independent signatures

static std::uint64_t mix(std::uint64_t h) {
  h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
  return h;
}
static std::uint64_t hash_point(const Point_3& p) {
  std::uint64_t h = 0;
  for(int i = 0; i < 3; ++i) {
    double d = p[i];
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    h = mix(h ^ (u + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)));
  }
  return h;
}
template <typename CH>
static std::uint64_t hash_cell(const DT& dt, CH c) {
  // order independent in the vertices: sort the point hashes
  std::array<std::uint64_t, 4> hs;
  for(int i = 0; i < 4; ++i)
    hs[i] = dt.is_infinite(c->vertex(i)) ? 0x1234567ULL : hash_point(c->vertex(i)->point());
  std::sort(hs.begin(), hs.end());
  std::uint64_t h = 0;
  for(auto x : hs) h = mix(h ^ (x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)));
  return h;
}
static std::uint64_t signature(const DT& dt) { // sum is independent of the cell order
  std::uint64_t s = 0;
  for(auto c : dt.all_cell_handles()) s += hash_cell(dt, c);
  return s;
}

// ---------------------------------------------------------------------------
// harness

static int repeat = 5;

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void report_time(const std::string& ds, const std::string& test, std::vector<double> t) {
  std::sort(t.begin(), t.end());
  std::cout << "TIME," << config_name << "," << ds << "," << test << "," << t.front() << ","
            << t[t.size() / 2] << "," << t.size() << std::endl;
}
template <typename T>
static void check(const std::string& ds, const std::string& key, const T& v) {
  std::cout << "CHECK," << config_name << "," << ds << "," << key << "," << v << std::endl;
}

// time `body` `repeat` times; `setup` (not timed) runs before each repetition
static void timed(const std::string& ds, const std::string& test,
                  const std::function<void()>& setup, const std::function<void()>& body) {
  std::vector<double> t;
  for(int r = 0; r < repeat; ++r) {
    setup();
    double t0 = now_ms();
    body();
    t.push_back(now_ms() - t0);
  }
  report_time(ds, test, t);
}

static void run_dataset(const std::string& ds, const std::vector<Point_3>& pts) {
  std::cout << "# " << config_name << " dataset " << ds << ": " << pts.size() << " points" << std::endl;
  check(ds, "n_points", pts.size());
  const auto noop = [] {};

  // 1. insert a range (spatial sort + incremental insertion) ------------------
  {
    std::size_t nc = 0;
    timed(ds, "insert_range", noop, [&] {
      DT dt(pts.begin(), pts.end());
      nc = dt.number_of_cells();
    });
    check(ds, "insert_range.cells", nc);
  }

  // memory footprint of one triangulation: heap bytes in use (glibc), otherwise
  // the growth of the resident set size (only meaningful for the first data set,
  // as freed memory is reused afterwards)
  {
#ifdef TDS_BENCH_HEAP_STATS
    auto in_use = [] { auto mi = mallinfo2(); return std::size_t(mi.uordblks + mi.hblkhd); };
    auto before = in_use();
    DT dt(pts.begin(), pts.end());
    auto after = in_use();
    std::cout << "MEM," << config_name << "," << ds << ",heap_in_use," << (after - before) << std::endl;
#else
    CGAL::Memory_sizer ms;
    auto before = ms.resident_size();
    DT dt(pts.begin(), pts.end());
    auto after = ms.resident_size();
    std::cout << "MEM," << config_name << "," << ds << ",resident_delta," << (after - before) << std::endl;
#endif
  }

  DT dt(pts.begin(), pts.end());
  check(ds, "valid", dt.is_valid() ? 1 : 0);
  check(ds, "vertices", dt.number_of_vertices());
  check(ds, "cells", dt.number_of_cells());
  check(ds, "finite_cells", dt.number_of_finite_cells());
  check(ds, "signature", signature(dt));

  // 2. insert one by one in input (random) order, with the previous vertex as hint
  {
    const std::size_t n1 = (std::min<std::size_t>)(pts.size(), 100000);
    std::uint64_t sig = 0;
    timed(ds, "insert_one_by_one_" + std::to_string(n1), noop, [&] {
      DT t;
      DT::Vertex_handle hint;
      for(std::size_t i = 0; i < n1; ++i) hint = t.insert(pts[i], hint);
      sig = t.number_of_cells();
    });
    DT ref(pts.begin(), pts.begin() + n1), t;
    DT::Vertex_handle hint;
    for(std::size_t i = 0; i < n1; ++i) hint = t.insert(pts[i], hint);
    check(ds, "insert_one_by_one.same_as_range", (signature(t) == signature(ref) && sig == ref.number_of_cells()) ? 1 : 0);
  }

  // 3. locate random query points, hint = previous result (long walks) --------
  {
    std::vector<Point_3> queries = make_cube(100000, 99);
    // map queries into the bounding box of the input
    CGAL::Bbox_3 bb = CGAL::bbox_3(pts.begin(), pts.end());
    for(auto& q : queries)
      q = Point_3(bb.xmin() + (q.x() + 1) / 2 * (bb.xmax() - bb.xmin()),
                  bb.ymin() + (q.y() + 1) / 2 * (bb.ymax() - bb.ymin()),
                  bb.zmin() + (q.z() + 1) / 2 * (bb.zmax() - bb.zmin()));
    std::uint64_t sig = 0;
    timed(ds, "locate_100k", noop, [&] {
      sig = 0;
      DT::Cell_handle c;
      for(const auto& q : queries) {
        c = dt.locate(q, c);
        sig += hash_cell(dt, c);
      }
    });
    check(ds, "locate.signature", sig);
  }

  // 4. traverse all finite cells and read their points --------------------------
  {
    std::uint64_t sum = 0; // integer sum: independent of the cell order
    timed(ds, "traverse_cells_points", noop, [&] {
      sum = 0;
      for(auto c : dt.finite_cell_handles())
        for(int i = 0; i < 4; ++i) sum += std::uint64_t(std::int64_t(c->vertex(i)->point().x() * 1e9));
    });
    check(ds, "traverse.sum", sum);
  }

  // 5. neighbor walk: visit all cells via adjacency (BFS-like with a counter)
  {
    std::size_t cnt = 0;
    timed(ds, "visit_neighbors", noop, [&] {
      cnt = 0;
      for(auto c : dt.all_cell_handles())
        for(int i = 0; i < 4; ++i) {
          auto n = c->neighbor(i);
          cnt += (n->neighbor(n->index(c)) == c) ? 1 : 0;
        }
    });
    check(ds, "neighbors.consistent", cnt == 4 * dt.number_of_cells() ? 1 : 0);
  }

  // 6. incident cells of every vertex -------------------------------------------
  {
    std::size_t cnt = 0;
    std::vector<DT::Cell_handle> cells;
    cells.reserve(256);
    timed(ds, "incident_cells", noop, [&] {
      cnt = 0;
      for(auto v : dt.finite_vertex_handles()) {
        cells.clear();
        dt.incident_cells(v, std::back_inserter(cells));
        cnt += cells.size();
      }
    });
    check(ds, "incident_cells.total", cnt);
  }

  // 7. copy -------------------------------------------------------------------
  {
    std::size_t nc = 0;
    timed(ds, "copy", noop, [&] {
      DT copy(dt);
      nc = copy.number_of_cells();
    });
    DT copy(dt);
    check(ds, "copy.same", (nc == dt.number_of_cells() && signature(copy) == signature(dt)) ? 1 : 0);
  }

  // 8. remove 10% of the vertices (random order) --------------------------------
  {
    const std::size_t nr = (std::min<std::size_t>)(pts.size() / 10, 20000);
    // pick the vertices by their point, so that the choice is storage independent
    std::vector<Point_3> to_remove(pts.begin(), pts.begin() + nr);
    DT work;
    std::uint64_t sig = 0;
    std::size_t nv = 0;
    timed(ds, "remove_" + std::to_string(nr),
          [&] { work = DT(pts.begin(), pts.end()); },
          [&] {
            for(const auto& p : to_remove) {
              DT::Vertex_handle v;
              if(work.is_vertex(p, v)) work.remove(v);
            }
          });
    nv = work.number_of_vertices();
    sig = signature(work);
    check(ds, "remove.vertices", nv);
    check(ds, "remove.valid", work.is_valid() ? 1 : 0);
    check(ds, "remove.signature", sig);
    DT ref(pts.begin() + nr, pts.end());
    check(ds, "remove.same_as_fresh", signature(ref) == sig ? 1 : 0);
  }
}

int main(int argc, char** argv) {
  std::size_t n = 1000000;
  std::vector<std::string> datasets;
  for(int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if(a == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
    else if(a == "--n" && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
    else if(a == "--dataset" && i + 1 < argc) datasets.push_back(argv[++i]);
    else { std::cerr << "unknown argument " << a << "\n"; return 1; }
  }
  if(datasets.empty()) datasets = {"cube", "sphere"};
  std::cout.precision(17);
  std::cout << "# config " << config_name << ", repeat " << repeat
            << ", sizeof(Vertex_handle) " << sizeof(DT::Vertex_handle)
            << ", sizeof(Cell_handle) " << sizeof(DT::Cell_handle) << std::endl;

  // Sanity check: iterating over all cells must be linear. On the branch, with
  // Index_tag, Prevent_deref<Index_iterator> compares iterators with
  // `(a - b) == 0` (boost::stl_interfaces default for random access
  // iterators), and Index_iterator::operator-() walks element by element when
  // the container has garbage, which makes every loop over all_cell_handles()
  // quadratic. See prevent_deref_equality.patch in this directory.
  {
    auto pts = make_cube(5000, 7);
    DT dt(pts.begin(), pts.end());
    double t0 = now_ms();
    std::size_t cnt = 0;
    for(auto c : dt.all_cell_handles()) { (void)c; ++cnt; }
    double t = now_ms() - t0;
    const bool linear = (cnt == dt.number_of_cells()) && t < 50.;
    check("sanity", "cell_iteration_is_linear", linear ? 1 : 0);
    if(!linear) {
      std::cerr << "ERROR (" << config_name << "): iterating over " << cnt << " cells took " << t
                << " ms, i.e. cell iteration is quadratic (see the comment in main() and "
                   "prevent_deref_equality.patch). Aborting, as the benchmark would not finish.\n";
      return 2;
    }
  }

  for(const auto& d : datasets) {
    if(d == "cube") run_dataset("cube_" + std::to_string(n), make_cube(n, 1));
    else if(d == "sphere") run_dataset("sphere_" + std::to_string(n / 4), make_sphere(n / 4, 2));
    else {
      std::string name = d.substr(d.find_last_of("/\\") + 1);
      run_dataset(name, read_xyz(d));
    }
  }
  return 0;
}
