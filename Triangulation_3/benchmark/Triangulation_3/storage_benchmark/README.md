# TDS_3 storage benchmark

Compares the storage back-ends of `Triangulation_data_structure_3` for a
`Delaunay_triangulation_3<Epick>`, for speed, memory and correctness.

| configuration | tree | TDS |
|---|---|---|
| `MAIN`   | baseline tree (default: git worktree of `2e165ee808b`, the cgal/main commit merged into this branch) | pointer based, `Compact_container` |
| `HANDLE` | this branch | `Triangulation_data_structure_3<Vb, Cb, Sequential_tag, Handle_tag>` |
| `INDEX`  | this branch | `Triangulation_data_structure_3<VertexWithPoint<K>, Cell4Delaunay<K>, Sequential_tag, Index_tag>` |

`MAIN` vs `HANDLE` isolates the changes in the algorithms (`Triangulation_3`,
`Delaunay_triangulation_3`, TDS functions working on descriptors); `HANDLE` vs
`INDEX` isolates the storage.

## Files

- `storage_benchmark.cpp`: the benchmark. It compiles against any CGAL tree
  (`__has_include(<CGAL/TDS_3/Indexed_storage.h>)` selects MAIN vs. branch; define
  `TDS_STORAGE_INDEX=1` for `Index_tag`).
- `CMakeLists.txt`: standalone project, configure with `-DCGAL_DIR=<tree>`.
- `compare_storage_benchmark.py`: builds all configurations, runs them interleaved,
  and writes a Markdown report with timings, memory and correctness checks.
- `prevent_deref_equality.patch`: fix of the quadratic cell iteration (see below).
- `RESULTS_2026-10-12.md`: results of the first run and the analysis of the performance difference.
- `indexed_storage_experiments.patch`: the experiments of the performance
  investigation, each behind a macro (`CGAL_EXP_NO_RESET`, `CGAL_EXP_REMOVED_CHAR`,
  `CGAL_EXP_RAW_PTR`, `CGAL_EXP_NO_CHAR_ALIAS`, `CGAL_EXP_NO_POPBACK`); without the
  macros the code is unchanged.

## Running

```sh
cd Triangulation_3/benchmark/Triangulation_3/storage_benchmark
python3 compare_storage_benchmark.py                       # all defaults (takes ~30 min)
python3 compare_storage_benchmark.py --n 200000 --rounds 1 # quick run
```

The baseline worktree and the build directories go to `./storage_benchmark_build`
(`--build-dir`). Useful options: `--configs HANDLE,INDEX`, `--dataset cube|sphere|file.xyz`
(repeatable), `--cxx-flags=-march=native`, `--cmake-arg=...` (e.g. a generator or
`-DBOOST_ROOT=...`), `--no-build`, `--baseline-ref <ref>` or `--baseline-dir <tree>`.

On Windows, run the same script from a "x64 Native Tools" prompt, e.g. with
`--cmake-arg=-G --cmake-arg=Ninja`, or with a Visual Studio generator.

Benchmark the branch with patches applied, without touching the checkout (a
worktree of `HEAD` is used, so commit first what you want to measure):

```sh
python3 compare_storage_benchmark.py --patch prevent_deref_equality.patch
python3 compare_storage_benchmark.py --patch prevent_deref_equality.patch \
    --patch indexed_storage_experiments.patch \
    --variant "INDEX_EXP=-DCGAL_EXP_NO_RESET -DCGAL_EXP_REMOVED_CHAR -DCGAL_EXP_RAW_PTR"
```

`--variant NAME=FLAGS` adds an `INDEX` build with extra compile flags.

## What is measured

Per data set (`cube`: `--n` uniform random points, `sphere`: `--n/4` points on the unit
sphere, and `Data/data/points_3/ocean_r.xyz`):

| test | what |
|---|---|
| `insert_range` | `DT dt(begin, end)` (spatial sort + insertion) |
| `insert_one_by_one_N` | `insert(p, hint)` in random order, hint = last vertex (long walks) |
| `locate_100k` | 100k random queries, hint = previous result |
| `traverse_cells_points` | loop over `finite_cell_handles()`, read the 4 points |
| `visit_neighbors` | for all cells and neighbors, `n->neighbor(n->index(c)) == c` |
| `incident_cells` | `incident_cells(v)` for every vertex |
| `copy` | copy constructor |
| `remove_N` | remove 10% of the vertices (at most 20000) |

The reported value is the minimum over all repetitions of all rounds (median in
parentheses). `MEM` is the growth of the resident set size when building one triangulation.

## Correctness checks

The Delaunay triangulation of points in general position is unique, so the following
`CHECK` values must be identical in all configurations; the script reports mismatches:
validity, numbers of vertices/cells, a cell-order independent hash of all cells (sorted
vertex point hashes), the same hash for the one-by-one insertion, the cells found by
`locate`, the number of incident cells, the copy, and the triangulation after removal
(also compared to a fresh triangulation of the remaining points).

## Known issues of the branch found with this benchmark (2026-10-12)

1. **Quadratic iteration with `Index_tag`.** `Prevent_deref` (used by `all_cell_handles()`,
   `finite_cell_handles()`, ...) was rewritten with `boost::stl_interfaces`, whose default
   `operator==` for random access iterators is `(a - b) == 0`. `Index_iterator::operator-`
   walks element by element when the container has garbage, which is always the case after
   Delaunay insertions. Every loop over all cells is therefore O(n^2): it does not finish for
   100k points. The benchmark checks this first and aborts with an error.
   `prevent_deref_equality.patch` fixes it.
2. **`remove()` does not compile with `Index_tag`.** `Triangulation_3::adjacent_vertices()` uses
   `Hash_handles_with_or_without_timestamps`, which takes `&*h`, but `Index_handle::operator*`
   returns a proxy by value. The removal test is skipped for `INDEX`
   (`-DTDS_BENCH_REMOVE=1` forces it once this is fixed).
