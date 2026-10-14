#!/usr/bin/env python3
"""Build and run storage_benchmark.cpp for the TDS_3 storage back-ends and compare them.

Configurations
  MAIN    pointer based TDS of a baseline CGAL tree (default: a git worktree of
          --baseline-ref, the cgal/main commit last merged into this branch)
  HANDLE  this tree, Triangulation_data_structure_3<..., CGAL::Handle_tag>
  INDEX   this tree, Triangulation_data_structure_3<..., CGAL::Index_tag>

MAIN vs HANDLE isolates the changes of the algorithms (Triangulation_3, ...),
HANDLE vs INDEX isolates the storage.

The runs of the configurations are interleaved (--rounds), and the minimum and
the median over all repetitions are reported. All CHECK values printed by the
benchmark must be identical across configurations; differences are reported.

Example
  python3 compare_storage_benchmark.py --rounds 3 --repeat 3
  python3 compare_storage_benchmark.py --configs HANDLE,INDEX --n 200000 --dataset cube
  python3 compare_storage_benchmark.py --cmake-arg=-G --cmake-arg="Visual Studio 17 2022"
  # HEAD of this branch with a fix applied (in a separate worktree, your checkout is not touched):
  python3 compare_storage_benchmark.py --patch prevent_deref_equality.patch
  # additional INDEX build with extra compile flags:
  python3 compare_storage_benchmark.py --patch prevent_deref_equality.patch \
      --patch indexed_storage_experiments.patch \
      --variant "INDEX_EXP=-DCGAL_EXP_NO_RESET -DCGAL_EXP_REMOVED_CHAR -DCGAL_EXP_RAW_PTR"
"""

import argparse
import collections
import datetime
import os
import pathlib
import shutil
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TREE = HERE.parents[3]  # <tree>/Triangulation_3/benchmark/Triangulation_3/storage_benchmark
DEFAULT_BASELINE_REF = "2e165ee808b"  # cgal/main merged in "Merge cgal/main" (85d92fbc82d)


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def ensure_baseline(tree, ref, build_dir):
    src = build_dir / "src-baseline"
    if not (src / "CGALConfig.cmake").exists():
        run(["git", "-C", tree, "worktree", "add", "--detach", src, ref])
    else:
        head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        want = subprocess.run(["git", "-C", tree, "rev-parse", ref], capture_output=True, text=True).stdout.strip()
        if head != want:
            run(["git", "-C", src, "checkout", "--detach", ref])
    return src


def ensure_patched(tree, patches, build_dir):
    """worktree of HEAD of `tree` with `patches` applied (re-created on every call)"""
    src = build_dir / "src-patched"
    head = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if not (src / "CGALConfig.cmake").exists():
        run(["git", "-C", tree, "worktree", "add", "--detach", src, head])
    else:
        run(["git", "-C", src, "checkout", "--force", "--detach", head])
        run(["git", "-C", src, "reset", "--hard", head])
    for p in patches:
        pp = pathlib.Path(p)
        if not pp.exists() and (HERE / p).exists():
            pp = HERE / p  # patches shipped next to this script
        run(["git", "-C", src, "apply", pp.resolve()])
    return src


def find_exe(bdir):
    for cand in ["storage_benchmark", "storage_benchmark.exe", "Release/storage_benchmark.exe",
                 "Release/storage_benchmark"]:
        p = bdir / cand
        if p.exists():
            return p
    raise FileNotFoundError(f"storage_benchmark not found in {bdir}")


def build(name, cgal_dir, index, build_dir, args, extra_flags=""):
    bdir = build_dir / f"build-{name}"
    cfg = ["cmake", "-S", HERE, "-B", bdir, f"-DCGAL_DIR={cgal_dir}", "-DCMAKE_BUILD_TYPE=Release",
           f"-DTDS_STORAGE_INDEX={'ON' if index else 'OFF'}"]
    flags = " ".join(f for f in (args.cxx_flags, extra_flags) if f)
    cfg.append(f"-DCMAKE_CXX_FLAGS={flags}")
    cfg += args.cmake_arg
    run(cfg)
    run(["cmake", "--build", bdir, "--config", "Release"])
    return find_exe(bdir)


def parse(output):
    times, checks, mems, notes = [], {}, {}, []
    for line in output.splitlines():
        f = line.strip().split(",")
        if f[0] == "TIME":
            _, cfg, ds, test, tmin, tmed, n = f
            times.append((ds, test, float(tmin), float(tmed)))
        elif f[0] == "CHECK":
            checks[(f[2], f[3])] = f[4]
        elif f[0] == "MEM":
            mems[(f[2], f[3])] = int(f[4])
        elif line.startswith("# "):
            notes.append(line[2:])
    return times, checks, mems, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=pathlib.Path, default=DEFAULT_TREE, help="CGAL tree of this branch")
    ap.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF, help="git ref of the pointer based baseline")
    ap.add_argument("--baseline-dir", type=pathlib.Path, help="use this CGAL tree as baseline instead of a worktree")
    ap.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path.cwd() / "storage_benchmark_build")
    ap.add_argument("--configs", default="MAIN,HANDLE,INDEX")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved runs of every configuration")
    ap.add_argument("--repeat", type=int, default=3, help="repetitions of each test inside one run")
    ap.add_argument("--n", type=int, default=1000000, help="number of points of the synthetic data sets")
    ap.add_argument("--dataset", action="append",
                    help="cube, sphere or an .xyz file (default: cube, sphere, Data/data/points_3/ocean_r.xyz)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per benchmark run")
    ap.add_argument("--cxx-flags", default="", help="CMAKE_CXX_FLAGS for all configurations, e.g. -march=native")
    ap.add_argument("--cmake-arg", action="append", default=[], help="extra argument for cmake configure")
    ap.add_argument("--no-build", action="store_true", help="reuse existing builds")
    ap.add_argument("--patch", action="append", default=[],
                    help="apply this patch to a worktree of HEAD of --tree and benchmark HANDLE/INDEX "
                         "from there (repeatable; uncommitted changes of --tree are not included)")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME[@TREE]=CXXFLAGS",
                    help="additional INDEX configuration built with extra compile flags, optionally "
                         "from another CGAL tree, e.g. a worktree with a prototype (repeatable)")
    args = ap.parse_args()

    tree = args.tree.resolve()
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    configs = [c.strip().upper() for c in args.configs.split(",")]
    variants = {}       # name -> extra flags
    variant_trees = {}  # name -> tree
    for v in args.variant:
        name, _, flags = v.partition("=")
        name, _, vtree = name.partition("@")
        name = name.strip().upper()
        variants[name] = flags
        if vtree:
            variant_trees[name] = pathlib.Path(vtree).resolve()
        configs.append(name)
    branch_src = ensure_patched(tree, args.patch, build_dir) if args.patch else tree

    datasets = args.dataset
    if not datasets:
        datasets = ["cube", "sphere"]
        ocean = tree / "Data" / "data" / "points_3" / "ocean_r.xyz"
        if ocean.exists():
            datasets.append(str(ocean))

    exes = {}
    for c in configs:
        if c == "MAIN":
            base = args.baseline_dir.resolve() if args.baseline_dir else ensure_baseline(tree, args.baseline_ref, build_dir)
            src, index = base, False
        elif c in ("HANDLE", "INDEX") or c in variants:
            src, index = variant_trees.get(c, branch_src), c != "HANDLE"
        else:
            sys.exit(f"unknown configuration {c}")
        bdir = build_dir / f"build-{c}"
        exes[c] = find_exe(bdir) if args.no_build else build(c, src, index, build_dir, args,
                                                               variants.get(c, ""))

    bench_args = ["--repeat", str(args.repeat), "--n", str(args.n)]
    for d in datasets:
        bench_args += ["--dataset", d]

    raw = build_dir / f"raw_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    samples = collections.defaultdict(list)       # (cfg, ds, test) -> [min of each run]
    medians = collections.defaultdict(list)
    checks = collections.defaultdict(dict)       # cfg -> {(ds,key): value}
    mems = collections.defaultdict(dict)
    notes = collections.defaultdict(list)
    failed = {}
    with open(raw, "w") as rawf:
        for r in range(args.rounds):
            for c in configs:
                if c in failed:
                    continue
                print(f"=== round {r + 1}/{args.rounds}: {c}", flush=True)
                try:
                    p = subprocess.run([str(exes[c])] + bench_args, capture_output=True, text=True,
                                       timeout=args.timeout)
                    out = p.stdout
                    if p.returncode != 0:
                        failed[c] = f"exit code {p.returncode}: {p.stderr[-500:]}"
                except subprocess.TimeoutExpired as e:
                    out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                    failed[c] = f"timeout after {args.timeout}s"
                rawf.write(out)
                t, ch, me, no = parse(out)
                for ds, test, tmin, tmed in t:
                    samples[(c, ds, test)].append(tmin)
                    medians[(c, ds, test)].append(tmed)
                checks[c].update(ch)
                mems[c].update(me)
                if r == 0:
                    notes[c] = no

    # ---------------------------------------------------------------- report
    lines = []
    lines.append(f"# TDS_3 storage benchmark ({datetime.datetime.now():%Y-%m-%d %H:%M})\n")
    if args.patch:
        lines.append(f"patches applied to HEAD of the tree: {', '.join(pathlib.Path(p).name for p in args.patch)}\n")
    for v, f in variants.items():
        where = f" from `{variant_trees[v]}`" if v in variant_trees else ""
        lines.append(f"variant {v}: INDEX{where} with `{f or '(no extra flags)'}`\n")
    lines.append(f"tree `{tree}`, baseline `{args.baseline_dir or args.baseline_ref}`, "
                 f"rounds {args.rounds} x repeat {args.repeat}, CXX flags `{args.cxx_flags or '(default Release)'}`\n")
    for c in configs:
        if notes[c]:
            lines.append(f"- {c}: {notes[c][0]}")
    for c, why in failed.items():
        lines.append(f"- **{c} FAILED: {why}** (results below are partial)")
    lines.append("")

    keys = []
    for (c, ds, test) in samples:
        if (ds, test) not in keys:
            keys.append((ds, test))
    hdr = "| dataset | test | " + " | ".join(f"{c} [ms]" for c in configs)
    ratios = [(a, b) for a, b in [("HANDLE", "MAIN"), ("INDEX", "HANDLE"), ("INDEX", "MAIN")]
              if a in configs and b in configs]
    ratios += [(v, "HANDLE") for v in variants if "HANDLE" in configs]
    hdr += " | " + " | ".join(f"{a}/{b}" for a, b in ratios) + " |"
    lines.append("Minimum over all repetitions (median in parentheses). Ratio < 1 means faster.\n")
    lines.append(hdr)
    lines.append("|" + "---|" * (2 + len(configs) + len(ratios)))
    for ds, test in keys:
        row = f"| {ds} | {test} | "
        best = {}
        cells = []
        for c in configs:
            s = samples.get((c, ds, test))
            if s:
                best[c] = min(s)
                cells.append(f"{min(s):.1f} ({statistics.median(medians[(c, ds, test)]):.1f})")
            else:
                cells.append("n/a")
        row += " | ".join(cells)
        rs = []
        for a, b in ratios:
            rs.append(f"{best[a] / best[b]:.3f}" if a in best and b in best and best[b] > 0 else "n/a")
        row += " | " + " | ".join(rs) + " |"
        lines.append(row)

    lines.append("\nMemory of one triangulation (MB; `heap_in_use` on glibc, otherwise growth of the "
                 "resident set, which is only meaningful for the first data set):\n")
    mkeys = sorted({k for c in configs for k in mems[c]})
    if mkeys:
        lines.append("| dataset | " + " | ".join(configs) + " |")
        lines.append("|" + "---|" * (1 + len(configs)))
        for k in mkeys:
            lines.append(f"| {k[0]} ({k[1]}) | " + " | ".join(
                f"{mems[c][k] / 2**20:.1f}" if k in mems[c] else "n/a" for c in configs) + " |")

    lines.append("\nCorrectness (CHECK values must be identical):\n")
    allkeys = sorted({k for c in configs for k in checks[c]})
    bad = 0
    for k in allkeys:
        vals = {c: checks[c].get(k) for c in configs}
        present = {v for v in vals.values() if v is not None}
        missing = [c for c, v in vals.items() if v is None and c not in failed]
        if len(present) > 1:
            bad += 1
            lines.append(f"- MISMATCH {k[0]} {k[1]}: " + ", ".join(f"{c}={v}" for c, v in vals.items()))
        elif missing:
            lines.append(f"- missing for {','.join(missing)}: {k[0]} {k[1]}")
        if k[1] in ("valid", "remove.valid", "insert_one_by_one.same_as_range", "neighbors.consistent",
                    "copy.same", "remove.same_as_fresh"):
            for c, v in vals.items():
                if v is not None and v != "1":
                    bad += 1
                    lines.append(f"- FAILED {c} {k[0]} {k[1]} = {v}")
    lines.append(f"\n{'All checks identical.' if bad == 0 else f'{bad} problem(s) found.'} "
                 f"({len(allkeys)} check keys)")

    report = "\n".join(lines)
    print("\n" + report)
    out = build_dir / f"report_{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text(report + "\n")
    print(f"\nreport: {out}\nraw output: {raw}")
    return 1 if (bad or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
