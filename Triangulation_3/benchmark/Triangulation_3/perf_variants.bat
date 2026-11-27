@echo off
rem ===========================================================================
rem  Windows / MSVC version of perf_variants.sh
rem
rem  For handle and index storage and each macro set, this script
rem    1. builds DT3_benchmark and DT3_check            (variants_win\)
rem    2. checks the triangulation (is_valid + fingerprint, must match "base")
rem    3. times DT3_benchmark pinned to one CPU (Google benchmark repetitions)
rem  and prints a summary table relative to "base".
rem
rem  There is no perf on Windows: only time is measured (no branch misses).
rem
rem  Run from an "x64 Native Tools Command Prompt for VS" (needs cl, cmake, ninja).
rem  Usage: perf_variants.bat [points_file] [iterations] [repetitions]
rem  Results: perf_results_win\
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion

rem ---- adjust to your machine (or set these variables before calling) -------
rem CGAL checkout containing the modified Triangulation_3 / TDS_3
if not defined CGAL_ROOT set "CGAL_ROOT=C:\dev\t3i"
rem Extra CMake arguments, e.g.
rem   set EXTRA_CMAKE=-Dbenchmark_DIR=C:/dev/benchmark/lib/cmake/benchmark -DBOOST_ROOT=C:/dev/boost
if not defined EXTRA_CMAKE set "EXTRA_CMAKE="
rem Affinity mask (hex) for the timing runs: 1 = logical CPU 0 (a P-core on hybrid CPUs)
if not defined CPU_MASK set "CPU_MASK=1"
rem ---------------------------------------------------------------------------

cd /d "%~dp0"

set "DATA=%~1"
if "%DATA%"=="" set "DATA=Nefertiti.xyz"
set "ITER=%~2"
if "%ITER%"=="" set "ITER=10"
set "REPS=%~3"
if "%REPS%"=="" set "REPS=3"

set "CGAL_ROOT_FWD=%CGAL_ROOT:\=/%"
set "VAR=variants_win"
set "OUT=perf_results_win"

where cl >nul 2>&1 || (echo ERROR: cl.exe not found: run from an "x64 Native Tools Command Prompt for VS" & exit /b 1)
where cmake >nul 2>&1 || (echo ERROR: cmake not found & exit /b 1)
where ninja >nul 2>&1 || (echo ERROR: ninja not found & exit /b 1)
if not exist "%DATA%" (echo ERROR: data file "%DATA%" not found & exit /b 1)
if not exist "%CGAL_ROOT%" (echo ERROR: CGAL_ROOT "%CGAL_ROOT%" not found, edit the top of this file & exit /b 1)

if not exist "%VAR%" mkdir "%VAR%"
if not exist "%OUT%" mkdir "%OUT%"
type nul > "%OUT%\check.txt"
type nul > "%OUT%\built.txt"

rem ------------------------------------------------------------ build + check
for %%S in (h i) do (
  for %%C in (base index hole unroll all) do (
    call :build_variant %%S %%C
  )
)

rem ---------------------------------------------------------------------- time
for /f "usebackq delims=" %%N in ("%OUT%\built.txt") do (
  echo === time %%N
  start "" /b /wait /affinity %CPU_MASK% "%VAR%\DT3_%%N.exe" "%DATA%" %ITER% --benchmark_repetitions=%REPS% --benchmark_out="%OUT%\%%N.json" --benchmark_out_format=json >nul
  if errorlevel 1 echo ERROR: run failed for %%N
)

rem ------------------------------------------------------- check + summary
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0summary_variants.ps1" -Out "%OUT%"
exit /b 0


rem ===========================================================================
:build_variant
set "S=%~1"
set "C=%~2"
set "NAME=%S%_%C%"
if "%S%"=="h" (set "DEFS=HANDLE_STORAGE") else (set "DEFS=INDEX_STORAGE")
if "%C%"=="index"  set "DEFS=!DEFS!;CGAL_TDS3_BRANCHLESS_INDEX"
if "%C%"=="hole"   set "DEFS=!DEFS!;CGAL_TDS3_SMALL_HOLE_DIRECTED_EDGES"
if "%C%"=="unroll" set "DEFS=!DEFS!;CGAL_T3_FIND_CONFLICTS_UNROLL"
if "%C%"=="all"    set "DEFS=!DEFS!;CGAL_TDS3_BRANCHLESS_INDEX;CGAL_TDS3_SMALL_HOLE_DIRECTED_EDGES;CGAL_T3_FIND_CONFLICTS_UNROLL"
set "BDIR=%VAR%\build_%NAME%"

echo === build %NAME%  (!DEFS!)
cmake -S win -B "%BDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCGAL_DIR=%CGAL_ROOT_FWD%" "-DVARIANT_DEFS=!DEFS!" %EXTRA_CMAKE% > "%BDIR%.cmake.log" 2>&1
if errorlevel 1 (
  echo ERROR: CMake failed for %NAME%, see %BDIR%.cmake.log
  exit /b 0
)
ninja -C "%BDIR%" DT3_benchmark DT3_check > "%BDIR%.build.log" 2>&1
if errorlevel 1 (
  echo ERROR: build failed for %NAME%, see %BDIR%.build.log
  exit /b 0
)
copy /y "%BDIR%\DT3_benchmark.exe" "%VAR%\DT3_%NAME%.exe" >nul

echo === check %NAME%
for /f "delims=" %%L in ('call "%BDIR%\DT3_check.exe" "%DATA%"') do (
  echo %NAME% %%L
  >>"%OUT%\check.txt" echo %NAME% %%L
)
>>"%OUT%\built.txt" echo %NAME%
exit /b 0
