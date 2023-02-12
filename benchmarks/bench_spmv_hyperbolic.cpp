#include <benchmark/benchmark.h>

#include "../include/parallel_for.h"
#include "spmv.h"

using namespace SPMV;

static constexpr size_t MATRIX_SIZE = 1 << 16;

static void BM_SpmvBenchHyperbolic(benchmark::State &state) {
  InitParallel(GetNumThreads());
  // cache matrix and vector for all iterations
  static auto A = GenSparseMatrix<double, SparseKind::HYPERBOLIC>(
      MATRIX_SIZE, MATRIX_SIZE, 1e-3);
  static auto x = GenVector<double>(MATRIX_SIZE);
  static std::vector<double> y(A.Dimensions.Rows);
  for (auto _ : state) {
    MultiplyMatrix(A, x, y);
  }
}

BENCHMARK(BM_SpmvBenchHyperbolic)
    ->Name("SpmvHyperbolic_" + GetParallelMode())
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();