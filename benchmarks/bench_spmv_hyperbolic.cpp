#include <benchmark/benchmark.h>

#include "benchmarks/spmv.h"
#include "parlay/parallel.h"
#include <unordered_map>

using namespace SPMV;

static void DoSetup(const benchmark::State &state) {
  parlay::init_plugin();
}

static constexpr auto width =
    std::array<size_t, 6>{1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15};

static auto cachedMatrix = [] {
  std::unordered_map<size_t, SparseMatrixCSR<double>> res;
  for (auto &&w : width) {
    res[w] =
        GenSparseMatrix<double, SparseKind::HYPERBOLIC>(MATRIX_SIZE, w + (parlay::num_workers() << 2) + 3, DENSITY);
    benchmark::DoNotOptimize(res[w]);
  }
  return res;
}();

static auto x = GenVector<double>(MATRIX_SIZE);
static std::vector<double> y(MATRIX_SIZE);

static void BM_SpmvBenchHyperbolic(benchmark::State &state) {
  benchmark::DoNotOptimize(x);
  benchmark::DoNotOptimize(y);

  auto &A = cachedMatrix.at(state.range(0));
  for (auto _ : state) {
    MultiplyMatrix(A, x, y);
    benchmark::ClobberMemory();
  }
}


BENCHMARK(BM_SpmvBenchHyperbolic)
    ->Name("SpmvHyperbolic_Latency_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("width")
    ->RangeMultiplier(2)
    ->Range(*width.begin(), *std::prev(width.end()))
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SpmvBenchHyperbolic)
    ->Name("SpmvHyperbolic_Throughput_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("width")
    ->RangeMultiplier(2)
    ->Range(*width.begin(), *std::prev(width.end()))
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(9);

BENCHMARK_MAIN();

