#include <benchmark/benchmark.h>

#include "benchmarks/spmv.h"
#include "parlay/parallel.h"

static const size_t MATRIX_SIZE = (parlay::num_workers() << 4) + parlay::num_workers();

static void DoSetup(const benchmark::State &state) {
  parlay::init_plugin();
}

static void BM_MatrixTranspose(benchmark::State &state) {
  static auto matrix = SPMV::GenDenseMatrix<double>(MATRIX_SIZE, MATRIX_SIZE);
  static auto out = SPMV::DenseMatrix<double>(MATRIX_SIZE, MATRIX_SIZE);
  benchmark::DoNotOptimize(matrix);
  benchmark::DoNotOptimize(out);
  for (auto _ : state) {
    SPMV::TransposeMatrix(matrix, out);
    benchmark::ClobberMemory();
  }
}


BENCHMARK(BM_MatrixTranspose)
    ->Name("MatrixTranspose_Latency_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMicrosecond);


BENCHMARK(BM_MatrixTranspose)
    ->Name("MatrixTranspose_Throughput_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(9);


BENCHMARK_MAIN();


