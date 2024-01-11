#include "../include/benchmarks/spmv.h"
#include <benchmark/benchmark.h>

#include "parlay/parallel.h"

static const size_t MATRIX_SIZE_HERE = parlay::num_workers() * 1 << 4;

static void DoSetup(const benchmark::State &) {
  parlay::init_plugin();
}

static auto left = SPMV::GenDenseMatrix<double>(MATRIX_SIZE_HERE, MATRIX_SIZE_HERE);
static auto right = SPMV::GenDenseMatrix<double>(MATRIX_SIZE_HERE, MATRIX_SIZE_HERE);
static auto out = SPMV::DenseMatrix<double>(MATRIX_SIZE_HERE, MATRIX_SIZE_HERE);

static void BM_MatrixMul(benchmark::State &state) {
  // cache data for all iterations
  for (auto _ : state) {
    SPMV::MultiplyMatrix(left, right, out);
  }
}


BENCHMARK(BM_MatrixMul)
    ->Name("MatrixMul_Latency_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatrixMul)
    ->Name("MatrixMul_Throughput_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(9);

BENCHMARK_MAIN();


