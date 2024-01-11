#include "benchmarks/spmv.h"
#include <atomic>
#include <benchmark/benchmark.h>
#include <chrono>
#include <thread>

#include "parlay/parallel.h"

static const size_t MAX_SIZE = parlay::num_workers() * (1 << 19);
// static constexpr size_t BLOCK_SIZE = 1 << 14;
// static constexpr size_t blocks = (MAX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

static void DoSetup(const benchmark::State &state) {
  parlay::init_plugin();
}

void __attribute__((noinline,noipa)) reduceImpl(std::vector<double> &data, size_t blocks, size_t blockSize) {
  parlay::parallel_for(0, blocks, [&](size_t i) {
    double res = 0;
    double sum = 0;
    auto start = i * blockSize;
    auto end = std::min(start + blockSize, MAX_SIZE);
    for (size_t j = start; j < end; ++j) {
      sum += data[j];
    }
    benchmark::DoNotOptimize(res += sum);
  });
}

auto data = SPMV::GenVector<double>(MAX_SIZE);
static void BM_ReduceBench(benchmark::State &state) {
  auto blockSize = state.range(0) + parlay::num_workers() + 3;
  auto blocks = (MAX_SIZE + blockSize - 1) / blockSize;
  for (auto _ : state) {
    reduceImpl(data, blocks, blockSize);
    // benchmark::ClobberMemory();
  }
}


BENCHMARK(BM_ReduceBench)
    ->Name("Reduce_Latency_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("blocksize")
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 19)
    ->Unit(benchmark::kMicrosecond);


BENCHMARK(BM_ReduceBench)
    ->Name("Reduce_Throughput_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("blocksize")
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 19)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(9);


BENCHMARK_MAIN();


