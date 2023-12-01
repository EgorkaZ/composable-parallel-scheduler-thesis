#!/bin/bash

benchname=$1
build_dir=${2:-build}

ompflags='OMP_MAX_ACTIVE_LEVELS=8 OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite KMP_AFFINITY="granularity=core,compact" LIBOMP_NUM_HIDDEN_HELPER_THREADS=0'
prefix_path="$build_dir/benchmarks"
cpu_speed=1995

res_dir="raw_results/$build_dir/$benchname"
mkdir -p "$res_dir"
touch "$res_dir/fails"

for x in $(ls -1 ${prefix_path}/bench_${benchname}_* | xargs -n 1 basename | grep -v OMP_RUNTIME | sort); do
    sh -c "$ompflags timeout 300 $prefix_path/$x --benchmark_out_format=json --benchmark_out=$res_dir/$x.json || echo $x timed out >> $res_dir/fails";
done

lb4ompmodes=("fsc" "fac" "fac2" "tap" "mfsc" "tfss" "fiss" "awf" "af")

for x in $(ls -1 ${prefix_path}/bench_${benchname}_* | xargs -n 1 basename | grep OMP_RUNTIME); do
    for schedule in ${lb4ompmodes[@]}; do
        sh -c "$ompflags KMP_CPU_SPEED=$cpu_speed OMP_SCHEDULE=$schedule timeout 300 $prefix_path/$x --benchmark_out_format=json --benchmark_out=$res_dir/${x}_${schedule}.json || echo ${x}_${schedule} timed out >> $res_dir/fails";
    done
done
