import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from matplotlib.gridspec import GridSpec


def split_bench_name(s):
    s = s.split("_")
    prefix = []
    for part in s:
        if part.islower():
            prefix.append(part)
        else:
            break
    return "_".join(prefix), "_".join(s[len(prefix):])


# return new plot
def plot_benchmark(benchmarks, title):
    benchmarks = sorted(benchmarks.items(), key=lambda x: x[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    min_time = sorted(benchmarks, key=lambda x: x[1])[0][1]
    ax1.set_xlabel("Normalized perfomance of " + title)
    # print([(name, min_time / time) for name, time in benchmarks])
    ax1 = ax1.barh(*zip(*[(name, min_time / time) for name, time in benchmarks]))

    ax2.set_xlabel("Time of " + title + ", us")
    ax2 = ax2.barh(*zip(*benchmarks))
    return fig


def parse_benchmarks(folder_name):
    benchmarks_by_type = {}
    for bench_file in os.listdir(folder_name):
        if not bench_file.endswith('.json'):
            continue
        with open(os.path.join(folder_name, bench_file)) as f:
            # print(bench_file)
            try:
                bench = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print("Error while parsing", bench_file, e, ", skipping")
                continue
            name = bench_file.split(".")[0]
            bench_type, bench_mode = split_bench_name(name)
            if bench_type not in benchmarks_by_type:
                benchmarks_by_type[bench_type] = {}
            if "benchmarks" in bench:
                # TODO: take not only last bench
                res = bench["benchmarks"][-1]
                if "real_time" in res:
                    report = res["real_time"]
                elif "manual_time" in res:
                    report = res["manual_time"]
                elif "cpu_time" in res:
                    report = res["cpu_time"]
                benchmarks_by_type[bench_type][bench_mode] = report
            else:
                benchmarks_by_type[bench_type][bench_mode] = bench
    return benchmarks_by_type


def plot_scheduling_benchmarks(scheduling_times):
    # x for thread_idx, y for time
    min_times = [(bench_type, [[min([t["time"] for t in times]) for times in iter.values()] for iter in results]) for bench_type, results in scheduling_times.items()]
    # print(list(scheduling_times.items())[0][1][0])
    # items = [(bench_type, [[t["time"] for t in iter.values()] for iter in tasks]) for bench_type, tasks in scheduling_times.items()]
    runtimes = list(set(name.split("_")[0] for (name, _) in min_times))
    # group by prefix:
    rows = len(runtimes) + 1
    cols = 1
    width = 18
    height = width / 3 * rows
    fig, ax = plt.subplots(rows, cols, figsize=(width, height))
    for bench_type, times in reversed(sorted(min_times,
                                 key=lambda x: np.max(np.mean(np.asarray(x[1]), axis=0)))):
        # todo: better way to visualize?
        # min time per thread for each idx in range of thread count
        times = np.asarray(times)
        # plot distribution of all times for iterations as scatter around time
        runtime_ax = ax[runtimes.index(bench_type.split("_")[0])]
        means = np.min(times, axis=0)
        # sort means array
        means = np.sort(means, axis=0)
        stds = np.std(times, axis=0)
        # print(bench_type, means, stds)
        # TODO: plot stds?
        style = random.choice(["-", "--", "-.", ":"])
        color = random.choice(["b", "g", "r", "c", "m", "y", "k"])
        runtime_ax.plot(range(len(means)), means, label=bench_type, linestyle=style, color=color)
        # all in one plot
        ax[-1].plot(range(len(means)), means, label=bench_type, linestyle=style, color=color)


    for runtime in runtimes:
        idx = runtimes.index(runtime)
        ax[idx].set_title("Scheduling time, " + runtime)
    ax[-1].set_title("Scheduling time, all")
    for runtime_ax in ax:
        runtime_ax.set_xlabel("Index of thread (sorted by time of first task)")
        runtime_ax.set_ylabel("Time, us")
        runtime_ax.legend()
    return fig


def plot_scheduling_dist(scheduling_dist):
    # plot heatmap for map of thread_idx -> tasks list
    row_count = len(scheduling_dist)
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle("Scheduling distribution, results for each iteration")
    fig.tight_layout()
    gs = GridSpec(row_count, 3, figure=fig, width_ratios=[1, 1, 4])

    for iter in range(row_count):
        ax = fig.add_subplot(gs[iter, 0])
        if iter == 0:
            ax.set_title("Distribution of tasks to threads")
        ax.set_ylabel("Thread")
        ax.set_xlabel("Task")
        ax.xaxis.set_label_position('top')
        # ax.get_figure().tight_layout()

        thread_count = len(scheduling_dist[iter])
        task_count = max(max(t["index"] for t in tasks) for tasks in scheduling_dist[iter].values()) + 1
        task_height = int(task_count / thread_count)
        data = np.ones((thread_count * task_height, task_count))
        for thread_id, tasks in sorted(scheduling_dist[iter].items(), key=lambda x: x[0]):
            for t in tasks:
                idx = thread_count - 1 if thread_id == "-1" else int(thread_id)
                # data[idx, t["index"]] = 0
                # fill rectangle by zeros
                data[idx * task_height: (idx + 1) * task_height, t["index"]] = 0
        ax.imshow(data, cmap='gray', origin='lower')

    # plot heatmap thread id, cpu id
    for iter in range(row_count):
        ax = fig.add_subplot(gs[iter, 1])
        if iter == 0:
            ax.set_title("Distribution of threads to cpus")
        ax.set_ylabel("Thread")
        ax.set_xlabel("Cpu")
        ax.xaxis.set_label_position('top')
        # ax.get_figure().tight_layout()

        thread_count = len(scheduling_dist[iter].keys())
        cpu_count = len(set(t["cpu"] for tasks in scheduling_dist[iter].values() for t in tasks))
        cpus = {}  # mapping cpu_id -> idx
        threads = {}  # mapping thread_id -> idx
        data = np.ones((thread_count, cpu_count))
        for thread_id, tasks in scheduling_dist[iter].items():
            for t in tasks:
                if thread_id not in threads:
                    threads[thread_id] = len(threads)
                cpu = t["cpu"]
                if cpu not in cpus:
                    cpus[cpu] = len(cpus)
                data[threads[thread_id], cpus[cpu]] = 0
        ax.imshow(data, cmap='gray', origin='lower')

    for iter in range(row_count):
        ax = fig.add_subplot(gs[iter, 2])
        ax.set_ylabel("Clock ticks")
        ax.set_xlabel("Thread")
        # ax.get_figure().tight_layout()
        if iter == 0:
            ax.set_title("Time since start of first executed task for each thread (sorted by time)")

        thread_count = len(scheduling_dist[iter])
        times = [min(task["time"] for task in tasks) for tasks in scheduling_dist[iter].values()]
        times = np.sort(np.asarray(times))
        ax.plot(range(len(times)), times)

    return fig


if __name__ == "__main__":
    # fetch folder from args or use current folder
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "raw_results"
    benchmarks = parse_benchmarks(folder_name)
    res_path = sys.argv[2] if len(sys.argv) > 2 else "bench_results"
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # plot main benchmarks
    for bench_type, bench in benchmarks.items():
        if not bench_type.startswith("bench") or bench_type.startswith("bench_scheduling"):
            continue
        print("Processing", bench_type)
        fig = plot_benchmark(bench, bench_type)
        fig.savefig(os.path.join(res_path, bench_type + '.png'), bbox_inches='tight')
        plt.close()

    if "scheduling_dist" in benchmarks:
        # plot with and without barrier
        # group scheduling_times by suffix
        scheduling_times_by_suffix = {}
        for bench_mode, res in benchmarks["scheduling_dist"].items():
            bench_mode, measure_mode = bench_mode.rsplit("_", 1)
            if measure_mode not in scheduling_times_by_suffix:
                scheduling_times_by_suffix[measure_mode] = {}
            scheduling_times_by_suffix[measure_mode][bench_mode] = res["results"]

        for measure_mode, times in scheduling_times_by_suffix.items():
            print("Processing scheduling time", measure_mode)
            fig = plot_scheduling_benchmarks(times)
            fig.savefig(os.path.join(res_path, "scheduling_time_" + measure_mode + '.png'), bbox_inches='tight')
            plt.close()

        # plot average scheduling time for all benchs
        # avg_times = {}
        # for bench_type, times in scheduling_times.items():
        #     avg_times[bench_type.split("_", 2)[-1]] = sum(times) / len(times)
        # fig = plot_benchmark(avg_times, "average scheduling time")
        # fig.savefig(os.path.join(res_path, 'avg_scheduling_time.png'))
        # plt.close()

        # plot scheduling dist
        for bench_mode, res in benchmarks["scheduling_dist"].items():
            print("Processing scheduling dist", bench_mode)
            fig = plot_scheduling_dist(res["results"])
            for iter in res["results"]:
                for thread_id, tasks in iter.items():
                    unique_cpus = set(t["cpu"] for t in tasks)
                    if len(unique_cpus) > 1:
                        print(f"{bench_mode}: thread {thread_id} has tasks executed on differenet cpus: {unique_cpus}")
            fig.savefig(os.path.join(res_path, "scheduling_dist_" + bench_mode + '.png'), bbox_inches='tight')
            plt.close()
