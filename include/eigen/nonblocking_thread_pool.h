// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
#define EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
#define EIGEN_POOL_RUNNEXT

#include "run_queue.h"
#include "stl_thread_env.h"
#include "max_size_vector.h"
#include <atomic>
#include <functional>
#include <thread>

namespace Eigen {

// This defines an interface that ThreadPoolDevice can take to use
// custom thread pools underneath.
class ThreadPoolInterface {
public:
  // Submits a closure to be run by a thread in the pool.
  virtual void Schedule(std::function<void()> fn) = 0;

  // Submits a closure to be run by threads in the range [start, end) in the
  // pool.
  virtual void ScheduleWithHint(std::function<void()> fn, int /*start*/,
                                int /*end*/) {
    // Just defer to Schedule in case sub-classes aren't interested in
    // overriding this functionality.
    Schedule(fn);
  }

  // If implemented, stop processing the closures that have been enqueued.
  // Currently running closures may still be processed.
  // If not implemented, does nothing.
  virtual void Cancel() {}

  // Returns the number of threads in the pool.
  virtual int NumThreads() const = 0;

  // Returns a logical thread index between 0 and NumThreads() - 1 if called
  // from one of the threads in the pool. Returns -1 otherwise.
  virtual int CurrentThreadId() const = 0;

  virtual ~ThreadPoolInterface() {}
};

template <typename Environment>
class ThreadPoolTempl : public Eigen::ThreadPoolInterface {
public:
  typedef typename Environment::Task Task;
  typedef RunQueue<Task, 1024> Queue;

  ThreadPoolTempl(int num_threads, Environment env = Environment())
      : ThreadPoolTempl(num_threads, true, false, env) {}

  ThreadPoolTempl(int num_threads, bool allow_spinning, bool use_main_thread,
                  Environment env = Environment())
      : env_(env), num_threads_(num_threads), allow_spinning_(allow_spinning),
        thread_data_(num_threads), all_coprimes_(num_threads),
        global_steal_partition_(EncodePartition(0, num_threads_)), blocked_(0),
        spinning_(0), done_(false), cancelled_(false) {
    // Calculate coprimes of all numbers [1, num_threads].
    // Coprimes are used for random walks over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a random starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    assert(num_threads_ < kMaxThreads);
    for (int i = 1; i <= num_threads_; ++i) {
      all_coprimes_.emplace_back(i);
      ComputeCoprimes(i, &all_coprimes_.back());
    }
    thread_data_.resize(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
      SetStealPartition(i, EncodePartition(0, num_threads_));
      if (i == 0) {
        PerThread *pt = GetPerThread();
        pt->pool = this;
        pt->rand = GlobalThreadIdHash();
        pt->thread_id = i;
      } else {
        thread_data_[i].thread.reset(env_.CreateThread([this, i]() {
          PerThread *pt = GetPerThread();
          pt->pool = this;
          pt->rand = GlobalThreadIdHash();
          pt->thread_id = i;
          WorkerLoop();
        }));
      }
    }
  }

  ~ThreadPoolTempl() {
    done_ = true;

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    if (cancelled_) {
      // Since we were cancelled, there might be entries in the queues.
      // Empty them to prevent their destructor from asserting.
      for (size_t i = 0; i < thread_data_.size(); i++) {
        thread_data_[i].queue.Flush();
      }
    }
    // Join threads explicitly (by destroying) to avoid destruction order within
    // this class.
    for (size_t i = 0; i < thread_data_.size(); ++i)
      thread_data_[i].thread.reset();
  }

  void SetStealPartitions(
      const std::vector<std::pair<unsigned, unsigned>> &partitions) {
    assert(partitions.size() == static_cast<std::size_t>(num_threads_));

    // Pass this information to each thread queue.
    for (int i = 0; i < num_threads_; i++) {
      const auto &pair = partitions[i];
      unsigned start = pair.first, end = pair.second;
      AssertBounds(start, end);
      unsigned val = EncodePartition(start, end);
      SetStealPartition(i, val);
    }
  }

  void Schedule(std::function<void()> fn) override {
    // schedule on main thread only when explicitly requested
    ScheduleWithHint(std::move(fn), 0, num_threads_);
  }

  void RunOnThread(std::function<void()> fn, size_t threadIndex) {
    threadIndex = threadIndex % num_threads_;
    Task t = env_.CreateTask(std::move(fn));
    auto p = new Task(std::move(t));
#ifdef EIGEN_POOL_RUNNEXT
    Task *expected = nullptr;
    if (thread_data_[threadIndex].runnext.compare_exchange_strong(
            expected, p, std::memory_order_release)) {
      return;
    }
#endif
    Queue &q = thread_data_[threadIndex].queue;
    t = q.PushBack(std::move(*p));
    if (t.f) {
      // failed to push, execute directly
      env_.ExecuteTask(t);
    }
  }

  void ScheduleWithHint(std::function<void()> fn, int start,
                        int limit) override {
    Task t = env_.CreateTask(std::move(fn));
    PerThread *pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue &q = thread_data_[pt->thread_id].queue;
      t = q.PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      assert(start < limit);
      assert(limit <= num_threads_);
      int num_queues = limit - start;
      int rnd = Rand(&pt->rand) % num_queues;
      assert(start + rnd < limit);
      Queue &q = thread_data_[start + rnd].queue;
      t = q.PushBack(std::move(t)); // TODO(vorkdenis): try to push front?
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (t.f) {
      env_.ExecuteTask(t); // Push failed, execute directly.
    }
  }

  void Cancel() override {
    cancelled_ = true;
    done_ = true;

    // Let each thread know it's been cancelled.
#ifdef EIGEN_THREAD_ENV_SUPPORTS_CANCELLATION
    for (size_t i = 0; i < thread_data_.size(); i++) {
      thread_data_[i].thread->OnCancel();
    }
#endif

    // Wake up the threads without work to let them exit on their own.
  }

  int NumThreads() const final { return num_threads_; }

  int CurrentThreadId() const final {
    const PerThread *pt = const_cast<ThreadPoolTempl *>(this)->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

  void JoinMainThread() { WorkerLoop(/* external */ true); }

private:
  // Create a single atomic<int> that encodes start and limit information for
  // each thread.
  // We expect num_threads_ < 65536, so we can store them in a single
  // std::atomic<unsigned>.
  // Exposed publicly as static functions so that external callers can reuse
  // this encode/decode logic for maintaining their own thread-safe copies of
  // scheduling and steal domain(s).
  static const int kMaxPartitionBits = 16;
  static const int kMaxThreads = 1 << kMaxPartitionBits;

  inline unsigned EncodePartition(unsigned start, unsigned limit) {
    return (start << kMaxPartitionBits) | limit;
  }

  inline void DecodePartition(unsigned val, unsigned *start, unsigned *limit) {
    *limit = val & (kMaxThreads - 1);
    val >>= kMaxPartitionBits;
    *start = val;
  }

  void AssertBounds(int start, int end) {
    assert(start >= 0);
    assert(start < end); // non-zero sized partition
    assert(end <= num_threads_);
  }

  inline void SetStealPartition(size_t i, unsigned val) {
    thread_data_[i].steal_partition.store(val, std::memory_order_relaxed);
  }

  inline unsigned GetStealPartition(int i) {
    return thread_data_[i].steal_partition.load(std::memory_order_relaxed);
  }

  void ComputeCoprimes(int N, MaxSizeVector<unsigned> *coprimes) {
    for (int i = 1; i <= N; i++) {
      unsigned a = i;
      unsigned b = N;
      // If GCD(a, b) == 1, then a and b are coprimes.
      while (b != 0) {
        unsigned tmp = a;
        a = b;
        b = tmp % b;
      }
      if (a == 1) {
        coprimes->push_back(i);
      }
    }
  }

  typedef typename Environment::EnvThread Thread;

  struct PerThread {
    constexpr PerThread() : pool(NULL), rand(0), thread_id(-1) {}
    ThreadPoolTempl *pool; // Parent pool, or null for normal threads.
    uint64_t rand;         // Random generator state.
    int thread_id;         // Worker thread index in pool.
  };

  struct ThreadData {
    constexpr ThreadData()
        : thread(), steal_partition(0), queue(), runnext(nullptr) {}
    std::unique_ptr<Thread> thread;
    std::atomic<unsigned> steal_partition;
    Queue queue;
    std::atomic<Task *> runnext;
  };

  Environment env_;
  const int num_threads_;
  const bool allow_spinning_;
  MaxSizeVector<ThreadData> thread_data_;
  MaxSizeVector<MaxSizeVector<unsigned>> all_coprimes_;
  unsigned global_steal_partition_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> spinning_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;

  // Main worker thread loop.
  void WorkerLoop(bool external = false) {
    // TODO: init in constructor?
    PerThread *pt = GetPerThread();
    auto thread_id = pt->thread_id;
    Queue &q = thread_data_[thread_id].queue;
    while (!cancelled_) {
      Task t;
#ifdef EIGEN_POOL_RUNNEXT
      auto p = thread_data_[thread_id].runnext.load(std::memory_order_relaxed);
      if (p) {
        auto success = thread_data_[thread_id].runnext.compare_exchange_strong(
            p, nullptr, std::memory_order_acquire);
        if (success) {
          t = std::move(*p);
          delete p;
        }
      }
#endif
      if (!t.f) {
        t = q.PopFront();
      }
      if (!t.f) {
        t = LocalSteal();
      }
      if (!t.f) {
        t = GlobalSteal();
      }
      if (!t.f && external) {
        // external thread shouldn't wait for work, it should just exit.
        return;
      }
      if (t.f) {
        env_.ExecuteTask(t);
      } else if (done_) {
        return;
      }
    }
  }

  // Steal tries to steal work from other worker threads in the range [start,
  // limit) in best-effort manner.
  Task Steal(unsigned start, unsigned limit) {
    PerThread *pt = GetPerThread();
    const size_t size = limit - start;
    unsigned r = Rand(&pt->rand);
    // Reduce r into [0, size) range, this utilizes trick from
    // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    assert(all_coprimes_[size - 1].size() < (1 << 30));
    unsigned victim = ((uint64_t)r * (uint64_t)size) >> 32;
    unsigned index =
        ((uint64_t)all_coprimes_[size - 1].size() * (uint64_t)r) >> 32;
    unsigned inc = all_coprimes_[size - 1][index];

    for (unsigned i = 0; i < size; i++) {
      assert(start + victim < limit);
      Task t = thread_data_[start + victim].queue.PopBack();
      if (t.f) {
        return t;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
#ifdef EIGEN_POOL_RUNNEXT
    for (unsigned i = 0; i < size; i++) {
      auto p =
          thread_data_[start + victim].runnext.load(std::memory_order_relaxed);
      if (p) {
        auto success =
            thread_data_[start + victim].runnext.compare_exchange_strong(
                p, nullptr, std::memory_order_acquire);
        if (success) {
          Task t = std::move(*p);
          delete p;
          return t;
        }
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
#endif
    return Task();
  }

  // Steals work within threads belonging to the partition.
  Task LocalSteal() {
    PerThread *pt = GetPerThread();
    unsigned partition = GetStealPartition(pt->thread_id);
    // If thread steal partition is the same as global partition, there is no
    // need to go through the steal loop twice.
    if (global_steal_partition_ == partition)
      return Task();
    unsigned start, limit;
    DecodePartition(partition, &start, &limit);
    AssertBounds(start, limit);

    return Steal(start, limit);
  }

  // Steals work from any other thread in the pool.
  Task GlobalSteal() { return Steal(0, num_threads_); }

  int NonEmptyQueueIndex() {
    PerThread *pt = GetPerThread();
    // We intentionally design NonEmptyQueueIndex to steal work from
    // anywhere in the queue so threads don't block in WaitForWork() forever
    // when all threads in their partition go to sleep. Steal is still local.
    const size_t size = thread_data_.size();
    unsigned r = Rand(&pt->rand);
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      if (!thread_data_[victim].queue.Empty()) {
        return victim;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return -1;
  }

  static __attribute__((always_inline)) inline uint64_t GlobalThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }

  __attribute__((always_inline)) inline PerThread *GetPerThread() {
    static thread_local PerThread per_thread_;
    PerThread *pt = &per_thread_;
    return pt;
  }

  static __attribute__((always_inline)) inline unsigned Rand(uint64_t *state) {
    uint64_t current = *state;
    // Update the internal state
    *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22)) >>
                                 (22 + (current >> 61)));
  }
};

typedef ThreadPoolTempl<StlThreadEnvironment> ThreadPool;

} // namespace Eigen

#endif // EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H