// Minimal portable thread-pool-free parallel_for built on std::thread
// (avoids an OpenMP dependency, which is awkward on macOS/Apple Clang).
#pragma once

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace fpe {

inline int resolve_threads(int n_threads) {
    if (n_threads > 0) return n_threads;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 4;
}

// Splits [0, n) into contiguous chunks and calls fn(begin, end, thread_id)
// on each from its own thread.
template <class Fn>
void parallel_for(std::int64_t n, int n_threads, Fn&& fn) {
    n_threads = std::min<std::int64_t>(resolve_threads(n_threads), std::max<std::int64_t>(n, 1));
    if (n_threads <= 1) {
        fn(static_cast<std::int64_t>(0), n, 0);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    const std::int64_t chunk = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        const std::int64_t begin = t * chunk;
        const std::int64_t end = std::min(n, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back([&fn, begin, end, t]() { fn(begin, end, t); });
    }
    for (auto& w : workers) w.join();
}

}  // namespace fpe
