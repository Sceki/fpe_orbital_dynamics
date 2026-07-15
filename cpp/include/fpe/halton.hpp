// Halton low-discrepancy sequence (Sec. 4.2 of the paper): deterministic
// quasi-Monte Carlo points with guaranteed error bounds and faster
// convergence than plain Monte Carlo for the high-dimensional Galerkin
// integrals.
#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <stdexcept>

namespace fpe {

inline const int* halton_primes() {
    static const int primes[25] = {2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37, 41,
                                   43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    return primes;
}

// Radical inverse of index i in the given base.
inline double radical_inverse(std::uint64_t i, int base) {
    double f = 1.0, r = 0.0;
    while (i > 0) {
        f /= base;
        r += f * static_cast<double>(i % base);
        i /= base;
    }
    return r;
}

// n points of the dim-dimensional Halton sequence in [0, 1)^dim, starting at
// index `skip` (skipping index 0 avoids the degenerate all-zeros point).
inline Eigen::MatrixXd halton(std::int64_t n, int dim, std::int64_t skip = 1) {
    if (dim < 1 || dim > 25) throw std::invalid_argument("halton: dim must be in [1, 25]");
    if (n < 1) throw std::invalid_argument("halton: n must be >= 1");
    Eigen::MatrixXd pts(n, dim);
    for (int d = 0; d < dim; ++d) {
        const int base = halton_primes()[d];
        for (std::int64_t i = 0; i < n; ++i) {
            pts(i, d) = radical_inverse(static_cast<std::uint64_t>(i + skip), base);
        }
    }
    return pts;
}

}  // namespace fpe
