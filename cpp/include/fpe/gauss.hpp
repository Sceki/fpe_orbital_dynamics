// Gauss-Legendre quadrature nodes/weights on [-1, 1], computed with Newton
// iterations on the Legendre polynomial (machine-precision accurate for the
// small orders used here).
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace fpe {

inline void gauss_legendre(int q, std::vector<double>& x, std::vector<double>& w) {
    if (q < 1) throw std::invalid_argument("gauss_legendre: q must be >= 1");
    x.assign(q, 0.0);
    w.assign(q, 0.0);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < q; ++i) {
        // Chebyshev-like initial guess for the i-th root.
        double xi = std::cos(pi * (i + 0.75) / (q + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            // Legendre recurrence: P_0 = 1, P_1 = x.
            double p0 = 1.0, p1 = xi;
            for (int k = 2; k <= q; ++k) {
                const double p2 = ((2.0 * k - 1.0) * xi * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            // Derivative from P_q and P_{q-1}.
            dp = q * (xi * p1 - p0) / (xi * xi - 1.0);
            const double dx = p1 / dp;
            xi -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        x[i] = xi;
        w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
    }
}

}  // namespace fpe
