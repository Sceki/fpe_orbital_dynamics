// One-dimensional B-spline basis on a clamped uniform knot vector
// (paper Sec. 4.1, Eqs. 14-17).
//
// An order-k B-spline basis function is a piecewise polynomial of degree
// k-1; basis function i is supported on [t_i, t_{i+k}], which is what makes
// the Galerkin matrices banded: <phi_i, phi_j> = 0 whenever |i - j| >= k.
// Values and derivatives are evaluated with the standard, numerically stable
// triangular scheme (Piegl & Tiller, "The NURBS Book", algorithms A2.1-A2.3),
// which is mathematically identical to the Cox-de Boor recursion of Eq. (14)
// and its derivative formulas (Eqs. 16-17).
#pragma once

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "fpe/gauss.hpp"

namespace fpe {

constexpr int kMaxOrder = 8;  // supports degree up to 7; the paper uses order 3

class BSpline1D {
public:
    BSpline1D(double lo, double hi, int n_basis, int order)
        : lo_(lo), hi_(hi), n_(n_basis), k_(order), p_(order - 1) {
        if (!(hi > lo)) throw std::invalid_argument("BSpline1D: hi must be > lo");
        if (order < 2 || order > kMaxOrder)
            throw std::invalid_argument("BSpline1D: order must be in [2, 8]");
        if (n_basis < order)
            throw std::invalid_argument("BSpline1D: n_basis must be >= order");
        // Clamped uniform knot vector with n_basis + order knots: the first
        // and last `order` knots are repeated at the boundary so that the
        // basis spans all polynomials of degree order-1 on [lo, hi].
        const int n_spans = n_ - k_ + 1;
        const double h = (hi_ - lo_) / n_spans;
        knots_.resize(n_ + k_);
        for (int i = 0; i < n_ + k_; ++i) {
            if (i < k_) {
                knots_[i] = lo_;
            } else if (i >= n_) {
                knots_[i] = hi_;
            } else {
                knots_[i] = lo_ + h * (i - k_ + 1);
            }
        }
    }

    double lo() const { return lo_; }
    double hi() const { return hi_; }
    int n_basis() const { return n_; }
    int order() const { return k_; }
    int degree() const { return p_; }
    int n_spans() const { return n_ - k_ + 1; }
    const std::vector<double>& knots() const { return knots_; }

    // Knot span index i such that knots[i] <= x < knots[i+1], clamped so
    // that boundary points fall in the first/last non-empty span
    // (NURBS Book A2.1). Valid spans are p_ .. n_-1.
    int find_span(double x) const {
        if (x >= hi_) return n_ - 1;
        if (x <= lo_) return p_;
        int low = p_, high = n_;
        int mid = (low + high) / 2;
        while (x < knots_[mid] || x >= knots_[mid + 1]) {
            if (x < knots_[mid]) {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        return mid;
    }

    // Values and derivatives of the `order` basis functions that are nonzero
    // on the span of x. ders is a (nders+1) x order row-major array:
    // ders[d*order + j] = d-th derivative of basis function (span - degree + j)
    // at x. Derivatives of order > degree are exactly zero.
    // (NURBS Book A2.3 "DersBasisFuns".)
    void ders_basis_funs(int span, double x, int nders, double* ders) const {
        const int p = p_;
        double ndu[kMaxOrder][kMaxOrder];
        double a[2][kMaxOrder];
        double left[kMaxOrder], right[kMaxOrder];

        ndu[0][0] = 1.0;
        for (int j = 1; j <= p; ++j) {
            left[j] = x - knots_[span + 1 - j];
            right[j] = knots_[span + j] - x;
            double saved = 0.0;
            for (int r = 0; r < j; ++r) {
                ndu[j][r] = right[r + 1] + left[j - r];
                const double temp = ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        std::memset(ders, 0, sizeof(double) * static_cast<size_t>(nders + 1) * k_);
        for (int j = 0; j <= p; ++j) ders[j] = ndu[j][p];

        const int nd = std::min(nders, p);
        for (int r = 0; r <= p; ++r) {
            int s1 = 0, s2 = 1;
            a[0][0] = 1.0;
            for (int kk = 1; kk <= nd; ++kk) {
                double d = 0.0;
                const int rk = r - kk;
                const int pk = p - kk;
                if (r >= kk) {
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                    d = a[s2][0] * ndu[rk][pk];
                }
                const int j1 = (rk >= -1) ? 1 : -rk;
                const int j2 = (r - 1 <= pk) ? kk - 1 : p - r;
                for (int j = j1; j <= j2; ++j) {
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                    d += a[s2][j] * ndu[rk + j][pk];
                }
                if (r <= pk) {
                    a[s2][kk] = -a[s1][kk - 1] / ndu[pk + 1][r];
                    d += a[s2][kk] * ndu[r][pk];
                }
                ders[kk * k_ + r] = d;
                std::swap(s1, s2);
            }
        }

        // Multiply through by the correct factors (NURBS Book, end of A2.3).
        double r = static_cast<double>(p);
        for (int kk = 1; kk <= nd; ++kk) {
            for (int j = 0; j <= p; ++j) ders[kk * k_ + j] *= r;
            r *= static_cast<double>(p - kk);
        }
    }

    // Dense basis matrix: entry (i, j) = d-th derivative of basis j at x[i].
    // Convenience for tests and plotting (not used in hot loops).
    std::vector<double> basis_matrix(const std::vector<double>& x, int der) const {
        std::vector<double> out(x.size() * static_cast<size_t>(n_), 0.0);
        std::vector<double> ders(static_cast<size_t>(der + 1) * k_);
        for (size_t i = 0; i < x.size(); ++i) {
            const int span = find_span(x[i]);
            ders_basis_funs(span, x[i], der, ders.data());
            const int first = span - p_;
            for (int j = 0; j < k_; ++j) {
                out[i * n_ + (first + j)] = ders[static_cast<size_t>(der) * k_ + j];
            }
        }
        return out;
    }

    // Exact Gram matrix G_ij = \int phi_i phi_j dx (dense n x n, banded with
    // bandwidth order-1), integrated span-by-span with Gauss-Legendre of
    // order k_, which is exact for the degree 2(k_-1) integrand.
    std::vector<double> gram() const {
        std::vector<double> G(static_cast<size_t>(n_) * n_, 0.0);
        std::vector<double> gx, gw, ders(static_cast<size_t>(k_));
        gauss_legendre(k_, gx, gw);
        for (int s = 0; s < n_spans(); ++s) {
            const int span = p_ + s;
            const double ta = knots_[span], tb = knots_[span + 1];
            const double mid = 0.5 * (ta + tb), half = 0.5 * (tb - ta);
            for (int q = 0; q < k_; ++q) {
                const double x = mid + half * gx[q];
                const double w = half * gw[q];
                ders_basis_funs(span, x, 0, ders.data());
                const int first = span - p_;
                for (int i = 0; i < k_; ++i) {
                    for (int j = 0; j < k_; ++j) {
                        G[static_cast<size_t>(first + i) * n_ + (first + j)] +=
                            w * ders[i] * ders[j];
                    }
                }
            }
        }
        return G;
    }

    // Exact moment integrals of each basis function:
    //   I0_j = \int phi_j dx,  I1_j = \int x phi_j dx,  I2_j = \int x^2 phi_j dx.
    // Used to compute pdf mass and moments in closed form (the basis is
    // separable, so multivariate moments factor into these 1D tables).
    void integrals(std::vector<double>& I0, std::vector<double>& I1,
                   std::vector<double>& I2) const {
        I0.assign(n_, 0.0);
        I1.assign(n_, 0.0);
        I2.assign(n_, 0.0);
        std::vector<double> gx, gw, ders(static_cast<size_t>(k_));
        const int q_order = k_ + 1;  // exact for degree (k_-1) + 2
        gauss_legendre(q_order, gx, gw);
        for (int s = 0; s < n_spans(); ++s) {
            const int span = p_ + s;
            const double ta = knots_[span], tb = knots_[span + 1];
            const double mid = 0.5 * (ta + tb), half = 0.5 * (tb - ta);
            for (int q = 0; q < q_order; ++q) {
                const double x = mid + half * gx[q];
                const double w = half * gw[q];
                ders_basis_funs(span, x, 0, ders.data());
                const int first = span - p_;
                for (int j = 0; j < k_; ++j) {
                    I0[first + j] += w * ders[j];
                    I1[first + j] += w * x * ders[j];
                    I2[first + j] += w * x * x * ders[j];
                }
            }
        }
    }

private:
    double lo_, hi_;
    int n_;  // number of basis functions
    int k_;  // order (degree + 1)
    int p_;  // degree
    std::vector<double> knots_;
};

}  // namespace fpe
