// Multivariate tensor-product B-spline basis (paper Eqs. 4-5):
//   Phi_j(x) = prod_i phi_{j_i}(x_i),  N = prod_i N_i.
//
// Multi-indices are flattened in row-major (C) order - dimension 0 is the
// slowest-varying index - so a NumPy reshape of the coefficient vector to
// `shape` recovers the tensor layout, and the multivariate Gram matrix is
// exactly kron(G_0, kron(G_1, ...)).
#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fpe/bspline.hpp"
#include "fpe/gauss.hpp"

namespace fpe {

constexpr int kMaxDims = 8;

class TensorBasis {
public:
    TensorBasis(const std::vector<double>& lo, const std::vector<double>& hi,
                const std::vector<int>& n_basis, const std::vector<int>& order) {
        const size_t n = lo.size();
        if (n < 1 || n > kMaxDims)
            throw std::invalid_argument("TensorBasis: dimension must be in [1, 8]");
        if (hi.size() != n || n_basis.size() != n || order.size() != n)
            throw std::invalid_argument("TensorBasis: inconsistent argument sizes");
        splines_.reserve(n);
        for (size_t d = 0; d < n; ++d) {
            splines_.emplace_back(lo[d], hi[d], n_basis[d], order[d]);
        }
        strides_.assign(n, 1);
        n_total_ = 1;
        for (int d = static_cast<int>(n) - 1; d >= 0; --d) {
            strides_[d] = n_total_;
            n_total_ *= n_basis[d];
        }
    }

    int dim() const { return static_cast<int>(splines_.size()); }
    std::int64_t n_total() const { return n_total_; }
    const BSpline1D& spline(int d) const { return splines_.at(d); }
    std::int64_t stride(int d) const { return strides_[d]; }

    std::vector<int> shape() const {
        std::vector<int> s(dim());
        for (int d = 0; d < dim(); ++d) s[d] = splines_[d].n_basis();
        return s;
    }

    // Number of knot spans (elements) per dimension and in total.
    std::vector<int> element_shape() const {
        std::vector<int> s(dim());
        for (int d = 0; d < dim(); ++d) s[d] = splines_[d].n_spans();
        return s;
    }
    std::int64_t n_elements() const {
        std::int64_t n = 1;
        for (int d = 0; d < dim(); ++d) n *= splines_[d].n_spans();
        return n;
    }

    // Flat element id of the point x (row-major over per-dim span indices).
    std::int64_t element_of(const double* x) const {
        std::int64_t id = 0;
        for (int d = 0; d < dim(); ++d) {
            const int e = splines_[d].find_span(x[d]) - splines_[d].degree();
            id = id * splines_[d].n_spans() + e;
        }
        return id;
    }

    // Tensor-product Gauss-Legendre quadrature with q points per knot span
    // per dimension. Exact for the Gram matrix (q >= order) and highly
    // accurate for the M matrix when the dynamics is smooth; the recommended
    // rule for low-dimensional problems. Returns X (n_pts x dim, row-major)
    // and weights W.
    void element_quadrature(int q, Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X,
                            Eigen::Ref<Eigen::VectorXd> W) const {
        const int n = dim();
        std::vector<std::vector<double>> px(n), pw(n);
        std::vector<double> gx, gw;
        gauss_legendre(q, gx, gw);
        std::vector<std::int64_t> sizes(n);
        for (int d = 0; d < n; ++d) {
            const BSpline1D& sp = splines_[d];
            const auto& knots = sp.knots();
            for (int s = 0; s < sp.n_spans(); ++s) {
                const int span = sp.degree() + s;
                const double ta = knots[span], tb = knots[span + 1];
                const double mid = 0.5 * (ta + tb), half = 0.5 * (tb - ta);
                for (int i = 0; i < q; ++i) {
                    px[d].push_back(mid + half * gx[i]);
                    pw[d].push_back(half * gw[i]);
                }
            }
            sizes[d] = static_cast<std::int64_t>(px[d].size());
        }
        std::int64_t total = 1;
        for (int d = 0; d < n; ++d) total *= sizes[d];
        if (X.rows() != total || X.cols() != n || W.size() != total)
            throw std::invalid_argument("element_quadrature: bad output shape");
        for (std::int64_t idx = 0; idx < total; ++idx) {
            std::int64_t rem = idx;
            double w = 1.0;
            for (int d = n - 1; d >= 0; --d) {
                const std::int64_t i = rem % sizes[d];
                rem /= sizes[d];
                X(idx, d) = px[d][static_cast<size_t>(i)];
                w *= pw[d][static_cast<size_t>(i)];
            }
            W(idx) = w;
        }
    }

    std::int64_t element_quadrature_size(int q) const {
        std::int64_t total = 1;
        for (int d = 0; d < dim(); ++d) {
            total *= static_cast<std::int64_t>(splines_[d].n_spans()) * q;
        }
        return total;
    }

private:
    std::vector<BSpline1D> splines_;
    std::vector<std::int64_t> strides_;
    std::int64_t n_total_{1};
};

}  // namespace fpe
