// Assembly of the Galerkin-projected Fokker-Planck operator (paper Eqs. 8-10).
//
//   B_kj = <Phi_k, Phi_j>
//   M_kj = <Phi_k, -sum_i d(f_i Phi_j)/dx_i + sum_{i,l} d^2(D_il Phi_j)/(dx_i dx_l)>
//
// Expanding the derivatives, the integrand against Phi_k for each j is
//   r_j(x) = -f(x) . grad Phi_j(x) - div f(x) Phi_j(x)
//            + sum_{i,l} D_il(x) d^2 Phi_j/(dx_i dx_l)
//            + 2 sum_i [sum_l dD_il/dx_l] dPhi_j/dx_i
//            + [sum_{i,l} d^2 D_il/(dx_i dx_l)] Phi_j(x),
// where the last two lines vanish for constant diffusion.
//
// Sparsity (paper Sec. 4.1): Phi_k Phi_j and all their derivative products
// vanish unless |k_i - j_i| < order for every dimension i, so both matrices
// are banded Kronecker-structured and are assembled directly in sparse form.
//
// Efficiency: instead of computing one integral per (k, j) pair, quadrature
// points are grouped by knot-span element. All points of an element share
// the same set of `order^dim` active basis functions, so each point
// contributes a dense (K x K) local block via an outer product, and each
// element scatters its block into global triplets once. The cost is
// O(n_points * K^2) regardless of the total basis count N. Works identically
// for tensor Gauss-Legendre points and for Halton quasi-Monte Carlo points
// (Sec. 4.2 of the paper).
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "fpe/tensor_basis.hpp"
#include "fpe/threading.hpp"

namespace fpe {

using SpMat = Eigen::SparseMatrix<double>;
using RowMatC = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace detail {

// Precomputed per-element machinery shared by the assembly routines.
struct ElementGroups {
    std::vector<std::int64_t> order;    // point indices sorted by element
    std::vector<std::int64_t> offsets;  // CSR-style offsets into `order`, size n_elements+1
};

inline ElementGroups group_points_by_element(const TensorBasis& basis,
                                             const Eigen::Ref<const RowMatC>& X) {
    const std::int64_t n_pts = X.rows();
    const std::int64_t n_el = basis.n_elements();
    std::vector<std::int64_t> elem_of(n_pts);
    for (std::int64_t i = 0; i < n_pts; ++i) {
        elem_of[i] = basis.element_of(X.row(i).data());
    }
    ElementGroups g;
    g.offsets.assign(n_el + 1, 0);
    for (std::int64_t i = 0; i < n_pts; ++i) g.offsets[elem_of[i] + 1]++;
    for (std::int64_t e = 0; e < n_el; ++e) g.offsets[e + 1] += g.offsets[e];
    g.order.resize(n_pts);
    std::vector<std::int64_t> cursor(g.offsets.begin(), g.offsets.end() - 1);
    for (std::int64_t i = 0; i < n_pts; ++i) {
        g.order[cursor[elem_of[i]]++] = i;
    }
    return g;
}

// Local multi-index tables for one element: the K = prod(order_d) active
// basis functions, their global-index offsets, and per-dim local offsets.
struct LocalIndexTable {
    int K{1};
    int dim{0};
    std::vector<std::array<int, kMaxDims>> offs;  // per local index, per dim
};

inline LocalIndexTable make_local_table(const TensorBasis& basis) {
    LocalIndexTable t;
    t.dim = basis.dim();
    for (int d = 0; d < t.dim; ++d) t.K *= basis.spline(d).order();
    t.offs.resize(t.K);
    for (int j = 0; j < t.K; ++j) {
        int rem = j;
        for (int d = t.dim - 1; d >= 0; --d) {
            const int kd = basis.spline(d).order();
            t.offs[j][d] = rem % kd;
            rem /= kd;
        }
    }
    return t;
}

// Decompose flat element id into per-dim element indices (row-major).
inline void element_multi_index(const TensorBasis& basis, std::int64_t elem, int* e_out) {
    for (int d = basis.dim() - 1; d >= 0; --d) {
        const int s = basis.spline(d).n_spans();
        e_out[d] = static_cast<int>(elem % s);
        elem /= s;
    }
}

}  // namespace detail

// Assemble the drift+diffusion matrix M from quadrature data.
//   X:    (n_pts x dim) quadrature points, W: weights
//   F:    (n_pts x dim) drift f at the points, divF: divergence of f
//   Dconst: (dim x dim) constant diffusion matrix D = sigma sigma^T / 2
// Optional state-dependent diffusion (pass empty matrices to disable):
//   Dpt:   (n_pts x dim*dim) D(x) row-major per point (overrides Dconst)
//   dDrow: (n_pts x dim) entries sum_l dD_il/dx_l
//   ddD:   (n_pts) entries sum_{i,l} d^2 D_il/(dx_i dx_l)
inline SpMat assemble_M(const TensorBasis& basis, const Eigen::Ref<const RowMatC>& X,
                        const Eigen::Ref<const Eigen::VectorXd>& W,
                        const Eigen::Ref<const RowMatC>& F,
                        const Eigen::Ref<const Eigen::VectorXd>& divF,
                        const Eigen::MatrixXd& Dconst, const RowMatC& Dpt, const RowMatC& dDrow,
                        const Eigen::VectorXd& ddD, int n_threads) {
    const int n = basis.dim();
    const std::int64_t n_pts = X.rows();
    if (X.cols() != n) throw std::invalid_argument("assemble_M: X has wrong dimension");
    if (W.size() != n_pts || F.rows() != n_pts || F.cols() != n || divF.size() != n_pts)
        throw std::invalid_argument("assemble_M: inconsistent quadrature arrays");
    const bool state_dep_D = Dpt.rows() > 0;
    if (state_dep_D && (Dpt.rows() != n_pts || Dpt.cols() != n * n || dDrow.rows() != n_pts ||
                        dDrow.cols() != n || ddD.size() != n_pts))
        throw std::invalid_argument("assemble_M: inconsistent state-dependent diffusion arrays");
    if (Dconst.rows() != n || Dconst.cols() != n)
        throw std::invalid_argument("assemble_M: Dconst must be dim x dim");

    const bool has_diffusion =
        state_dep_D || Dconst.cwiseAbs().maxCoeff() > 0.0;

    const auto groups = detail::group_points_by_element(basis, X);
    const auto table = detail::make_local_table(basis);
    const int K = table.K;
    const std::int64_t n_el = basis.n_elements();

    // Non-empty elements only (Halton points may leave gaps if too few).
    std::vector<std::int64_t> active_elems;
    active_elems.reserve(n_el);
    for (std::int64_t e = 0; e < n_el; ++e) {
        if (groups.offsets[e + 1] > groups.offsets[e]) active_elems.push_back(e);
    }

    const int nt = resolve_threads(n_threads);
    std::vector<std::vector<Eigen::Triplet<double>>> triplets(nt);

    parallel_for(static_cast<std::int64_t>(active_elems.size()), nt,
                 [&](std::int64_t eb, std::int64_t ee, int tid) {
        auto& trip = triplets[tid];
        std::vector<double> block(static_cast<size_t>(K) * K);
        std::vector<double> val(K), rj(K), grad(static_cast<size_t>(K) * n);
        // Per-dim basis values/derivatives at one point: [d][der*order + j].
        std::vector<std::vector<double>> ders(n);
        for (int d = 0; d < n; ++d) ders[d].resize(3 * static_cast<size_t>(basis.spline(d).order()));
        int emi[kMaxDims];

        for (std::int64_t ei = eb; ei < ee; ++ei) {
            const std::int64_t elem = active_elems[ei];
            std::fill(block.begin(), block.end(), 0.0);
            detail::element_multi_index(basis, elem, emi);

            for (std::int64_t pi = groups.offsets[elem]; pi < groups.offsets[elem + 1]; ++pi) {
                const std::int64_t pt = groups.order[pi];
                const double w = W(pt);

                for (int d = 0; d < n; ++d) {
                    const BSpline1D& sp = basis.spline(d);
                    const int span = emi[d] + sp.degree();
                    sp.ders_basis_funs(span, X(pt, d), 2, ders[d].data());
                }

                // Diffusion matrix at this point.
                const double* Dmat = nullptr;
                if (state_dep_D) Dmat = Dpt.row(pt).data();

                for (int j = 0; j < K; ++j) {
                    const auto& off = table.offs[j];
                    // Phi_j and its gradient.
                    double v = 1.0;
                    for (int d = 0; d < n; ++d) v *= ders[d][off[d]];
                    val[j] = v;
                    for (int i = 0; i < n; ++i) {
                        double gi = 1.0;
                        for (int d = 0; d < n; ++d) {
                            const int kd = basis.spline(d).order();
                            gi *= (d == i) ? ders[d][kd + off[d]] : ders[d][off[d]];
                        }
                        grad[static_cast<size_t>(j) * n + i] = gi;
                    }
                    // Drift part: -f . grad Phi_j - div f * Phi_j.
                    double r = -divF(pt) * v;
                    for (int i = 0; i < n; ++i) {
                        r -= F(pt, i) * grad[static_cast<size_t>(j) * n + i];
                    }
                    // Diffusion part: sum_{i,l} D_il d^2 Phi_j / dx_i dx_l.
                    if (has_diffusion) {
                        double hd = 0.0;
                        for (int i = 0; i < n; ++i) {
                            for (int l = i; l < n; ++l) {
                                const double Dil = state_dep_D
                                                       ? (i == l ? Dmat[i * n + i]
                                                                 : Dmat[i * n + l] + Dmat[l * n + i])
                                                       : (i == l ? Dconst(i, i)
                                                                 : Dconst(i, l) + Dconst(l, i));
                                if (Dil == 0.0) continue;
                                double prod = 1.0;
                                for (int d = 0; d < n; ++d) {
                                    const int kd = basis.spline(d).order();
                                    if (i == l) {
                                        prod *= (d == i) ? ders[d][2 * kd + off[d]] : ders[d][off[d]];
                                    } else {
                                        prod *= (d == i || d == l) ? ders[d][kd + off[d]]
                                                                   : ders[d][off[d]];
                                    }
                                }
                                hd += Dil * prod;
                            }
                        }
                        r += hd;
                        if (state_dep_D) {
                            double extra = ddD(pt) * v;
                            for (int i = 0; i < n; ++i) {
                                extra += 2.0 * dDrow(pt, i) * grad[static_cast<size_t>(j) * n + i];
                            }
                            r += extra;
                        }
                    }
                    rj[j] = r;
                }

                // Outer-product accumulation: block(k, j) += w * Phi_k * r_j.
                for (int kk = 0; kk < K; ++kk) {
                    const double wv = w * val[kk];
                    if (wv == 0.0) continue;
                    double* brow = block.data() + static_cast<size_t>(kk) * K;
                    for (int j = 0; j < K; ++j) brow[j] += wv * rj[j];
                }
            }

            // Scatter local block to global triplets. Global basis index of
            // local offset o in dim d is (element_index_d + o_d), because the
            // first active basis on span s is exactly s - degree = element id.
            for (int kk = 0; kk < K; ++kk) {
                std::int64_t gk = 0;
                for (int d = 0; d < n; ++d) gk += (emi[d] + table.offs[kk][d]) * basis.stride(d);
                for (int j = 0; j < K; ++j) {
                    const double vv = block[static_cast<size_t>(kk) * K + j];
                    if (vv == 0.0) continue;
                    std::int64_t gj = 0;
                    for (int d = 0; d < n; ++d) gj += (emi[d] + table.offs[j][d]) * basis.stride(d);
                    trip.emplace_back(static_cast<int>(gk), static_cast<int>(gj), vv);
                }
            }
        }
    });

    std::vector<Eigen::Triplet<double>> all;
    std::size_t total = 0;
    for (const auto& t : triplets) total += t.size();
    all.reserve(total);
    for (auto& t : triplets) {
        all.insert(all.end(), t.begin(), t.end());
        t.clear();
        t.shrink_to_fit();
    }
    SpMat M(basis.n_total(), basis.n_total());
    M.setFromTriplets(all.begin(), all.end());  // duplicate entries are summed
    M.makeCompressed();
    return M;
}

// Evaluate p(x) = sum_j a_j Phi_j(x) at each row of X, using only the
// order^dim locally supported basis functions per point.
inline Eigen::VectorXd evaluate_pdf(const TensorBasis& basis,
                                    const Eigen::Ref<const Eigen::VectorXd>& a,
                                    const Eigen::Ref<const RowMatC>& X, int n_threads) {
    const int n = basis.dim();
    const std::int64_t n_pts = X.rows();
    if (a.size() != basis.n_total()) throw std::invalid_argument("evaluate_pdf: bad coefficients");
    if (X.cols() != n) throw std::invalid_argument("evaluate_pdf: X has wrong dimension");
    const auto table = detail::make_local_table(basis);
    Eigen::VectorXd out(n_pts);
    parallel_for(n_pts, n_threads, [&](std::int64_t b, std::int64_t e, int) {
        std::vector<std::vector<double>> ders(n);
        for (int d = 0; d < n; ++d) ders[d].resize(static_cast<size_t>(basis.spline(d).order()));
        int first[kMaxDims];
        for (std::int64_t i = b; i < e; ++i) {
            for (int d = 0; d < n; ++d) {
                const BSpline1D& sp = basis.spline(d);
                const int span = sp.find_span(X(i, d));
                sp.ders_basis_funs(span, X(i, d), 0, ders[d].data());
                first[d] = span - sp.degree();
            }
            double acc = 0.0;
            for (int j = 0; j < table.K; ++j) {
                const auto& off = table.offs[j];
                double v = 1.0;
                std::int64_t gj = 0;
                for (int d = 0; d < n; ++d) {
                    v *= ders[d][off[d]];
                    gj += (first[d] + off[d]) * basis.stride(d);
                }
                acc += a(gj) * v;
            }
            out(i) = acc;
        }
    });
    return out;
}

// Projection right-hand side c_k = <Phi_k, p> ~= sum_q w_q p(x_q) Phi_k(x_q)
// (paper Eq. 12). The initial coefficients follow from solving B a0 = c.
inline Eigen::VectorXd project_rhs(const TensorBasis& basis, const Eigen::Ref<const RowMatC>& X,
                                   const Eigen::Ref<const Eigen::VectorXd>& W,
                                   const Eigen::Ref<const Eigen::VectorXd>& pvals, int n_threads) {
    const int n = basis.dim();
    const std::int64_t n_pts = X.rows();
    if (W.size() != n_pts || pvals.size() != n_pts || X.cols() != n)
        throw std::invalid_argument("project_rhs: inconsistent inputs");
    const auto table = detail::make_local_table(basis);
    const int nt = resolve_threads(n_threads);
    std::vector<Eigen::VectorXd> partial(nt, Eigen::VectorXd::Zero(basis.n_total()));
    parallel_for(n_pts, nt, [&](std::int64_t b, std::int64_t e, int tid) {
        Eigen::VectorXd& c = partial[tid];
        std::vector<std::vector<double>> ders(n);
        for (int d = 0; d < n; ++d) ders[d].resize(static_cast<size_t>(basis.spline(d).order()));
        int first[kMaxDims];
        for (std::int64_t i = b; i < e; ++i) {
            const double wp = W(i) * pvals(i);
            if (wp == 0.0) continue;
            for (int d = 0; d < n; ++d) {
                const BSpline1D& sp = basis.spline(d);
                const int span = sp.find_span(X(i, d));
                sp.ders_basis_funs(span, X(i, d), 0, ders[d].data());
                first[d] = span - sp.degree();
            }
            for (int j = 0; j < table.K; ++j) {
                const auto& off = table.offs[j];
                double v = 1.0;
                std::int64_t gj = 0;
                for (int d = 0; d < n; ++d) {
                    v *= ders[d][off[d]];
                    gj += (first[d] + off[d]) * basis.stride(d);
                }
                c(gj) += wp * v;
            }
        }
    });
    Eigen::VectorXd c = Eigen::VectorXd::Zero(basis.n_total());
    for (const auto& p : partial) c += p;
    return c;
}

// Diagnostic for Halton quadrature: Count how many Halton quadrature points 
// lie inside each element (i.e., each knot span). Elements with zero points are locally
// under-integrated -> increase n_points.
inline Eigen::VectorXi points_per_element(const TensorBasis& basis,
                                          const Eigen::Ref<const RowMatC>& X) {
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(basis.n_elements());
    for (std::int64_t i = 0; i < X.rows(); ++i) {
        counts(basis.element_of(X.row(i).data()))++;
    }
    return counts;
}

}  // namespace fpe
