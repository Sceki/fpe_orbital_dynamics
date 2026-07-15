// Time propagation of the Galerkin coefficients (paper Eqs. 11-13):
//   B da/dt = M a  =>  a(t) = expm(B^{-1} M t) a0.
//
// Two engines are provided:
//  - expm():            dense scaling-and-squaring with a [13/13] Pade
//                       approximant (Higham 2005), for moderate N.
//  - KrylovPropagator:  matrix-free action expm(t B^{-1} M) v via an Arnoldi
//                       (Krylov) projection with adaptive sub-stepping and
//                       the standard augmented-matrix local error estimate
//                       (Saad 1992 / EXPOKIT). B is factorized once with a
//                       sparse LDLT (it is a symmetric positive-definite
//                       Gram matrix), so each operator application is a
//                       sparse mat-vec plus a triangular solve. This scales
//                       to the large, sparse, banded systems produced by
//                       tensor B-spline bases.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>

namespace fpe {

// Dense matrix exponential, Pade-13 with scaling and squaring.
inline Eigen::MatrixXd expm(const Eigen::MatrixXd& A) {
    if (A.rows() != A.cols()) throw std::invalid_argument("expm: matrix must be square");
    const int n = static_cast<int>(A.rows());
    static const double b[14] = {64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
                                 1187353796428800.0,  129060195264000.0,   10559470521600.0,
                                 670442572800.0,      33522128640.0,       1323241920.0,
                                 40840800.0,          960960.0,            16380.0,
                                 182.0,               1.0};
    const double theta13 = 5.371920351148152;
    const double norm1 = A.cwiseAbs().colwise().sum().maxCoeff();
    int s = 0;
    if (norm1 > theta13) {
        s = static_cast<int>(std::ceil(std::log2(norm1 / theta13)));
    }
    const Eigen::MatrixXd As = A / std::pow(2.0, s);
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd A2 = As * As;
    const Eigen::MatrixXd A4 = A2 * A2;
    const Eigen::MatrixXd A6 = A2 * A4;
    const Eigen::MatrixXd U =
        As * (A6 * (b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 +
              b[1] * I);
    const Eigen::MatrixXd V =
        A6 * (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I;
    Eigen::MatrixXd R = (V - U).partialPivLu().solve(V + U);
    for (int i = 0; i < s; ++i) R = R * R;
    return R;
}

class KrylovPropagator {
public:
    using SpMat = Eigen::SparseMatrix<double>;

    KrylovPropagator(const SpMat& B, const SpMat& M, int m = 40, double tol = 1e-10)
        : M_(M), m_(m), tol_(tol) {
        if (B.rows() != B.cols() || M.rows() != M.cols() || B.rows() != M.rows())
            throw std::invalid_argument("KrylovPropagator: B and M must be square, same size");
        if (m_ < 2) throw std::invalid_argument("KrylovPropagator: m must be >= 2");
        if (tol_ <= 0.0) throw std::invalid_argument("KrylovPropagator: tol must be > 0");
        Bsolver_.compute(B);
        if (Bsolver_.info() != Eigen::Success)
            throw std::runtime_error("KrylovPropagator: LDLT factorization of B failed");
        n_ = B.rows();
    }

    Eigen::Index size() const { return n_; }

    // y = B^{-1} M v (the ODE right-hand-side operator).
    Eigen::VectorXd matvec(const Eigen::VectorXd& v) const { return Bsolver_.solve(M_ * v); }

    // w ~= expm(t B^{-1} M) v.
    Eigen::VectorXd apply(const Eigen::VectorXd& v, double t) const {
        if (v.size() != n_) throw std::invalid_argument("KrylovPropagator::apply: bad vector size");
        if (t == 0.0) return v;

        // Small systems: build the dense operator and exponentiate directly.
        const int m = static_cast<int>(std::min<Eigen::Index>(m_, n_ - 1));
        if (n_ <= m_ + 2) {
            Eigen::MatrixXd Ad = Bsolver_.solve(Eigen::MatrixXd(M_));
            return expm(t * Ad) * v;
        }

        Eigen::VectorXd w = v;
        double t_done = 0.0;
        double tau = t;  // trial sub-step
        const double t_total = t;
        int rejections = 0;

        Eigen::MatrixXd V(n_, m + 1);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m + 1, m);

        while (t_done < t_total) {
            const double beta = w.norm();
            if (beta == 0.0) return w;

            // Arnoldi with modified Gram-Schmidt (+ one reorthogonalization).
            H.setZero();
            V.col(0) = w / beta;
            int mj = m;
            bool happy = false;
            for (int j = 0; j < m; ++j) {
                Eigen::VectorXd p = matvec(V.col(j));
                for (int i = 0; i <= j; ++i) {
                    const double hij = V.col(i).dot(p);
                    H(i, j) += hij;
                    p -= hij * V.col(i);
                }
                for (int i = 0; i <= j; ++i) {  // reorthogonalize
                    const double corr = V.col(i).dot(p);
                    H(i, j) += corr;
                    p -= corr * V.col(i);
                }
                const double hnext = p.norm();
                H(j + 1, j) = hnext;
                if (hnext < 1e-14 * (1.0 + H.cwiseAbs().maxCoeff())) {
                    mj = j + 1;
                    happy = true;  // invariant subspace found: projection exact
                    break;
                }
                V.col(j + 1) = p / hnext;
            }

            // Adaptive sub-step with the augmented-matrix error estimate.
            while (true) {
                const double tau_try = std::min(tau, t_total - t_done);
                Eigen::VectorXd y;
                double err_loc = 0.0;
                if (happy) {
                    const Eigen::MatrixXd E = expm(tau_try * H.topLeftCorner(mj, mj));
                    y = beta * E.col(0);
                } else {
                    Eigen::MatrixXd Haug = Eigen::MatrixXd::Zero(mj + 1, mj + 1);
                    Haug.topLeftCorner(mj, mj) = H.topLeftCorner(mj, mj);
                    Haug(mj, mj - 1) = H(mj, mj - 1);
                    const Eigen::MatrixXd E = expm(tau_try * Haug);
                    y = beta * E.col(0).head(mj);
                    err_loc = beta * std::abs(E(mj, 0));
                }
                const double budget = tol_ * std::max(beta, 1.0) * (tau_try / t_total);
                if (happy || err_loc <= budget || tau_try <= 1e-14 * std::abs(t_total)) {
                    w = V.leftCols(mj) * y;
                    t_done += tau_try;
                    if (happy) t_done = t_total;  // exact in the Krylov subspace
                    tau = tau_try * 2.0;          // gently grow the step
                    break;
                }
                tau = tau_try * 0.5;
                if (++rejections > 200)
                    throw std::runtime_error("KrylovPropagator: step size underflow");
            }
        }
        return w;
    }

private:
    SpMat M_;
    Eigen::SimplicialLDLT<SpMat> Bsolver_;
    Eigen::Index n_{0};
    int m_;
    double tol_;
};

}  // namespace fpe
