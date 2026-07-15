// Deterministic drift models f(x) and their divergence div f = sum_i df_i/dx_i,
// which is what the Galerkin-projected Fokker-Planck operator needs
// (paper Eq. 10, after expanding d(f_i Phi_j)/dx_i).
//
// Built-in models evaluate the divergence exactly: analytically when trivial,
// otherwise with forward-mode automatic differentiation (dual numbers),
// mirroring the paper's use of AD (Sec. 3.2). Arbitrary Python dynamics are
// supported from the Python layer by passing precomputed arrays of f and
// div f at the quadrature points to the assembly routine.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include "fpe/dual.hpp"
#include "fpe/threading.hpp"

namespace fpe {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class Dynamics {
public:
    explicit Dynamics(int dim) : dim_(dim) {}
    virtual ~Dynamics() = default;

    int dim() const { return dim_; }

    // Drift f and divergence at a single point.
    virtual void eval(const double* x, double* f, double* divf) const = 0;

    // Threaded batch evaluation at all quadrature points.
    void eval_batch(const Eigen::Ref<const RowMat>& X, Eigen::Ref<RowMat> F,
                    Eigen::Ref<Eigen::VectorXd> divF, int n_threads) const {
        const std::int64_t n_pts = X.rows();
        if (X.cols() != dim_ || F.rows() != n_pts || F.cols() != dim_ || divF.size() != n_pts)
            throw std::invalid_argument("Dynamics::eval_batch: bad shapes");
        parallel_for(n_pts, n_threads, [&](std::int64_t b, std::int64_t e, int) {
            for (std::int64_t i = b; i < e; ++i) {
                eval(X.row(i).data(), F.row(i).data(), &divF(i));
            }
        });
    }

private:
    int dim_;
};

// Damped harmonic oscillator with additive noise on the velocity
// (paper Sec. 5.1, Zorzano et al. 1999):
//   dx = v dt
//   dv = (-K x - gamma v) dt + sqrt(2 sigma) dW
// The diffusion enters through the (constant) diffusion matrix D with
// D_vv = sigma, supplied separately to the assembly.
class DampedOscillator : public Dynamics {
public:
    DampedOscillator(double k, double gamma) : Dynamics(2), k_(k), gamma_(gamma) {}

    void eval(const double* x, double* f, double* divf) const override {
        f[0] = x[1];
        f[1] = -k_ * x[0] - gamma_ * x[1];
        *divf = -gamma_;  // df0/dx + df1/dv = 0 - gamma
    }

    double k() const { return k_; }
    double gamma() const { return gamma_; }

private:
    double k_, gamma_;
};

// Orbit-averaged equinoctial dynamics with in-plane atmospheric drag
// (paper Sec. 5.2, Eqs. 24-27; Di Carlo et al. 2017). State: (a, P1, P2)
// with P1 = e sin(pomega), P2 = e cos(pomega). The out-of-plane acceleration
// is zero, so Q1/Q2 do not enter. The average over the true longitude L,
//   dE/dt = 1/(2 pi) \int_{-pi}^{pi} (B^3 / Phi^2) (dE/dt)|_Gauss dL,
// is computed with a uniform midpoint rule, which converges spectrally for
// this periodic integrand.
//
// Units: consistent with mu (default km^3/s^2 -> a in km, delta = rho Cd A/m
// in 1/km, time in seconds).
template <class T>
inline void equinoctial_drag_rhs(const T& a, const T& P1, const T& P2, double mu, double delta,
                                 int n_quad_L, T* out) {
    using std::sqrt;  // alongside fpe::sqrt(Dual), found by overload resolution
    const double pi = 3.14159265358979323846;
    const T B2 = 1.0 - P1 * P1 - P2 * P2;
    out[0] = T(0.0);
    out[1] = T(0.0);
    out[2] = T(0.0);
    const T Bs = sqrt(B2);
    const T p = a * B2;         // semilatus rectum
    const T h = sqrt(mu * p);   // angular momentum
    const double dL = 2.0 * pi / n_quad_L;
    for (int i = 0; i < n_quad_L; ++i) {
        const double L = -pi + (i + 0.5) * dL;
        const double sL = std::sin(L), cL = std::cos(L);
        const T Phi = 1.0 + P1 * sL + P2 * cL;             // p / r
        const T esf = P2 * sL - P1 * cL;                   // e sin(true anomaly)
        const T v2 = (mu / a) * (2.0 * Phi / B2 - 1.0);    // vis-viva speed^2
        const T Dv = sqrt(1.0 + P1 * P1 + P2 * P2 + 2.0 * (P1 * sL + P2 * cL));
        const T c = -0.5 * delta * v2;                     // drag opposes velocity
        const T ar = c * esf / Dv;                         // radial acceleration
        const T at = c * Phi / Dv;                         // transverse acceleration
        const T w = Bs * B2 / (Phi * Phi);                 // B^3 / Phi^2 averaging factor
        const T r_over_h = p / (h * Phi);
        const T da = (2.0 * a * a / h) * (esf * ar + Phi * at);
        const T dP1 = r_over_h * (-Phi * cL * ar + (P1 + (1.0 + Phi) * sL) * at);
        const T dP2 = r_over_h * (Phi * sL * ar + (P2 + (1.0 + Phi) * cL) * at);
        out[0] += w * da;
        out[1] += w * dP1;
        out[2] += w * dP2;
    }
    const double scale = 1.0 / n_quad_L;  // (1/2pi) * dL sum
    out[0] *= scale;
    out[1] *= scale;
    out[2] *= scale;
}

class EquinoctialAveragedDrag : public Dynamics {
public:
    EquinoctialAveragedDrag(double mu, double delta, int n_quad_L)
        : Dynamics(3), mu_(mu), delta_(delta), n_quad_L_(n_quad_L) {
        if (n_quad_L < 4)
            throw std::invalid_argument("EquinoctialAveragedDrag: n_quad_L must be >= 4");
    }

    void eval(const double* x, double* f, double* divf) const override {
        // Guard against unphysical states (e >= 1) that a generous basis
        // domain could in principle contain; the drift is set to zero there.
        if (1.0 - x[1] * x[1] - x[2] * x[2] <= 1e-12 || x[0] <= 0.0) {
            f[0] = f[1] = f[2] = 0.0;
            *divf = 0.0;
            return;
        }
        const Dual<3> a = Dual<3>::seed(x[0], 0);
        const Dual<3> P1 = Dual<3>::seed(x[1], 1);
        const Dual<3> P2 = Dual<3>::seed(x[2], 2);
        Dual<3> out[3];
        equinoctial_drag_rhs(a, P1, P2, mu_, delta_, n_quad_L_, out);
        double div = 0.0;
        for (int i = 0; i < 3; ++i) {
            f[i] = out[i].v;
            div += out[i].g[i];
        }
        *divf = div;
    }

    // Drift only, without derivatives (for tests / plain propagation).
    void eval_f(const double* x, double* f) const {
        if (1.0 - x[1] * x[1] - x[2] * x[2] <= 1e-12 || x[0] <= 0.0) {
            f[0] = f[1] = f[2] = 0.0;
            return;
        }
        double out[3];
        equinoctial_drag_rhs(x[0], x[1], x[2], mu_, delta_, n_quad_L_, out);
        f[0] = out[0];
        f[1] = out[1];
        f[2] = out[2];
    }

    double mu() const { return mu_; }
    double delta() const { return delta_; }
    int n_quad_L() const { return n_quad_L_; }

private:
    double mu_;
    double delta_;
    int n_quad_L_;
};

}  // namespace fpe
