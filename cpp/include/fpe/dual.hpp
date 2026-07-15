// Forward-mode automatic differentiation via dual numbers.
//
// A Dual<N> carries a value and N tangent (derivative) components. Seeding
// the i-th input with tangent e_i and evaluating a templated function yields
// the exact Jacobian column-by-column in a single pass, mirroring the
// forward-mode AD used in the paper (Sec. 3.2) to differentiate the dynamics.
#pragma once

#include <array>
#include <cmath>

namespace fpe {

template <int N>
struct Dual {
    double v{0.0};
    std::array<double, N> g{};

    Dual() = default;
    Dual(double value) : v(value) { g.fill(0.0); }  // NOLINT(google-explicit-constructor)

    static Dual seed(double value, int direction) {
        Dual d(value);
        d.g[direction] = 1.0;
        return d;
    }

    Dual& operator+=(const Dual& o) {
        v += o.v;
        for (int i = 0; i < N; ++i) g[i] += o.g[i];
        return *this;
    }
    Dual& operator-=(const Dual& o) {
        v -= o.v;
        for (int i = 0; i < N; ++i) g[i] -= o.g[i];
        return *this;
    }
    Dual& operator*=(double s) {
        v *= s;
        for (int i = 0; i < N; ++i) g[i] *= s;
        return *this;
    }

    friend Dual operator+(Dual a, const Dual& b) { return a += b; }
    friend Dual operator-(Dual a, const Dual& b) { return a -= b; }
    friend Dual operator-(const Dual& a) {
        Dual r;
        r.v = -a.v;
        for (int i = 0; i < N; ++i) r.g[i] = -a.g[i];
        return r;
    }

    friend Dual operator*(const Dual& a, const Dual& b) {
        Dual r;
        r.v = a.v * b.v;
        for (int i = 0; i < N; ++i) r.g[i] = a.g[i] * b.v + a.v * b.g[i];
        return r;
    }

    friend Dual operator/(const Dual& a, const Dual& b) {
        Dual r;
        const double inv = 1.0 / b.v;
        r.v = a.v * inv;
        for (int i = 0; i < N; ++i) r.g[i] = (a.g[i] - r.v * b.g[i]) * inv;
        return r;
    }

    // Mixed double/Dual arithmetic.
    friend Dual operator+(const Dual& a, double b) { Dual r(a); r.v += b; return r; }
    friend Dual operator+(double a, const Dual& b) { return b + a; }
    friend Dual operator-(const Dual& a, double b) { Dual r(a); r.v -= b; return r; }
    friend Dual operator-(double a, const Dual& b) { return -b + a; }
    friend Dual operator*(Dual a, double b) { return a *= b; }
    friend Dual operator*(double a, Dual b) { return b *= a; }
    friend Dual operator/(const Dual& a, double b) { Dual r(a); r *= (1.0 / b); return r; }
    friend Dual operator/(double a, const Dual& b) { return Dual(a) / b; }
};

template <int N>
inline Dual<N> sqrt(const Dual<N>& a) {
    Dual<N> r;
    r.v = std::sqrt(a.v);
    const double d = 0.5 / r.v;
    for (int i = 0; i < N; ++i) r.g[i] = a.g[i] * d;
    return r;
}

template <int N>
inline Dual<N> sin(const Dual<N>& a) {
    Dual<N> r;
    r.v = std::sin(a.v);
    const double c = std::cos(a.v);
    for (int i = 0; i < N; ++i) r.g[i] = a.g[i] * c;
    return r;
}

template <int N>
inline Dual<N> cos(const Dual<N>& a) {
    Dual<N> r;
    r.v = std::cos(a.v);
    const double s = -std::sin(a.v);
    for (int i = 0; i < N; ++i) r.g[i] = a.g[i] * s;
    return r;
}

}  // namespace fpe
