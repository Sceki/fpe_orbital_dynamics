// pybind11 bindings for the fpe C++ core.
//
// The Python layer (python/fpe) provides the user-facing API; this module
// exposes the performance-critical kernels: B-spline bases, quadrature
// generation, sparse Galerkin assembly, pdf evaluation/projection, built-in
// dynamics with exact divergences, and matrix-exponential propagators.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "fpe/assembly.hpp"
#include "fpe/bspline.hpp"
#include "fpe/dynamics.hpp"
#include "fpe/expm.hpp"
#include "fpe/halton.hpp"
#include "fpe/tensor_basis.hpp"

namespace py = pybind11;
using fpe::RowMatC;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of fpe: Galerkin projection of the Fokker-Planck equation";

    py::class_<fpe::BSpline1D>(m, "BSpline1D",
                               "Clamped uniform B-spline basis on [lo, hi] (paper Eqs. 14-17)")
        .def(py::init<double, double, int, int>(), py::arg("lo"), py::arg("hi"),
             py::arg("n_basis"), py::arg("order"))
        .def_property_readonly("lo", &fpe::BSpline1D::lo)
        .def_property_readonly("hi", &fpe::BSpline1D::hi)
        .def_property_readonly("n_basis", &fpe::BSpline1D::n_basis)
        .def_property_readonly("order", &fpe::BSpline1D::order)
        .def_property_readonly("n_spans", &fpe::BSpline1D::n_spans)
        .def_property_readonly("knots",
                               [](const fpe::BSpline1D& s) {
                                   return Eigen::Map<const Eigen::VectorXd>(
                                       s.knots().data(), static_cast<Eigen::Index>(s.knots().size()))
                                       .eval();
                               })
        .def(
            "basis_matrix",
            [](const fpe::BSpline1D& s, const Eigen::VectorXd& x, int der) {
                std::vector<double> xv(x.data(), x.data() + x.size());
                const auto flat = s.basis_matrix(xv, der);
                RowMatC out(x.size(), s.n_basis());
                for (Eigen::Index i = 0; i < x.size(); ++i)
                    for (int j = 0; j < s.n_basis(); ++j)
                        out(i, j) = flat[static_cast<size_t>(i) * s.n_basis() + j];
                return out;
            },
            py::arg("x"), py::arg("der") = 0,
            "Dense matrix of the der-th derivative of every basis function at each x")
        .def("gram",
             [](const fpe::BSpline1D& s) {
                 const auto flat = s.gram();
                 RowMatC out(s.n_basis(), s.n_basis());
                 for (int i = 0; i < s.n_basis(); ++i)
                     for (int j = 0; j < s.n_basis(); ++j)
                         out(i, j) = flat[static_cast<size_t>(i) * s.n_basis() + j];
                 return out;
             },
             "Exact Gram matrix <phi_i, phi_j> (Gauss-Legendre per knot span)")
        .def("integrals",
             [](const fpe::BSpline1D& s) {
                 std::vector<double> I0, I1, I2;
                 s.integrals(I0, I1, I2);
                 auto to_vec = [](const std::vector<double>& v) {
                     return Eigen::Map<const Eigen::VectorXd>(v.data(),
                                                              static_cast<Eigen::Index>(v.size()))
                         .eval();
                 };
                 return py::make_tuple(to_vec(I0), to_vec(I1), to_vec(I2));
             },
             "Exact (I0, I1, I2) with I0=int phi, I1=int x phi, I2=int x^2 phi");

    py::class_<fpe::TensorBasis>(m, "TensorBasis",
                                 "Tensor-product multivariate B-spline basis (paper Eqs. 4-5)")
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<int>&, const std::vector<int>&>(),
             py::arg("lo"), py::arg("hi"), py::arg("n_basis"), py::arg("order"))
        .def_property_readonly("dim", &fpe::TensorBasis::dim)
        .def_property_readonly("n_total", &fpe::TensorBasis::n_total)
        .def_property_readonly("shape", &fpe::TensorBasis::shape)
        .def_property_readonly("element_shape", &fpe::TensorBasis::element_shape)
        .def_property_readonly("n_elements", &fpe::TensorBasis::n_elements)
        .def("spline", &fpe::TensorBasis::spline, py::arg("d"),
             py::return_value_policy::copy, "The 1D basis of dimension d")
        .def(
            "element_quadrature",
            [](const fpe::TensorBasis& b, int q) {
                const std::int64_t n = b.element_quadrature_size(q);
                RowMatC X(n, b.dim());
                Eigen::VectorXd W(n);
                b.element_quadrature(q, X, W);
                return py::make_tuple(X, W);
            },
            py::arg("q"),
            "Tensor Gauss-Legendre quadrature with q points per knot span per dimension");

    m.def("halton", &fpe::halton, py::arg("n"), py::arg("dim"), py::arg("skip") = 1,
          "Halton low-discrepancy points in [0, 1)^dim");

    py::class_<fpe::Dynamics>(m, "Dynamics")
        .def_property_readonly("dim", &fpe::Dynamics::dim)
        .def(
            "eval",
            [](const fpe::Dynamics& d, const Eigen::VectorXd& x) {
                if (x.size() != d.dim()) throw std::invalid_argument("eval: bad state size");
                Eigen::VectorXd f(d.dim());
                double divf = 0.0;
                d.eval(x.data(), f.data(), &divf);
                return py::make_tuple(f, divf);
            },
            py::arg("x"), "Drift f(x) and its divergence at a single state")
        .def(
            "eval_batch",
            [](const fpe::Dynamics& d, const Eigen::Ref<const RowMatC>& X, int n_threads) {
                RowMatC F(X.rows(), d.dim());
                Eigen::VectorXd divF(X.rows());
                {
                    py::gil_scoped_release release;
                    d.eval_batch(X, F, divF, n_threads);
                }
                return py::make_tuple(F, divF);
            },
            py::arg("X"), py::arg("n_threads") = 0,
            "Drift and divergence at every row of X (threaded)");

    py::class_<fpe::DampedOscillator, fpe::Dynamics>(
        m, "DampedOscillator",
        "dx = v dt; dv = (-k x - gamma v) dt + sqrt(2 sigma) dW (paper Sec. 5.1)")
        .def(py::init<double, double>(), py::arg("k"), py::arg("gamma"))
        .def_property_readonly("k", &fpe::DampedOscillator::k)
        .def_property_readonly("gamma", &fpe::DampedOscillator::gamma);

    py::class_<fpe::EquinoctialAveragedDrag, fpe::Dynamics>(
        m, "EquinoctialAveragedDrag",
        "Orbit-averaged (a, P1, P2) dynamics with in-plane drag (paper Sec. 5.2, Eqs. 24-27). "
        "delta = rho Cd A/m in units consistent with mu; divergence via forward-mode AD.")
        .def(py::init<double, double, int>(), py::arg("mu"), py::arg("delta"),
             py::arg("n_quad_L") = 64)
        .def_property_readonly("mu", &fpe::EquinoctialAveragedDrag::mu)
        .def_property_readonly("delta", &fpe::EquinoctialAveragedDrag::delta)
        .def_property_readonly("n_quad_L", &fpe::EquinoctialAveragedDrag::n_quad_L)
        .def(
            "eval_f",
            [](const fpe::EquinoctialAveragedDrag& d, const Eigen::VectorXd& x) {
                if (x.size() != 3) throw std::invalid_argument("eval_f: state must have size 3");
                Eigen::VectorXd f(3);
                d.eval_f(x.data(), f.data());
                return f;
            },
            py::arg("x"), "Drift only (no derivatives)");

    m.def(
        "assemble_M",
        [](const fpe::TensorBasis& basis, const Eigen::Ref<const RowMatC>& X,
           const Eigen::Ref<const Eigen::VectorXd>& W, const Eigen::Ref<const RowMatC>& F,
           const Eigen::Ref<const Eigen::VectorXd>& divF, const Eigen::MatrixXd& Dconst,
           std::optional<RowMatC> Dpt, std::optional<RowMatC> dDrow,
           std::optional<Eigen::VectorXd> ddD, int n_threads) {
            RowMatC Dpt_ = Dpt.value_or(RowMatC());
            RowMatC dDrow_ = dDrow.value_or(RowMatC());
            Eigen::VectorXd ddD_ = ddD.value_or(Eigen::VectorXd());
            py::gil_scoped_release release;
            return fpe::assemble_M(basis, X, W, F, divF, Dconst, Dpt_, dDrow_, ddD_, n_threads);
        },
        py::arg("basis"), py::arg("X"), py::arg("W"), py::arg("F"), py::arg("divF"),
        py::arg("Dconst"), py::arg("Dpt") = py::none(), py::arg("dDrow") = py::none(),
        py::arg("ddD") = py::none(), py::arg("n_threads") = 0,
        "Sparse Galerkin drift+diffusion matrix M (paper Eq. 10) from quadrature data");

    m.def(
        "evaluate_pdf",
        [](const fpe::TensorBasis& basis, const Eigen::Ref<const Eigen::VectorXd>& a,
           const Eigen::Ref<const RowMatC>& X, int n_threads) {
            py::gil_scoped_release release;
            return fpe::evaluate_pdf(basis, a, X, n_threads);
        },
        py::arg("basis"), py::arg("a"), py::arg("X"), py::arg("n_threads") = 0,
        "p(x) = sum_j a_j Phi_j(x) at every row of X (threaded, local support only)");

    m.def(
        "project_rhs",
        [](const fpe::TensorBasis& basis, const Eigen::Ref<const RowMatC>& X,
           const Eigen::Ref<const Eigen::VectorXd>& W,
           const Eigen::Ref<const Eigen::VectorXd>& pvals, int n_threads) {
            py::gil_scoped_release release;
            return fpe::project_rhs(basis, X, W, pvals, n_threads);
        },
        py::arg("basis"), py::arg("X"), py::arg("W"), py::arg("pvals"), py::arg("n_threads") = 0,
        "Projection right-hand side c_k = <Phi_k, p> (paper Eq. 12)");

    m.def(
        "points_per_element",
        [](const fpe::TensorBasis& basis, const Eigen::Ref<const RowMatC>& X) {
            return fpe::points_per_element(basis, X);
        },
        py::arg("basis"), py::arg("X"),
        "Quadrature-coverage diagnostic: points falling in each knot-span element");

    m.def(
        "expm",
        [](const Eigen::MatrixXd& A) {
            py::gil_scoped_release release;
            return fpe::expm(A);
        },
        py::arg("A"), "Dense matrix exponential (Pade-13 scaling and squaring)");

    py::class_<fpe::KrylovPropagator>(
        m, "KrylovPropagator",
        "Matrix-free expm(t B^{-1} M) v via Arnoldi projection with adaptive sub-stepping")
        .def(py::init<const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&, int,
                      double>(),
             py::arg("B"), py::arg("M"), py::arg("m") = 40, py::arg("tol") = 1e-10)
        .def_property_readonly("size", &fpe::KrylovPropagator::size)
        .def(
            "apply",
            [](const fpe::KrylovPropagator& p, const Eigen::VectorXd& v, double t) {
                py::gil_scoped_release release;
                return p.apply(v, t);
            },
            py::arg("v"), py::arg("t"), "w = expm(t B^{-1} M) v")
        .def(
            "matvec",
            [](const fpe::KrylovPropagator& p, const Eigen::VectorXd& v) {
                py::gil_scoped_release release;
                return p.matvec(v);
            },
            py::arg("v"), "y = B^{-1} M v");
}
