#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "minimax_cpppy/minimax.hpp"
#include "minimax_cpppy/minimax_lp.hpp"
#include "minimax_cpppy/biasing.hpp"
#include <vector>

namespace py = pybind11;

static std::vector<double> arr_to_vec(py::object obj) {
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(obj);
    return std::vector<double>(arr.data(), arr.data() + arr.size());
}

PYBIND11_MODULE(_minimax_core, m) {
    m.doc() = "minimax_cpppy C++ bridge — use the minimax_cpppy Python package, not this module directly";

    m.def("laplace_minimax",
        [](int nlap, double ymin, double ymax,
           py::object init_expon, py::object init_weight,
           int verbose) {
            if (nlap < 1 || nlap > 30)
                throw py::value_error("laplace_minimax: nlap must be in [1, 30]");
            if (ymin <= 0.0 || ymax <= ymin)
                throw py::value_error("laplace_minimax: need 0 < ymin < ymax");
            bool has_e = !init_expon.is_none();
            bool has_w = !init_weight.is_none();
            if (has_e != has_w) {
                throw py::value_error(
                    "laplace_minimax: init_expon and init_weight must both be "
                    "provided or both be None");
            }
            minimax_cpppy::MinimaxResult r;
            if (has_e) {
                r = minimax_cpppy::laplaceMinimax(
                    nlap, ymin, ymax,
                    arr_to_vec(init_expon),
                    arr_to_vec(init_weight),
                    verbose);
            } else {
                r = minimax_cpppy::laplaceMinimax(nlap, ymin, ymax, verbose);
            }
            py::array_t<double> expon(static_cast<py::ssize_t>(r.expon.size()),
                                      r.expon.data());
            py::array_t<double> weight(static_cast<py::ssize_t>(r.weight.size()),
                                       r.weight.data());
            return py::make_tuple(expon, weight, r.errmax);
        },
        py::arg("nlap"), py::arg("ymin"), py::arg("ymax"),
        py::kw_only(),
        py::arg("init_expon") = py::none(),
        py::arg("init_weight") = py::none(),
        py::arg("verbose") = 3,
        R"doc(
Compute MINIMAX-optimal Laplace exponents and weights for 1/x on [ymin, ymax].

Parameters
----------
nlap : int
    Number of quadrature points. Must satisfy 1 <= nlap <= 53.
ymin : float
    Lower bound of approximation interval. Must be > 0.
ymax : float
    Upper bound of approximation interval. Must be > ymin.
init_expon : ndarray, shape (nlap,), float64, optional
    Initial exponents in [ymin, ymax] domain. Must be provided together with
    init_weight. Bypasses the pre-tabulated lookup.
init_weight : ndarray, shape (nlap,), float64, optional
    Initial weights in [ymin, ymax] domain. Must be provided together with
    init_expon.
verbose : int, optional
    Verbosity level (default 3).
    0 = silent, 1 = errors only, 2 = +warnings, 3 = +info per Remez/NR iter,
    4 = +debug per Newton step. Output goes to stderr.

Returns
-------
exponents : ndarray, shape (nlap,), float64
weights : ndarray, shape (nlap,), float64
errmax : float
    Maximum absolute approximation error on [ymin, ymax].

Raises
------
ValueError
    If nlap is not in [1, 30], if ymin/ymax are not 0 < ymin < ymax, if exactly
    one of init_expon/init_weight is provided, or sizes do not equal nlap.
RuntimeError
    If no tabulated initial data exists and no initial guess is provided.
)doc");

    m.def("laplace_lp",
        [](int nlap, double ymin, double ymax, double norm_p, int verbose) {
            if (nlap < 1 || nlap > 30)
                throw py::value_error("laplace_lp: nlap must be in [1, 30]");
            if (ymin <= 0.0 || ymax <= ymin)
                throw py::value_error("laplace_lp: need 0 < ymin < ymax");
            if (norm_p < 1.0)
                throw py::value_error("laplace_lp: norm_p must be >= 1");
            auto r = minimax_cpppy::laplaceLp(nlap, ymin, ymax, norm_p, verbose);
            py::array_t<double> expon (static_cast<py::ssize_t>(r.expon.size()),  r.expon.data());
            py::array_t<double> weight(static_cast<py::ssize_t>(r.weight.size()), r.weight.data());
            return py::make_tuple(expon, weight, r.errmax);
        },
        py::arg("nlap"), py::arg("ymin"), py::arg("ymax"), py::arg("norm_p"),
        py::kw_only(), py::arg("verbose") = 3,
        R"doc(
L_p-norm-optimal Laplace quadrature for 1/x on [ymin, ymax].

Minimises the L_p norm (norm_p >= 1) of the quadrature error, warm-started from
the unbiased minimax solution. Even integer norm_p uses |eta|^p == eta^p; odd
integer and non-integer norm_p minimise the absolute-value loss ∫|eta|^p;
norm_p=1 uses a smoothed L1 continuation. Larger norm_p approaches minimax (L_inf).

Parameters
----------
nlap : int       Number of quadrature points (1 <= nlap <= 30).
ymin : float     Lower bound (> 0).
ymax : float     Upper bound (> ymin).
norm_p : float   Loss order; real >= 1 (norm_p=1 is the L1 / mean-abs problem).
                 Low non-integer norm_p (< ~1.25) use a continuation-in-norm_p
                 fallback (slower); orders near 1 may still fail to converge.
verbose : int    Verbosity (default 3).

Returns
-------
exponents : ndarray, shape (nlap,)
weights : ndarray, shape (nlap,)
lp_norm : float   The L_p norm (1/ymin)*(integral eta^p dt)^(1/p) of the result.

Raises
------
ValueError    Bad nlap/range, or norm_p < 1.
RuntimeError  If Newton fails to converge.
)doc");

    py::class_<minimax_cpppy::DenominatorDensity>(m, "DenominatorDensity",
        "Orbital energy denominator density via FFT + quintic Hermite spline.\n"
        "Construct from numpy arrays of occupied and virtual orbital energies.\n"
        "C (>= 0): constant density offset added in units of the uniform density\n"
        "(1/t_max); C=0 is fully biased, C->inf recovers unbiased minimax.\n")
        .def(py::init([](py::object occ_obj, py::object virt_obj,
                         double bandwidth, int n_fft, int n_t,
                         double floor_frac,
                         double floor_frac_max, double C, int n_exc) {
            auto occ  = py::array_t<double, py::array::c_style | py::array::forcecast>(occ_obj);
            auto virt = py::array_t<double, py::array::c_style | py::array::forcecast>(virt_obj);
            return minimax_cpppy::DenominatorDensity(
                occ.data(),  static_cast<int>(occ.size()),
                virt.data(), static_cast<int>(virt.size()),
                bandwidth, n_fft, n_t, floor_frac,
                floor_frac_max, C, n_exc);
        }),
        py::arg("occ"), py::arg("virt"),
        py::arg("bandwidth"),
        py::arg("n_fft")          = 4096,
        py::arg("n_t")            = 512,
        py::arg("floor_frac")     = 1e-3,
        py::arg("floor_frac_max") = -1.0,
        py::arg("C")              = 0.0,
        py::arg("n_exc")          = 2)
        .def_property_readonly("ratio",     &minimax_cpppy::DenominatorDensity::ratio)
        .def_property_readonly("delta_min", &minimax_cpppy::DenominatorDensity::deltaMin)
        .def_property_readonly("delta_max", &minimax_cpppy::DenominatorDensity::deltaMax);

    m.def("biased_laplace",
        [](int nlap, double ymin, double ymax,
           py::object occ_obj, py::object virt_obj,
           double bandwidth, int n_fft, int n_t,
           double floor_frac, double floor_frac_max, double C, int n_exc,
           int verbose) {
            if (nlap < 1 || nlap > 30)
                throw py::value_error("biased_laplace: nlap must be in [1, 30]");
            if (ymin <= 0.0 || ymax <= ymin)
                throw py::value_error("biased_laplace: need 0 < ymin < ymax");
            auto occ  = py::array_t<double, py::array::c_style | py::array::forcecast>(occ_obj);
            auto virt = py::array_t<double, py::array::c_style | py::array::forcecast>(virt_obj);
            auto r = minimax_cpppy::biasedLaplace(
                nlap, ymin, ymax,
                occ.data(),  static_cast<int>(occ.size()),
                virt.data(), static_cast<int>(virt.size()),
                bandwidth, n_fft, n_t, floor_frac, floor_frac_max, C, n_exc,
                verbose);
            py::array_t<double> expon (static_cast<py::ssize_t>(r.expon.size()),  r.expon.data());
            py::array_t<double> weight(static_cast<py::ssize_t>(r.weight.size()), r.weight.data());
            return py::make_tuple(expon, weight, r.errmax);
        },
        py::arg("nlap"), py::arg("ymin"), py::arg("ymax"),
        py::arg("occ"), py::arg("virt"),
        py::arg("bandwidth"),
        py::kw_only(),
        py::arg("n_fft")          = 4096,
        py::arg("n_t")            = 512,
        py::arg("floor_frac")     = 1e-3,
        py::arg("floor_frac_max") = -1.0,
        py::arg("C")              = 0.0,
        py::arg("n_exc")          = 2,
        py::arg("verbose")        = 3,
        R"doc(
Density-uncorrelated bias-correction Laplace quadrature for 1/x on [ymin, ymax].

Drives the density-weighted net signed bias r = integral of eta(x)*p(x) over
[ymin, ymax] toward zero, warm-started from the unbiased minimax solution.
Builds its own denominator density internally from occ/virt/bandwidth (same
inputs as DenominatorDensity); ymin/ymax must match that density's
delta_min/delta_max within 1 ppm.

Parameters
----------
nlap : int          Number of quadrature points (1 <= nlap <= 30).
ymin : float         Lower bound (= delta_min for the implied density).
ymax : float         Upper bound (= delta_max for the implied density).
occ : ndarray        Occupied orbital energies [Ha].
virt : ndarray        Virtual orbital energies [Ha].
bandwidth : float     KDE bandwidth (see DenominatorDensity).
n_fft, n_t, floor_frac, floor_frac_max, C, n_exc : see DenominatorDensity.
verbose : int, optional   Verbosity level (default 3).

Returns
-------
exponents : ndarray, shape (nlap,)
weights : ndarray, shape (nlap,)
errmax : float
    Maximum absolute normalised-domain approximation error (same convention as
    laplace_minimax) -- not the achieved density-weighted bias, which is a
    separate, smaller quantity by design.

Raises
------
ValueError
    If nlap is not in [1, 30], if ymin/ymax are not 0 < ymin < ymax, or if the
    density implied by occ/virt/bandwidth does not match ymax/ymin.
)doc");
}
