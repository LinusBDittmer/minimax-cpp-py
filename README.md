# minimax-cpppy

![Build](https://github.com/LinusBDittmer/minimax-cpp-py/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/minimax-cpppy.svg)](https://pypi.org/project/minimax-cpppy/)
[![License](https://img.shields.io/github/license/LinusBDittmer/minimax-cpp-py)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Optimised Laplace quadrature for quantum chemistry.  Given an interval
$`[y_\min, y_\max]`$ and a number of quadrature points `nlap`, the library returns
exponents $`\{a_k\}`$ and weights $`\{w_k\}`$ such that

```math
f(x) = \sum_k w_k \cdot \exp(-a_k \cdot x) \approx \frac{1}{x}, \quad x \in [y_\min, y_\max]
```

with the smallest possible error according to a specified error criterion.

Three quadrature variants are provided:

- **`laplace_minimax`** — the uniform-error minimax ($`L_\infty`$) solution.
- **`laplace_lp`** — minimises the $`L_p`$ norm ($`\text{norm\_p} \geq 1`$) of the error
  instead of $`L_\infty`$; larger `norm_p` approaches the minimax solution.
- **`biased_laplace`** — minimises the square of the leading order orbital
  energy denominator density weighted quadrature error. 
---

## Background

### The Laplace transform trick

In second-order Møller–Plesset perturbation theory (MP2) and coupled-cluster
methods (CC2), the correlation energy involves a four-index sum over orbital energy
denominators.  For closed-shell RHF the MP2 correlation energy is:

```math
E_{\text{MP2}} = \sum_{ijab} \frac{(ia|jb)[2(ia|jb) - (ib|ja)]}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
= -\sum_{ijab} \frac{(ia|jb)[2(ia|jb) - (ib|ja)]}{\Delta^{ab}_{ij}}
```

where $`\Delta^{ab}_{ij} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0`$ 
and indices $`i,j`$ run over occupied and $`a,b`$ over virtual orbitals. The
denominator couples the occupied pair $`(i,j)`$ and the virtual pair $`(a,b)`$,
preventing any factorisation and forcing an $`O(N^5)`$ scaling with system size.

In **SOS-MP2** (Scaled Opposite Spin MP2) the exchange term $`(ib|ja)`$ is
dropped, leaving only the Coulomb-like $`(ia|jb)^2`$ contribution (scaled by an
empirical constant $`c_\text{OS} \approx 1.3`$):

```math
E_{\text{SOS-MP2}} = -c_\text{OS} \cdot \sum_{i,j,a,b} \frac{(ia|jb)^2}{\Delta^{ab}_{ij}}
```

With density fitting (DF/RI) the four-index integrals factorise into
three-index quantities.
Applying the Laplace transform (SOS-DF-LT-MP2) then achieves $`O(N^4)`$ per
Laplace point. Since the number of Laplace points is size-intensive,
SOS-MP2 scales as $`O(N^4)`$ in its entirety.

The key identity is the Laplace representation of $`1/x`$:

```math
\frac{1}{\Delta^{ab}_{ij}} = \int_0^\infty \exp(-t \cdot D_{ia}) \cdot \exp(-t \cdot D_{jb}) \, dt
```

Where $`D_{ia} = \varepsilon_a - \varepsilon_i`$. Replacing the integral by a quadrature with `nlap` nodes $`\{t_k\}`$ and weights
$`\{w_k\}`$:

```math
\frac{1}{\Delta^{ab}_{ij}} \approx \sum_k w_k \cdot \exp(-t_k \cdot D_{ia}) \cdot \exp(-t_k \cdot D_{jb})
```

The right-hand side *factorises* over the $`(i,a)`$ and $`(j,b)`$ pairs. Combined
with density fitting (DF / resolution-of-identity), uces the leading
scaling from $`O(N^5)`$ to $`O(N^4)`$ per Laplace point.

### Minimax optimality

The approximation $`\frac{1}{x} \approx \sum_k w_k \exp(-a_k x)`$ on $`[y_\min, y_\max]`$ is
minimax-optimal when the error function

```math
e(x) = \frac{1}{x} - \sum_k w_k \cdot \exp(-a_k \cdot x)
```

equioscillates at exactly $`2 \cdot \texttt{nlap} + 1`$ points. According to the Chebyshev alternation theorem,
this criterion ensures the best possible approximation in the $`L^\infty`$ norm for a given `nlap`.

### $`L_p`$ optimality

The minimax criterion controls the *worst* error on $`[y_\min, y_\max]`$.  If the
quantity of interest is an aggregate over many denominators rather than a single
worst case, the $`L_p`$ norm

```math
\|e\|_p = \left( \int_{y_\min}^{y_\max} |e(x)|^p \, dx \right)^{1/p}
```

can be a more natural target: it trades a slightly higher peak error for a smaller
error over the bulk of the interval.  $`p \to \infty`$ recovers the minimax
solution, $`p = 1`$ minimises the mean absolute error.

There is no alternation theorem here — $`\|e\|_p`$ is just a smooth (for even $`p`$)
function of the $`2\,\texttt{nlap}`$ parameters, so `laplace_lp` minimises it directly
by damped Newton in $`z = (\ln a_k, w_k)`$ (the log reparametrisation keeps the
exponents positive), warm-started from the $`L_\infty`$ minimax solution.  What
changes with `norm_p` is how the integral is evaluated:

- **$`p = 2`$** — closed form via the exponential integral $`E_1`$; no quadrature.
- **even integer $`p`$** — $`|e|^p = e^p`$ is smooth, so one fixed quadrature rule suffices.
- **odd integer or non-integer $`p > 1`$** — $`|e|^p`$ has kinks at the zeros of $`e`$.  The
  zeros are located explicitly and the quadrature panels are split there; after each
  Newton solve the zeros are re-found and the rule rebuilt, until they stop moving.
- **$`p = 1`$** — piecewise-analytic integral between the zeros, with rank-1 Hessian
  corrections at the kinks.

The optimisation, and the norm reported back as `lp_norm`, are carried out on the
normalised interval $`[1, R]`$, $`R = y_\max/y_\min`$.

For $`p`$ close to 1 the $`|e|^{p-2}`$ Hessian is near-singular at the zeros and the
direct solve can stall; the fallback is a continuation on the smoothed objective
$`\int (e^2 + \varepsilon^2)^{p/2}`$ with $`\varepsilon`$ annealed downwards.

### Density-biased optimisation

Minimax and $`L_p`$ both treat every $`x \in [y_\min, y_\max]`$ as equally important.  In
an actual calculation the denominators $`\Delta^{ab}_{ij}`$ are far from uniformly
distributed — they pile up in a narrow region and are sparse elsewhere — and the
error that survives in the correlation energy is the *signed*, density-weighted
one, in which errors of opposite sign cancel:

```math
r(\theta) = \int_0^{\ln R} e(t;\theta)\, \rho(t) \, dt , \qquad t = \ln x
```

where $`\rho`$ is the density of pairwise denominators (a Gaussian KDE over
$`\Delta^{ab}_{ij}`$, evaluated by FFT and stored as a spline: `DenominatorDensity`).
`biased_laplace` minimises $`r(\theta)^2`$, i.e. drives the leading-order bias of the
Laplace approximation to zero, rather than the pointwise error itself.  This
generally *increases* $`\max_x |e(x)|`$ while reducing the actual energy error.

The solve is Newton in the same $`z = (\ln a_k, w_k)`$ parametrisation, warm-started
from the unbiased minimax solution, in two phases:

1. **Closed-form phase** — $`\rho`$ is replaced by a moment-generating-function model
   built directly from the orbital energies, so $`r`$ and its derivatives are
   available analytically at $`O(\texttt{nlap} \cdot (n_\text{occ} + n_\text{virt}))`$ per
   iteration, with no quadrature and no FFT.  Cheap approach to a near-root.
2. **Exact phase** — $`r`$ is evaluated by composite Gauss–Legendre quadrature
   against the real spline density in double-double precision, polishing the
   phase-1 root.  The phase-1 point is only accepted as the seed if it actually
   lowers the phase-2 objective; if phase-2 Newton diverges, the unbiased minimax
   result is returned instead.

The pedestal parameter `C` interpolates between the two regimes: `C = 0` is fully
biased, and `C → ∞` flattens $`\rho`$ back to a uniform density, recovering the
unbiased solution.

### Algorithm

The implementation is a C++ rewrite of the original Minimax code by Helmich-Paris and Visscher.
`laplace_minimax` uses the **Remez exchange algorithm** with:

- **Newton–Maehly extremum search** — finds the $`2 \cdot \texttt{nlap} - 1`$ interior
  extrema of the current error curve (which, with the two fixed interval endpoints,
  give the $`2 \cdot \texttt{nlap} + 1`$ equioscillation points) via Newton's method with
  Maehly deflation to separate roots.
- **Newton–Raphson equioscillation solver** — updates the exponents, weights,
  and error amplitude to enforce the equioscillation conditions at the located
  extrema, with Armijo backtracking line search.
- **128-bit double-double arithmetic** — all internal calculations use a
  software-emulated `DD` type (two `double` values, ~32 significant decimal
  digits) to avoid numerical cancellation during the Remez iterations.

#### Initialisation

The Remez loop is only locally convergent, so the starting guess matters.  It
comes from pre-tabulated solutions covering $`\texttt{nlap} \in \{1, \ldots, 30\}`$
over a grid of interval ratios $`R = y_\max/y_\min`$.  The grid is two-tier and
non-uniform (rows clustered around the cusp where the solution changes fastest),
with per-`nlap` row counts stored in `EXT_COUNTS`.  A lookup binary-searches
$`\log_{10} R`$ and applies an order-9 local Lagrange interpolation to the stored
exponents, weights, error amplitude, and equioscillation extrema.

The interpolated entry is verified rather than trusted: the actual error
$`\max_x |e(x)|`$ of the interpolated solution is sampled at $`2\,\texttt{nlap}+1`$
log-spaced points, and the solver dispatches on that sampled error:

| Sampled error | Action |
|---------------|--------|
| $`< 10^{-12}`$ | The table already holds a machine-precision solution — return it directly.  Running Remez would fail, since Maehly cannot locate extrema of an already-flat error curve. |
| $`\leq 0.5`$, extrema inside $`(1, R)`$ and largest extremum $`\geq \sqrt{R}`$ | Warm-start Remez from the interpolated exponents/weights, passing the interpolated extrema as the hint for the first Newton–Maehly call. |
| $`\leq 0.5`$, extrema out of range or clustered near $`x = 1`$ | Extrema are stale (Lagrange overshoot or a copied neighbour row) — discard them, try the log-space LS fallback, and return it directly if it already reaches $`< 1\%`$ error (over-parameterised case: small $`R`$, large `nlap`); otherwise run Remez cold from the table values. |
| $`> 0.5`$ | Table row is a stale copy for this ratio — same LS dispatch, but Remez is seeded from the log-space fallback (accepted at a loose $`10.0`$ threshold) instead of the table. |

The log-space LS fallback places exponents on a log-uniform interior grid,
$`a_k = R^{-(k+1)/(\texttt{nlap}+1)}`$, and obtains weights in two stages: a
Tikhonov-regularised least-squares solve, followed — when
$`\log(R)/\texttt{nlap} \geq 0.05`$ — by a *linear* Remez that holds the exponents
fixed and optimises only the weights and the error amplitude, alternating a
$`(\texttt{nlap}+1) \times (\texttt{nlap}+1)`$ Chebyshev equioscillation solve with a
Newton–Maehly update of the alternation points (up to 50 iterations).  Below
that threshold the exponential columns are nearly collinear, the linear system
is ill-conditioned, and only the weakly regularised LS stage is used.

Supplying `init_expon` / `init_weight` bypasses the table entirely; the initial
error amplitude is then sampled at $`8\,\texttt{nlap}+1`$ points and handed straight
to the Remez loop.

---

## Installation

### Python

```bash
pip install minimax-cpppy
```

Requires Python ≥ 3.9, NumPy, and a C++17 compiler.  The build system uses
[scikit-build-core](https://scikit-build-core.readthedocs.io) and
[pybind11](https://pybind11.readthedocs.io); both are fetched automatically by
pip.

To install from source (e.g. for development):

```bash
git clone https://github.com/LinusBDittmer/minimax-cpp-py
cd minimax-cpp-py
pip install -e ".[dev]"
```

### C++ (CMake)

**Option A — subdirectory of your project**

Add `minimax-cpp-py` as a subdirectory of your project and wire it up in your
`CMakeLists.txt`:

```cmake
add_subdirectory(path/to/minimax-cpp-py)
target_link_libraries(your_target PRIVATE minimax_cpppy)
```

Linking against the `minimax_cpppy` CMake target automatically propagates the
public include path, so `#include "minimax_cpppy/minimax.hpp"` resolves without
any extra `-I` flag.

**Option B — installed library**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix /your/install/prefix
```

Then in your project:

```cmake
find_package(minimax_cpppy REQUIRED)
target_link_libraries(your_target PRIVATE minimax_cpppy::minimax_cpppy)
```

Pass `-DCMAKE_PREFIX_PATH=/your/install/prefix` to CMake if the library is not
in a standard search path.

---

## Testing

### C++ tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMINIMAX_CPPPY_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For a Debug build (enables parameter-validation assertions):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMINIMAX_CPPPY_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Python tests

```bash
pip install -e ".[dev]"    # or: pip install . numpy pytest
pytest tests/python -v
```

---

## Usage — Python

```python
import minimax_cpppy as mm
```

### Standard minimax quadrature

```python
exponents, weights, errmax = mm.laplace_minimax(nlap, ymin, ymax)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `nlap` | `int` | Number of Laplace quadrature points; `1 ≤ nlap ≤ 30` |
| `ymin` | `float` | Lower interval bound; must satisfy `0 < ymin < ymax` |
| `ymax` | `float` | Upper interval bound |
| `init_expon` | `ndarray`, shape `(nlap,)`, optional, keyword-only | Initial exponents in `[ymin, ymax]`; bypasses table lookup. Must be paired with `init_weight`. |
| `init_weight` | `ndarray`, shape `(nlap,)`, optional, keyword-only | Initial weights in `[ymin, ymax]`; must be paired with `init_expon`. |
| `verbose` | `int`, optional, keyword-only | Verbosity level (default `3`): `0`=silent, `1`=errors, `2`=+warnings, `3`=+info per Remez/NR iter, `4`=+debug per Newton step. Output goes to stderr. |

| Return value | Type | Description |
|--------------|------|-------------|
| `exponents` | `numpy.ndarray`, shape `(nlap,)`, `float64` | Exponents $`a_k`$ |
| `weights` | `numpy.ndarray`, shape `(nlap,)`, `float64` | Weights $`w_k`$ |
| `errmax` | `float` | Guaranteed max $`|1/x - f(x)|`$ for all $`x \in [y_\min, y_\max]`$ |

**Errors raised**

- `ValueError` — invalid `nlap` (outside `[1, 30]`), inverted interval (`ymax ≤ ymin`), or exactly one of `init_expon`/`init_weight` provided.
- `RuntimeError` — no tabulated initial data exists for the requested `(nlap, ymax/ymin)` combination and no initial guess was supplied.

### $`L_p`$-norm minimax quadrature

```python
exponents, weights, lp_norm = mm.laplace_lp(nlap, ymin, ymax, norm_p)
```

Minimises the $`L_p`$ norm ($`\text{norm\_p} \geq 1`$) of the quadrature error
instead of the $`L_\infty`$ (minimax) norm, warm-started from the unbiased
minimax solution. Even integer `norm_p` uses $`\eta^p`$ directly; odd integer
and non-integer `norm_p` minimise $`\int |\eta|^p`$ via a kink-split
quadrature rule.  Larger `norm_p` approaches the minimax solution; `norm_p=1`
is the mean-absolute-error problem and uses a smoothed continuation. Low
non-integer `norm_p` (roughly `< 1.25`) fall back to a slower
continuation-in-`norm_p` and may still fail to converge for `norm_p` very
close to 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `nlap` | `int` | Number of quadrature points; `1 ≤ nlap ≤ 30` |
| `ymin` | `float` | Lower interval bound; must satisfy `0 < ymin < ymax` |
| `ymax` | `float` | Upper interval bound |
| `norm_p` | `float` | Loss order, `norm_p ≥ 1` |
| `verbose` | `int`, optional, keyword-only | Verbosity level (default `3`) |

Returns `(exponents, weights, lp_norm)`, where `lp_norm` is the achieved
$`L_p`$ norm — **not** the $`L_\infty`$ error (use `errmax` from
`laplace_minimax` for that).

**Errors raised**

- `ValueError` — invalid `nlap`/range, or `norm_p < 1`.
- `RuntimeError` — the Newton optimiser failed to converge.

### Density-biased quadrature

For better accuracy when the orbital energy denominators are non-uniformly
distributed, use the biased variant.  It minimises the square of the signed,
density-weighted error $`\left( \int e(t)\,\rho(t)\,dt \right)^2`$ — the
leading-order bias of the Laplace approximation — where $`\rho`$ is the empirical
density of pairwise denominators $`D_{ia} + D_{jb}`$, built internally from the
same occupied/virtual orbital energies used by `DenominatorDensity`.  See
[Density-biased optimisation](#density-biased-optimisation) for the two-phase
solve.

```python
density = mm.DenominatorDensity(occ, virt, bandwidth=1.0)
ymin, ymax = density.delta_min, density.delta_max

exponents, weights, errmax = mm.biased_laplace(
    nlap, ymin, ymax, occ, virt, bandwidth=1.0)
```

`ymin`/`ymax` must match the density implied by `occ`/`virt`/`bandwidth`
(i.e. its `delta_min`/`delta_max`) to within 1 ppm — in practice, always derive
them from a `DenominatorDensity` built with the same inputs, as above.

**`DenominatorDensity` constructor**

| Parameter | Type | Description |
|-----------|------|-------------|
| `occ` | `ndarray`, `float64` | Occupied orbital energies $`\varepsilon_i`$ |
| `virt` | `ndarray`, `float64` | Virtual orbital energies $`\varepsilon_a`$ |
| `bandwidth` | `float` | KDE bandwidth in log-space |
| `n_fft` | `int`, optional | FFT grid size (power of 2, default `4096`) |
| `n_t` | `int`, optional | Number of spline knots (default `512`) |
| `floor_frac` | `float`, optional | Minimum density floor as fraction of peak (default `1e-3`) |
| `floor_frac_max` | `float`, optional | Upper clamp on the floor fraction (default `-1.0`, i.e. unclamped) |
| `C` | `float`, optional | Constant density pedestal, in units of the uniform density (default `0.0`); `C=0` is fully biased, `C → ∞` recovers unbiased minimax |
| `n_exc` | `int`, optional | Excitation order (`1`=singles, `2`=doubles, `3`=triples, …; default `2`) |

| Property | Description |
|----------|-------------|
| `delta_min` | Minimum pairwise denominator |
| `delta_max` | Maximum pairwise denominator |
| `ratio` | `delta_max / delta_min` |

**`biased_laplace`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `nlap` | `int` | Number of quadrature points; `1 ≤ nlap ≤ 30` |
| `ymin` | `float` | Lower bound (= `delta_min` of the implied density) |
| `ymax` | `float` | Upper bound (= `delta_max` of the implied density) |
| `occ` | `ndarray`, `float64` | Occupied orbital energies |
| `virt` | `ndarray`, `float64` | Virtual orbital energies |
| `bandwidth` | `float` | KDE bandwidth (see `DenominatorDensity`) |
| `n_fft`, `n_t`, `floor_frac`, `floor_frac_max`, `C`, `n_exc` | optional, keyword-only | Same as `DenominatorDensity` |
| `verbose` | `int`, optional, keyword-only | Verbosity level (default `3`) |

Returns `(exponents, weights, errmax)`. `errmax` is the $`L_\infty`$ error on the
normalised domain, same convention as `laplace_minimax` — it is *not* the
minimised density-weighted bias, and it is expected to be *larger* than the
unbiased minimax `errmax`; that is the trade being made.

### Quickstart example

```python
import numpy as np
import minimax_cpppy as mm

# 7-point approximation of 1/x on [1, 1000]
exponents, weights, errmax = mm.laplace_minimax(7, 1.0, 1000.0)
print(f"Guaranteed max error: {errmax:.3e}")

# Evaluate f(x) = Σ_k w_k exp(-a_k x) on a log-spaced grid
x = np.geomspace(1.0, 1000.0, 500)
f = np.sum(weights * np.exp(-exponents[:, None] * x[None, :]), axis=0)
print(f"Actual max error:     {np.max(np.abs(1/x - f)):.3e}")
```

### Quantum chemistry: orbital energy denominators

```python
import numpy as np
import minimax_cpppy as mm

# D_ia = eps_a - eps_i  (virtual minus occupied orbital energies, Hartree)
D_ia = ...   # shape (nocc, nvir), all entries > 0

# Pairwise denominators D_ia + D_jb range over [2*D_min, 2*D_max]
ymin = 2.0 * D_ia.min()
ymax = 2.0 * D_ia.max()

exponents, weights, errmax = mm.laplace_minimax(7, ymin, ymax)
print(f"Max Laplace error: {errmax:.2e}")

# Laplace weights for each (i,a) pair at quadrature node t_k:
#   exp(-t_k * D_ia / 2)
# The factor 1/2 distributes the exponent symmetrically over both pairs.
for k, (t_k, w_k) in enumerate(zip(exponents, weights)):
    scale = np.exp(-0.5 * t_k * D_ia)   # shape (nocc, nvir)
    # ... accumulate MP2 energy at this Laplace point ...
```

See [`examples/python/01_basic_usage.py`](examples/python/01_basic_usage.py)
for a full walkthrough and convergence study,
[`examples/python/02_lt_mp2_application.py`](examples/python/02_lt_mp2_application.py)
for a complete SOS-DF-LT-MP2 implementation using PySCF,
[`examples/python/03_denominator_density.py`](examples/python/03_denominator_density.py)
for the `DenominatorDensity` object on its own, and
[`examples/python/04_biased_vs_unbiased.py`](examples/python/04_biased_vs_unbiased.py) /
[`examples/python/05_biased_accuracy_comparison.py`](examples/python/05_biased_accuracy_comparison.py)
for the biased quadrature workflow and its effect on SOS-DF-LT-MP2 accuracy.

---

## Usage — C++

The public API lives in `minimax_cpppy/minimax.hpp` (standard minimax),
`minimax_cpppy/laplace_lp.hpp` ($`L_p`$-norm minimax),
`minimax_cpppy/denominator_density.hpp` (`DenominatorDensity`), and
`minimax_cpppy/biasing.hpp` (density-biased variant),
all in the `minimax_cpppy` namespace.

### Data types

```cpp
namespace minimax_cpppy {

struct MinimaxResult {
    std::vector<double> expon;   // exponents a_k, length nlap
    std::vector<double> weight;  // weights   w_k, length nlap
    double              errmax;  // guaranteed max |1/x - f(x)| on [ymin, ymax]
                                  // (Lp variant: the achieved L_p norm instead)
};

// Standard minimax — uses pre-tabulated initial data.
MinimaxResult laplaceMinimax(int nlap, double ymin, double ymax,
                              int verbose = 3,
                              std::ostream& os = std::cerr);

// Overload — provide initial guess; bypasses table lookup.
MinimaxResult laplaceMinimax(int nlap, double ymin, double ymax,
                              std::vector<double> init_expon,
                              std::vector<double> init_weight,
                              int verbose = 3,
                              std::ostream& os = std::cerr);

// L_p-norm minimax (laplace_lp.hpp).
MinimaxResult laplaceLp(int nlap, double ymin, double ymax, double normP,
                         int verbose = 3, std::ostream& os = std::cerr);

// Density-biased quadrature (biasing.hpp). Builds its own DenominatorDensity
// internally from (occ, virt, bandwidth, ...); ymin/ymax must match that
// density's deltaMin()/deltaMax() within 1 ppm.
MinimaxResult biasedLaplace(
    int nlap, double ymin, double ymax,
    const double* occ, int n_occ,
    const double* virt, int n_virt,
    double bandwidth,
    int n_fft = 4096, int n_t = 512,
    double floor_frac = 1e-3, double floor_frac_max = -1.0,
    double C = 0.0, int n_exc = 2,
    int verbose = 3, std::ostream& os = std::cerr);

} // namespace minimax_cpppy
```

### Single call

```cpp
#include "minimax_cpppy/minimax.hpp"

// 7-point approximation of 1/x on [1.0, 1000.0]
minimax_cpppy::MinimaxResult r = minimax_cpppy::laplaceMinimax(7, 1.0, 1000.0);

printf("errmax = %.3e\n", r.errmax);
for (int k = 0; k < 7; ++k)
    printf("  a[%d] = %.14f   w[%d] = %.14f\n", k, r.expon[k], k, r.weight[k]);
```

**Exceptions thrown**

- `std::invalid_argument` — invalid parameters (`nlap < 1`, `nlap > 30`,
  `ymin ≤ 0`, `ymax ≤ ymin`).  Always active regardless of build type.
- `std::runtime_error` — Remez (or, for `laplaceLp`, damped-Newton)
  algorithm failed to converge for the requested `(nlap, ymax/ymin)` combination.

### Density-biased quadrature

```cpp
#include "minimax_cpppy/biasing.hpp"           // biasedLaplace
#include "minimax_cpppy/denominator_density.hpp"

// Occupied and virtual orbital energies (Hartree)
std::vector<double> occ  = { -1.2, -0.8, -0.5 };
std::vector<double> virt = {  0.4,  1.1,  2.3,  4.0 };

minimax_cpppy::DenominatorDensity density(
    occ.data(),  static_cast<int>(occ.size()),
    virt.data(), static_cast<int>(virt.size()),
    /*bandwidth=*/ 0.05);

minimax_cpppy::MinimaxResult r = minimax_cpppy::biasedLaplace(
    7, density.deltaMin(), density.deltaMax(),
    occ.data(),  static_cast<int>(occ.size()),
    virt.data(), static_cast<int>(virt.size()),
    /*bandwidth=*/ 0.05);

printf("errmax = %.3e\n", r.errmax);
```

### Quantum chemistry: orbital energy denominators

```cpp
#include <cmath>
#include <vector>
#include "minimax_cpppy/minimax.hpp"

// Mock orbital energy differences D_ia = eps_a - eps_i  (Hartree)
std::vector<double> D_values = {0.30, 0.55, 0.82, 1.40, 2.10};

double ymin = 2.0 * *std::min_element(D_values.begin(), D_values.end());
double ymax = 2.0 * *std::max_element(D_values.begin(), D_values.end());

minimax_cpppy::MinimaxResult r = minimax_cpppy::laplaceMinimax(5, ymin, ymax);
printf("errmax = %.3e\n", r.errmax);

// Laplace weights for a single D value at node t_k:
//   exp(-t_k * D / 2)
for (int ia = 0; ia < (int)D_values.size(); ++ia) {
    for (int k = 0; k < (int)r.expon.size(); ++k) {
        double scale = std::exp(-0.5 * r.expon[k] * D_values[ia]);
        // ... accumulate MP2 energy contribution ...
    }
}
```

### Building the standalone C++ examples

Each example under `examples/cpp/` is a self-contained CMake project that
pulls in `minimax_cpppy` via `add_subdirectory(../../..)`:

```bash
cd examples/cpp/01_basic_usage
cmake -B build -S .
cmake --build build
./build/basic_usage
```

For a Debug build (enables parameter-validation exceptions):

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

See [`examples/cpp/01_basic_usage/`](examples/cpp/01_basic_usage/) for the
basic-usage walkthrough,
[`examples/cpp/02_denominator_density/`](examples/cpp/02_denominator_density/)
for `DenominatorDensity` on its own, and
[`examples/cpp/03_biased_vs_unbiased/`](examples/cpp/03_biased_vs_unbiased/)
for the biased quadrature workflow.

---

## Architecture

```
include/minimax_cpppy/minimax.hpp   ← public C++ API (MinimaxResult, laplaceMinimax)
include/minimax_cpppy/laplace_lp.hpp← public C++ API (laplaceLp, L_p-norm minimax)
include/minimax_cpppy/denominator_density.hpp
                                    ← public C++ API (DenominatorDensity)
include/minimax_cpppy/biasing.hpp   ← public C++ API (biasedLaplace)
src/laplace.cpp                     ← thin shim: delegates to detail::laplaceMinimax
src/laplace_lp.cpp                  ← thin shim: delegates to the Ln-loss solver
src/biasing.cpp                     ← biased Remez + DenominatorDensity construction
src/core/algorithm.hpp              ← init dispatch + full Remez (header-only, detail namespace)
src/core/algorithm_biased.hpp       ← two-phase Newton solver for the density-biased objective
src/core/ln_loss.hpp                ← L_n-loss value/gradient/Hessian (even + |eta|^n split-rule paths)
src/core/remez.hpp                  ← remezLoop, logspaceInitFallback, computeInitialError
src/core/maehly.hpp                 ← Newton–Maehly extremum search
src/core/paraopt.hpp                ← Newton–Raphson equioscillation solver + Armijo line search
src/core/newton.hpp                 ← generic damped-Newton driver (Ln-loss and biased solvers)
src/core/dd128.hpp                  ← software 128-bit double-double arithmetic (DD type)
src/core/expint.hpp                 ← double-double exponential integral E1 (Ln-loss kernel)
src/core/quadrature.hpp             ← exact/panelled quadrature rules used by the Ln-loss solver
src/core/data_ext.hpp               ← interpolatedLookup, interpolateEntries, extTableForNlap, MAX_NLAP
src/core/data_ext_counts.{cpp,hpp}  ← EXT_COUNTS[30]: per-nlap variable row counts
src/core/data_ext_NN.{cpp,hpp}      ← two-tier non-uniform tables, nlap=01..30 (generated by gen_table)
src/core/density.hpp                ← DenominatorDensity internals (FFT KDE + spline)
src/core/ifft.hpp                   ← inverse FFT helper
src/core/log_pretty.hpp             ← banner/table/citation formatting for verbose logging
src/core/grid.hpp                   ← cuspLog10 + two-tier buildLog10Grid (ratio grid for gen_table)
python/_minimax_core.cpp            ← pybind11 module (_minimax_core)
python/minimax_cpppy/__init__.py    ← re-exports laplace_minimax, laplace_lp,
                                       biased_laplace, DenominatorDensity from _minimax_core
tools/gen_table.cpp                 ← table generator: runs Remez over the two-tier grid for nlap=1..30
tools/minimax_loss_func.cpp         ← diagnostic CLI: prints e(x) CSV on log grid
tools/denominator_density.cpp       ← diagnostic CLI: evaluates DenominatorDensity::evalW CSV
tools/laplace_lp_loss.cpp           ← internal dev tool: evaluates the Ln-loss solver at a point
tools/ln_quad_scan.cpp              ← internal dev tool: Ln-loss quadrature panel-count convergence scan
tools/dump_reference.cpp            ← internal dev tool: regenerates tests/data/reference_outputs.csv
```

### Algorithm flow (`src/core/algorithm.hpp`)

`laplaceMinimax(nlap, ymin, ymax)`:
1. Call `data::interpolatedLookup(nlap, R)` (9-point Lagrange interpolation in log10-ratio space) where `R = ymax/ymin`.
2. Dispatch based on the sampled approximation error `sampledErr`:
   - `sampledErr < 1e-12` — table holds a machine-precision solution; return directly without Remez.
   - `sampledErr ≤ 0.5` + valid extrema (bounds & sqrt-span check) — warm-start Remez from table values.
   - `sampledErr ≤ 0.5` but bad extrema — try `logspaceInitFallback(0.01)`; return LS solution if over-parameterised, otherwise cold Maehly with table init.
   - `sampledErr > 0.5` (table entry too far from this ratio) — LS dispatch, falls back to `logspaceInitFallback(10.0)` seed.
3. Remez outer loop (up to `maxIter=200`):
   - **`maehly`** — Newton + Maehly deflation to find `2*nlap-1` extrema of `e'(x)`.
   - **`paraopt`** — Newton-Raphson to update `(expon, weight, error)` to satisfy equioscillation conditions; uses Armijo backtracking via `lnsrch`.
   - Stops when `paraopt` converges in a single NR step.
4. Scale results back from normalised interval `[1, R]` to `[ymin, ymax]`.

All internal arithmetic uses `DD` (double-double, ~32 significant decimal digits) to avoid cancellation.

`laplaceLp` and `biasedLaplace` both drop the equioscillation outer loop entirely: they call `laplaceMinimax` on the normalised $`[1, R]`$ domain for a warm start, reparametrise as $`z = (\ln a_k, w_k)`$, and run a damped Newton minimisation (`newton.hpp`) of their own objective — the $`L_p`$ loss (`ln_loss.hpp`) and the squared density-weighted bias (`algorithm_biased.hpp`), respectively.  Their initialisation is therefore exactly the table-lookup/dispatch path described above, inherited from `laplaceMinimax`.

### Key constraints

- `1 ≤ nlap ≤ 30` — range covered by the ext tables.
- `0 < ymin < ymax` — validated by the public API in all build types (`std::invalid_argument`); internal assertions additionally guarded by `MINIMAX_CPPPY_DEBUG_MODE__` in Debug builds.
- Python bindings raise `ValueError` for invalid parameters and `RuntimeError` if the algorithm fails to converge.

### CLI diagnostic tools

Build:

```bash
cmake -B build_tools -DCMAKE_BUILD_TYPE=Debug \
    -DMINIMAX_CPPPY_BUILD_TOOLS=ON -DMINIMAX_CPPPY_BUILD_TESTS=OFF
cmake --build build_tools --parallel
```

This also builds `gen_table` (see "Regenerating ext tables" below) and three
tools used only for internal development (`laplace_lp_loss`, `ln_quad_scan`,
`dump_reference`) — not needed for normal use of the library.

**`minimax_loss_func <ratio> <resolution> <nlap>`**

Prints a CSV (`t,x,e_x`) of the approximation error $`e(x) = 1/x - \sum_k w_k e^{-a_k x}`$
on a log-spaced grid over $`[1, \text{ratio}]`$.

**`denominator_density <orbital_energy_file> <resolution> <bandwidth> [n_fft]`**

Reads occupied (first line) and virtual (last line) orbital energies from a
text file, builds a `DenominatorDensity`, and prints a CSV
(`t,x,w,dw,d2w`) of the spline on a uniform $`t`$-grid.

### Regenerating ext tables

Only needed if you change the Remez solver's convergence behavior or the
`[ymin, ymax]` grid:

```bash
./build_tools/gen_table src/core          # writes data_ext_01..30.{cpp,hpp}, ~30-60 min
```

---

## Parameter guide

| `nlap` | Typical use case | Rule of thumb |
|--------|-----------------|---------------|
| 3–5 | Tight denominators (`ymax/ymin < 50`) | Fast, coarse |
| 7–10 | Most molecular calculations | Good for chemical accuracy |
| 15–20 | Wide denominator ranges or high precision | High accuracy |
| ≤ 30 | Maximum supported | |

More quadrature points reduce the Laplace approximation error but increase the
cost of each Laplace iteration.  For most closed-shell molecular calculations,
`nlap = 7` achieves errors well below 0.1 mEh.

---

## Citation

If you use this library in published work, please cite:

```bibtex
@article{helmich2016improvements,
  title={Improvements on the minimax algorithm for the Laplace transformation of orbital energy denominators},
  author={Helmich-Paris, Benjamin and Visscher, Lucas},
  journal={Journal of Computational Physics},
  volume={321},
  pages={927--931},
  year={2016},
  publisher={Elsevier}
}
```

---

## License

See [LICENSE](LICENSE).
