// include/minimax_cpppy/minimax_lp.hpp
#pragma once
#include <iosfwd>
#include "minimax_cpppy/minimax.hpp"

namespace minimax_cpppy {

// Minimise the L_p norm of the Laplace quadrature error for 1/x on [ymin, ymax].
// normP is a real exponent >= 1 (validated in Debug builds / by the Python layer).
// Even integers use eta^p directly; odd integers and arbitrary non-integer normP
// minimise the L_p norm of |eta| via a kink-split rule; normP=1 uses a smoothed-L1
// continuation. Low non-integer normP (roughly normP < 1.25) stall when warm-started
// from minimax (the |eta|^{p-2} Hessian is singular at eta's zeros); these fall back
// to a continuation in normP -- anchor at normP=2 (or 3) and bisect toward the
// target -- which converges but is markedly slower. Very low orders (normP -> 1+,
// and normP < 1, which is not a norm) may still exhaust the continuation and throw.
// Warm-started from the unbiased minimax solution; exponents reparametrised as
// a_k = exp(p_k) (kept positive), weights free.
// result.errmax holds the L_p NORM (1/ymin)·(∫ |η|^p dt)^{1/p}, NOT the L_inf error.
// Throws std::invalid_argument (Debug) on bad nlap/range/normP; std::runtime_error
// if the damped-Newton optimiser fails to converge.
MinimaxResult laplaceLp(int nlap, double ymin, double ymax, double normP,
                        int verbose = 3, std::ostream& os = std::cerr);

} // namespace minimax_cpppy
