// tools/ln_quad_scan.cpp
//
// Diagnostic CLI: scans quadrature panel multiplier 1,2,4,...,64 for a single
// (nlap, ymin, ymax, n, kind) Ln-loss case, reporting how the gradient and
// Hessian inf-norms (not just the L value) converge as panel count grows,
// plus rule-build and eval wall time.
//
// Usage: ln_quad_scan <nlap> <ymin> <ymax> <n> <kind>
//   kind : even | abs
//
// See docs/superpowers/specs/2026-07-02-ln-quadrature-point-experiment-design.md
#include "minimax_cpppy/minimax.hpp"
#include "ln_loss.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace minimax_cpppy::detail;
using minimax_cpppy::MinimaxResult;

namespace {

class Stopwatch {
public:
    Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0_)
            .count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

int basePanelsFor(int nlap) { return 64 > 8 * nlap ? 64 : 8 * nlap; }

// Mirrors buildLnRule's per-multiplier panel count (single segment [0,b]).
QuadRule buildEvenRule(double R, int nlap, int mult) {
    const double b = std::log(R);
    const int panels = basePanelsFor(nlap) * mult;
    return compositeGaussLegendre(0.0, b, panels);
}

// Mirrors buildSplitRule's per-multiplier panel count, given precomputed zeros.
QuadRule buildAbsRule(double R, int nlap, int mult, const std::vector<double>& zeros) {
    const double b = std::log(R);
    std::vector<double> brk;
    brk.push_back(0.0);
    for (double z : zeros) brk.push_back(z);
    brk.push_back(b);
    const int basePanels = basePanelsFor(nlap);
    QuadRule rule;
    for (size_t s = 0; s + 1 < brk.size(); ++s) {
        const double lo = brk[s], hi = brk[s + 1];
        int seg = static_cast<int>(std::ceil((hi - lo) / b * basePanels * mult));
        if (seg < 1) seg = 1;
        QuadRule r = compositeGaussLegendre(lo, hi, seg);
        rule.t.insert(rule.t.end(), r.t.begin(), r.t.end());
        rule.w.insert(rule.w.end(), r.w.begin(), r.w.end());
    }
    return rule;
}

double ddInfNorm(const std::vector<DD>& v) {
    double m = 0.0;
    for (const DD& x : v) m = std::max(m, std::fabs(x.hi));
    return m;
}

std::string fmt(double v) {
    if (std::isnan(v)) return "";
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

double relChange(double cur, double prev, bool havePrev) {
    if (!havePrev) return std::nan("");
    double ref = std::fabs(prev) > 1e-300 ? std::fabs(prev) : 1.0;
    return std::fabs(cur - prev) / ref;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: " << argv[0] << " nlap ymin ymax n kind(even|abs)\n";
        return 1;
    }
    const int nlap = std::atoi(argv[1]);
    const double ymin = std::atof(argv[2]);
    const double ymax = std::atof(argv[3]);
    const double n = std::atof(argv[4]);
    const std::string kind = argv[5];
    if (kind != "even" && kind != "abs") {
        std::cerr << "kind must be 'even' or 'abs'\n";
        return 1;
    }

    const double R = ymax / ymin;
    MinimaxResult mm = minimax_cpppy::laplaceMinimax(nlap, 1.0, R, 0);

    std::vector<DD> a(nlap), w(nlap);
    for (int k = 0; k < nlap; ++k) { a[k] = DD(mm.expon[k]); w[k] = DD(mm.weight[k]); }

    std::vector<double> zeros;
    if (kind == "abs") {
        Stopwatch sw;
        zeros = findEtaZeros(R, nlap, a.data(), w.data());
        std::cout << "# findEtaZeros: " << zeros.size() << " zeros in "
                  << sw.elapsedMs() << " ms\n";
    }

    std::cout << "# mult,points,L,gradInfNorm,gradRelChange,hessInfNorm,hessRelChange,buildMs,evalMs\n";

    const int dim = 2 * nlap;
    std::vector<DD> gA(nlap), gW(nlap), gAll(dim);
    std::vector<DD> hAW(static_cast<size_t>(dim) * dim);

    double prevGrad = 0.0, prevHess = 0.0;
    bool havePrev = false;

    const int mults[] = {1, 2, 4, 8, 16, 32, 64};
    for (int mult : mults) {
        Stopwatch swBuild;
        QuadRule rule = (kind == "even") ? buildEvenRule(R, nlap, mult)
                                          : buildAbsRule(R, nlap, mult, zeros);
        const double buildMs = swBuild.elapsedMs();

        Stopwatch swEval;
        DD L = (kind == "even")
                   ? evalLnLoss(rule, nlap, static_cast<int>(n), a.data(), w.data(),
                                gA.data(), gW.data(), hAW.data())
                   : evalLnLossAbs(rule, nlap, n, a.data(), w.data(),
                                   gA.data(), gW.data(), hAW.data());
        const double evalMs = swEval.elapsedMs();

        for (int k = 0; k < nlap; ++k) { gAll[k] = gA[k]; gAll[nlap + k] = gW[k]; }
        const double gradNorm = ddInfNorm(gAll);
        const double hessNorm = ddInfNorm(hAW);
        const double gradRel = relChange(gradNorm, prevGrad, havePrev);
        const double hessRel = relChange(hessNorm, prevHess, havePrev);

        std::cout << mult << "," << rule.t.size() << "," << fmt(L.hi) << ","
                  << fmt(gradNorm) << "," << fmt(gradRel) << ","
                  << fmt(hessNorm) << "," << fmt(hessRel) << ","
                  << fmt(buildMs) << "," << fmt(evalMs) << "\n";

        prevGrad = gradNorm; prevHess = hessNorm; havePrev = true;
    }
    return 0;
}
