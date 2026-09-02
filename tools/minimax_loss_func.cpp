/**
 * tools/minimax_loss_func.cpp
 *
 * Diagnostic CLI: evaluates the *unbiased* minimax approximation error
 * e(x) = 1/x - sum_k w_k * exp(-a_k * x) on a log-spaced grid using
 * double-double (DD) arithmetic.
 *
 * Two modes (selected by argument count):
 *
 * Unbiased (points/weights from standard laplaceMinimax over [1, ratio]):
 *   minimax_loss_func <ratio> <resolution> <nlap>
 *     ratio      : ymax/ymin, must be > 1.0
 *     resolution : number of grid points, must be >= 2
 *     nlap       : number of Laplace points, must be in [1, 30]
 *   Grid spans [1, ratio].
 *
 * Biased (points/weights from biasedLaplace, error still unbiased):
 *   minimax_loss_func <orbital_file> <resolution> <nlap> <bandwidth> [n_fft] [n_exc]
 *     orbital_file : first line = occ energies, last line = virt energies
 *     bandwidth    : KDE bandwidth in t-space, must be > 0
 *     n_fft        : FFT grid size, power of 2, default 4096
 *     n_exc        : excitation order (1=singles, 2=doubles, ...), default 2
 *   Grid spans [deltaMin, deltaMax] from the density.
 *
 * Output: CSV to stdout with header "t,x,e_x" and one row per grid point.
 *   t_i = ln(x_i),  x_i log-spaced over the grid interval
 *   e_x = 1/x_i - sum_k weight[k] * exp(-expon[k] * x_i)   (DD arithmetic)
 */

#include "minimax_cpppy/minimax.hpp"
#include "minimax_cpppy/biasing.hpp"
#include "dd128.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using minimax_cpppy::detail::DD;

static void print_usage(const char* prog) {
    std::cerr << "Usage (unbiased): " << prog << " <ratio> <resolution> <nlap>\n"
              << "  ratio      : ymax/ymin, must be > 1.0\n"
              << "  resolution : number of grid points, must be >= 2\n"
              << "  nlap       : number of Laplace points, must be in [1, 30]\n"
              << "\n"
              << "Usage (biased): " << prog
              << " <orbital_file> <resolution> <nlap> <bandwidth> [n_fft] [n_exc]\n"
              << "  orbital_file : first line = occ energies, last line = virt energies\n"
              << "  bandwidth    : KDE bandwidth in t-space, must be > 0\n"
              << "  n_fft        : FFT grid size, power of 2, default 4096\n"
              << "  n_exc        : excitation order (1=singles, 2=doubles, ...), default 2\n";
}

// ponytail: copied from denominator_density.cpp; one tool reading orbital files,
// not worth a shared header for two call sites.
static bool read_orbital_file(
    const char*          path,
    std::vector<double>& occ,
    std::vector<double>& virt)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open '" << path << "'\n";
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) lines.push_back(line);

    std::string first_line, last_line;
    for (const auto& l : lines) {
        if (!l.empty() && l.find_first_not_of(" \t\r\n") != std::string::npos) {
            if (first_line.empty()) first_line = l;
            last_line = l;
        }
    }

    if (first_line.empty()) {
        std::cerr << "Error: orbital file '" << path << "' has no data lines.\n";
        return false;
    }
    if (first_line == last_line) {
        std::cerr << "Error: orbital file '" << path
                  << "' needs at least two non-empty lines (occ and virt).\n";
        return false;
    }

    auto parse_line = [](const std::string& s, std::vector<double>& out,
                         const char* label) -> bool {
        std::istringstream iss(s);
        double v;
        while (iss >> v) out.push_back(v);
        if (out.empty()) {
            std::cerr << "Error: " << label << " line is empty or unparseable.\n";
            return false;
        }
        return true;
    };

    return parse_line(first_line, occ, "occupied") &&
           parse_line(last_line, virt, "virtual");
}

// Parse an int that must consume the whole token.
static bool parse_int(const char* s, int& out, const char* name) {
    try {
        std::size_t pos;
        out = std::stoi(std::string(s), &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("trailing");
    } catch (const std::exception&) {
        std::cerr << "minimax_loss_func: invalid " << name << " '" << s << "'\n";
        return false;
    }
    return true;
}

static bool parse_double(const char* s, double& out, const char* name) {
    try {
        std::size_t pos;
        out = std::stod(std::string(s), &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("trailing");
    } catch (const std::exception&) {
        std::cerr << "minimax_loss_func: invalid " << name << " '" << s << "'\n";
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    const bool biased = (argc >= 5);
    if (argc != 4 && !(argc >= 5 && argc <= 7)) {
        std::cerr << "Error: expected 3 args (unbiased) or 4-6 args (biased), got "
                  << (argc - 1) << ".\n";
        print_usage(argv[0]);
        return 1;
    }

    // resolution and nlap are common to both modes (argv[2], argv[3]).
    int resolution, nlap;
    if (!parse_int(argv[2], resolution, "resolution")) return 1;
    if (!parse_int(argv[3], nlap, "nlap")) return 1;
    if (resolution < 2) {
        std::cerr << "Error: resolution must be >= 2, got " << resolution << ".\n";
        return 1;
    }
    if (nlap < 1 || nlap > 30) {
        std::cerr << "Error: nlap must be in [1, 30], got " << nlap << ".\n";
        return 1;
    }

    // Obtain quadrature points/weights and the grid interval [xlo, xhi].
    minimax_cpppy::MinimaxResult result;
    double xlo, xhi;

    if (!biased) {
        double ratio;
        if (!parse_double(argv[1], ratio, "ratio")) return 1;
        if (ratio <= 1.0) {
            std::cerr << "Error: ratio must be > 1.0, got " << ratio << ".\n";
            return 1;
        }
        try {
            result = minimax_cpppy::laplaceMinimax(nlap, 1.0, ratio);
        } catch (const std::exception& e) {
            std::cerr << "Error: laplaceMinimax failed: " << e.what() << "\n";
            return 1;
        }
        xlo = 1.0;
        xhi = ratio;
    } else {
        const char* orbital_file = argv[1];
        double bandwidth;
        int    n_fft = 4096;
        int    n_exc = 2;
        if (!parse_double(argv[4], bandwidth, "bandwidth")) return 1;
        if (argc >= 6 && !parse_int(argv[5], n_fft, "n_fft")) return 1;
        if (argc == 7 && !parse_int(argv[6], n_exc, "n_exc")) return 1;
        if (bandwidth <= 0.0) {
            std::cerr << "Error: bandwidth must be > 0, got " << bandwidth << ".\n";
            return 1;
        }
        if (n_fft < 2 || (n_fft & (n_fft - 1)) != 0) {
            std::cerr << "Error: n_fft must be a power of 2 >= 2, got " << n_fft << ".\n";
            return 1;
        }
        if (n_exc < 1) {
            std::cerr << "Error: n_exc must be >= 1, got " << n_exc << ".\n";
            return 1;
        }

        std::vector<double> occ, virt;
        if (!read_orbital_file(orbital_file, occ, virt)) return 1;

        try {
            minimax_cpppy::DenominatorDensity dens(
                occ.data(),  static_cast<int>(occ.size()),
                virt.data(), static_cast<int>(virt.size()),
                bandwidth, n_fft, 512, 1e-3, -1.0, 0.0, n_exc);
            xlo = dens.deltaMin();
            xhi = dens.deltaMax();
            result = minimax_cpppy::biasedLaplace(
                nlap, xlo, xhi,
                occ.data(),  static_cast<int>(occ.size()),
                virt.data(), static_cast<int>(virt.size()),
                bandwidth, n_fft, 512, 1e-3, -1.0, 0.0, n_exc);
        } catch (const std::exception& e) {
            std::cerr << "Error: biased optimisation failed: " << e.what() << "\n";
            return 1;
        }
    }

    const std::vector<double>& expon  = result.expon;
    const std::vector<double>& weight = result.weight;

    // Precompute DD versions of exponents and weights
    std::vector<DD> expon_dd(nlap), weight_dd(nlap);
    for (int k = 0; k < nlap; ++k) {
        expon_dd[k]  = DD(expon[k]);
        weight_dd[k] = DD(weight[k]);
    }

    const double t_lo     = std::log(xlo);
    const double t_span   = std::log(xhi) - t_lo;
    const int    n_minus1 = resolution - 1;

    // Output CSV
    std::cout << "t,x,e_x\n";
    std::cout << std::scientific << std::setprecision(15);

    const DD one(1.0);

    for (int i = 0; i < resolution; ++i) {
        // x log-spaced over [xlo, xhi]
        double t_d = t_lo + (static_cast<double>(i) / static_cast<double>(n_minus1)) * t_span;
        DD t(t_d);
        DD x = DD::ddExp(t);

        // unbiased error e(x) = 1/x - sum_k weight[k] * exp(-expon[k] * x)
        DD ex = one / x;
        for (int k = 0; k < nlap; ++k) {
            DD neg_ak_x = -expon_dd[k] * x;
            DD term = weight_dd[k] * DD::ddExp(neg_ak_x);
            ex = ex - term;
        }

        std::cout << t.hi << ',' << x.hi << ',' << ex.hi << '\n';
    }

    return 0;
}
