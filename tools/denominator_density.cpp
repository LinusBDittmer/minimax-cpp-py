/**
 * tools/denominator_density.cpp
 *
 * Diagnostic CLI: evaluates DenominatorDensity::evalW on a uniform t-grid in
 * [0, ln(ratio)] where ratio = deltaMax/deltaMin.
 *
 * Usage: denominator_density <orbital_energy_file> <resolution> <bandwidth> [n_fft] [n_exc]
 *   orbital_energy_file : text file; first line = space-separated occupied energies,
 *                         last line = space-separated virtual energies
 *   resolution          : number of grid points, must be >= 2
 *   bandwidth           : KDE bandwidth in t-space, must be > 0
 *   n_fft               : FFT grid size (power of 2, default 4096).  The FFT grid
 *                         covers [0, 2·t_max] in t-space (buffer approach), so the
 *                         physical region [0, t_max] has ~n_fft/ratio_phys knots.
 *                         Increase n_fft proportionally to ratio_phys for best results.
 *   n_exc               : excitation order (1=singles, 2=doubles, ...), default 2.
 *
 * Output: CSV to stdout with header "t,x,w,dw,d2w" and one row per grid point.
 *   t_i = (i / (resolution-1)) * ln(ratio)
 *   x_i = exp(t_i)   [normalised denominator, in [1, ratio]]
 *   w, dw, d2w = DenominatorDensity::evalW(t_i)
 */

#include "minimax_cpppy/biasing.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <orbital_energy_file> <resolution> <bandwidth> [n_fft] [n_exc]\n"
              << "  orbital_energy_file : first line = occ energies, last line = virt energies\n"
              << "  resolution          : number of grid points, >= 2\n"
              << "  bandwidth           : KDE bandwidth in t-space, > 0\n"
              << "  n_fft               : FFT grid size, power of 2, default 4096\n"
              << "  n_exc               : excitation order (1=singles, 2=doubles, ...), default 2\n";
}

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
    while (std::getline(f, line)) {
        lines.push_back(line);
    }

    // Find first and last non-empty lines
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

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Error: expected 3–5 arguments, got " << (argc - 1) << ".\n";
        print_usage(argv[0]);
        return 1;
    }

    const char* orbital_file = argv[1];

    int    resolution;
    double bandwidth;
    int    n_fft = 4096;
    int    n_exc = 2;

    try {
        std::size_t pos_res, pos_bw;
        resolution = std::stoi(std::string(argv[2]), &pos_res);
        if (pos_res != std::strlen(argv[2])) {
            std::cerr << "denominator_density: invalid resolution '" << argv[2] << "'\n";
            return 1;
        }
        bandwidth = std::stod(std::string(argv[3]), &pos_bw);
        if (pos_bw != std::strlen(argv[3])) {
            std::cerr << "denominator_density: invalid bandwidth '" << argv[3] << "'\n";
            return 1;
        }
        if (argc >= 5) {
            std::size_t pos_fft;
            n_fft = std::stoi(std::string(argv[4]), &pos_fft);
            if (pos_fft != std::strlen(argv[4])) {
                std::cerr << "denominator_density: invalid n_fft '" << argv[4] << "'\n";
                return 1;
            }
        }
        if (argc == 6) {
            std::size_t pos_exc;
            n_exc = std::stoi(std::string(argv[5]), &pos_exc);
            if (pos_exc != std::strlen(argv[5])) {
                std::cerr << "denominator_density: invalid n_exc '" << argv[5] << "'\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to parse arguments: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    if (resolution < 2) {
        std::cerr << "Error: resolution must be >= 2, got " << resolution << ".\n";
        print_usage(argv[0]);
        return 1;
    }
    if (bandwidth <= 0.0) {
        std::cerr << "Error: bandwidth must be > 0, got " << bandwidth << ".\n";
        print_usage(argv[0]);
        return 1;
    }
    if (n_fft < 2 || (n_fft & (n_fft - 1)) != 0) {
        std::cerr << "Error: n_fft must be a power of 2 >= 2, got " << n_fft << ".\n";
        print_usage(argv[0]);
        return 1;
    }
    if (n_exc < 1) {
        std::cerr << "Error: n_exc must be >= 1, got " << n_exc << ".\n";
        print_usage(argv[0]);
        return 1;
    }

    std::vector<double> occ, virt;
    if (!read_orbital_file(orbital_file, occ, virt)) {
        return 1;
    }

    // Construct density object
    minimax_cpppy::DenominatorDensity dens = [&]() {
        try {
            return minimax_cpppy::DenominatorDensity(
                occ.data(),  static_cast<int>(occ.size()),
                virt.data(), static_cast<int>(virt.size()),
                bandwidth,
                n_fft,
                512,    // n_t default
                1e-3,   // floor_frac default
                -1.0,   // floor_frac_max default
                0.0,    // C default
                n_exc);
        } catch (const std::exception& e) {
            std::cerr << "Error: DenominatorDensity construction failed: "
                      << e.what() << "\n";
            std::exit(1);
        }
    }();

    const double ln_ratio = std::log(dens.ratio());
    const int    n_minus1 = resolution - 1;

    // Output CSV
    std::cout << "t,x,w,dw,d2w\n";
    std::cout << std::scientific << std::setprecision(15);

    for (int i = 0; i < resolution; ++i) {
        double t = (static_cast<double>(i) / static_cast<double>(n_minus1)) * ln_ratio;
        double x = std::exp(t);

        double w = 0.0, dw = 0.0, d2w = 0.0;
        dens.evalW(t, w, dw, d2w);

        std::cout << t << ',' << x << ',' << w << ',' << dw << ',' << d2w << '\n';
    }

    return 0;
}
