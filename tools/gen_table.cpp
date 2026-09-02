/**
 * tools/gen_table.cpp
 *
 * Generate data_ext_NN.hpp for NN = 01..30.
 *
 * Usage:   gen_table <output_dir>
 * Build:   cmake -DMINIMAX_CPPPY_BUILD_GEN_TABLE=ON (Release recommended)
 *
 * The generated .hpp files are committed to the repository and included by
 * src/core/data_ext_NN.cpp.  Re-run this tool only to regenerate data.
 *
 * Algorithm per nlap (1..30, all covered by the existing ext tables):
 *   1. Build a small two-tier non-uniform grid (core/grid.hpp::buildLog10Grid):
 *      dense cusp-clustered nodes plus a sparse uniform tail (~60-140 nodes
 *      total, vs. the previous fixed 3000-point uniform grid).
 *   2. For each grid node independently, seed from the still-present dense
 *      (3000-point) ext table via interpolatedLookup (near machine-precision
 *      seed, incl. extrema) and run remezLoop. No neighbor-to-neighbor
 *      marching is needed: the dense table gives a strong seed at every
 *      ratio, so each node converges on its own.
 *   3. Write data_ext_NN.hpp (variable length) to output_dir, and a
 *      data_ext_counts.cpp with the real per-nlap node counts.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/algorithm.hpp"
#include "core/grid.hpp"

// Initial two-tier grid params per nlap. Task 6 raises n_dense where the C++
// regression exceeds 1e-7. Kept as a small editable function (not a table)
// so Task 6 can tune thresholds without touching call sites.
static minimax_cpppy::detail::GridParams gridParamsFor(int nlap) {
    using minimax_cpppy::detail::GridParams;
    // Dense side carries the no-polish / over-resolved region (grows with nlap);
    // sparse tail is the polished region (Remez fixes crude seeds there).
    int n_dense  = 60;
    int n_sparse = 40;
    if (nlap >= 16) { n_dense = 90; n_sparse = 30; }
    if (nlap >= 24) { n_dense = 120; n_sparse = 20; }
    return GridParams{ n_dense, n_sparse, /*half_width_dec=*/3.5, /*margin_dec=*/1.0 };
}

struct Entry {
    double range;
    double errmax;
    double expon[53];
    double weight[53];
    double extrema[105];
};

// Compute remezLoop at (nlap, ratio) starting from (init_expon, init_weight, init_errmax).
// If init_extrema is non-null, it provides a warm-start hint for maehlySolver (length 2*nlap-1).
static bool compute(int nlap, double ratio,
                    const double* init_expon, const double* init_weight, double init_errmax,
                    Entry& out,
                    const double* init_extrema = nullptr)
{
    using namespace minimax_cpppy::detail;

    std::vector<DD> exp_dd(nlap), w_dd(nlap);
    for (int k = 0; k < nlap; ++k) {
        exp_dd[k] = DD(init_expon[k]);
        w_dd[k]   = DD(init_weight[k]);
    }
    DD erramp = DD(std::abs(init_errmax));
    if (erramp <= DD(0.0)) {
        erramp = computeInitialError(exp_dd.data(), w_dd.data(), nlap, ratio, 2 * nlap + 1);
    }

    // Validate and scale extrema hint: must lie strictly inside (1, ratio).
    // The ext table stores extrema for the ratio they were computed at; when using
    // as a warm start for a nearby ratio we may need to clamp slightly.
    const int n_ext = 2 * nlap - 1;
    std::vector<double> extrema_hint;
    if (init_extrema != nullptr && n_ext > 0) {
        extrema_hint.resize(n_ext);
        bool valid = true;
        for (int k = 0; k < n_ext; ++k) {
            double x = init_extrema[k];
            if (x <= 1.0 || x >= ratio) { valid = false; break; }
            extrema_hint[k] = x;
        }
        if (!valid) extrema_hint.clear();
    }

    std::vector<double> extrema;
    try {
        const std::vector<double>* hint_ptr = extrema_hint.empty() ? nullptr : &extrema_hint;
        auto result = remezLoop(nlap, 1.0, ratio,
                                std::move(exp_dd), std::move(w_dd), erramp,
                                200, 1e-10, 1e-15, 0.3, 1e-6, 1e-4,
                                hint_ptr, &extrema);

        out.range  = ratio;
        out.errmax = result.errmax;
        std::memset(out.expon,   0, sizeof(out.expon));
        std::memset(out.weight,  0, sizeof(out.weight));
        std::memset(out.extrema, 0, sizeof(out.extrema));

        for (int k = 0; k < nlap; ++k) {
            out.expon[k]  = result.expon[k];
            out.weight[k] = result.weight[k];
        }
        for (int k = 0; k < n_ext && k < (int)extrema.size(); ++k) {
            out.extrema[k] = extrema[k];
        }
        return true;
    } catch (...) {
        return false;
    }
}

// Seed from the existing ext table (nlap=1..MAX_NLAP).
// Also fills extrema_out (length 2*nlap-1) with the table's interior extrema.
static bool seedFromExtTable(int nlap, double ratio,
                             double* expon_out, double* weight_out, double& errmax_out,
                             double* extrema_out = nullptr)
{
    using namespace minimax_cpppy::detail;
    if (nlap < 1 || nlap > data::MAX_NLAP) return false;
    try {
        data::ExtTableEntry e = data::interpolatedLookup(nlap, ratio);
        for (int k = 0; k < nlap; ++k) {
            expon_out[k]  = e.expon[k];
            weight_out[k] = e.weight[k];
        }
        errmax_out = std::abs(e.errmax);
        if (extrema_out) {
            const int n_ext = 2 * nlap - 1;
            for (int k = 0; k < n_ext; ++k)
                extrema_out[k] = e.extrema[k];
        }
        return true;
    } catch (...) {
        return false;
    }
}

// Seed using log-spaced fallback (last resort for hard cases).
static bool seedFromLogspace(int nlap, double ratio,
                             double* expon_out, double* weight_out, double& errmax_out)
{
    using namespace minimax_cpppy::detail;
    std::vector<DD> exp_dd, w_dd;
    DD err;
    // Generous threshold: we only need a seed; remezLoop handles convergence.
    if (!logspaceInitFallback(exp_dd, w_dd, err, nlap, ratio, 1.0))
        return false;
    for (int k = 0; k < nlap; ++k) {
        expon_out[k]  = exp_dd[k].hi;
        weight_out[k] = w_dd[k].hi;
    }
    errmax_out = err.hi;
    return true;
}

static void writeFile(const char* outdir, int nlap, const std::vector<Entry>& entries, int count)
{
    std::string path = std::string(outdir) + "/data_ext_" +
                       (nlap < 10 ? "0" : "") + std::to_string(nlap) + ".hpp";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        throw std::runtime_error(std::string("gen_table: cannot open ") + path);
    }

    // No #include or namespace here -- file is #include'd by data_ext_NN.cpp
    // inside the minimax_cpppy::detail::data namespace block.
    std::fprintf(f, "// AUTO-GENERATED by tools/gen_table -- do not edit.\n");
    std::fprintf(f, "// Included by src/core/data_ext_%02d.cpp inside the data namespace.\n", nlap);
    std::fprintf(f, "const ExtTableEntry EXT_TABLE_%02d[%d] = {\n", nlap, count);

    for (int i = 0; i < count; ++i) {
        const Entry& e = entries[i];
        std::fprintf(f, "  { %.20e, %.20e,\n", e.range, e.errmax);
        std::fprintf(f, "    {");
        for (int k = 0; k < 53; ++k) {
            std::fprintf(f, " %.20e%s", e.expon[k], k < 52 ? "," : "");
        }
        std::fprintf(f, " },\n");
        std::fprintf(f, "    {");
        for (int k = 0; k < 53; ++k) {
            std::fprintf(f, " %.20e%s", e.weight[k], k < 52 ? "," : "");
        }
        std::fprintf(f, " },\n");
        std::fprintf(f, "    {");
        for (int k = 0; k < 105; ++k) {
            std::fprintf(f, " %.20e%s", e.extrema[k], k < 104 ? "," : "");
        }
        std::fprintf(f, " } }%s\n", i < count - 1 ? "," : "");
    }

    std::fprintf(f, "};\n");
    std::fclose(f);
    std::fprintf(stderr, "  wrote %s\n", path.c_str());

    // Write companion .cpp that defines the compile unit (namespace wrapper).
    std::string cpp_path = std::string(outdir) + "/data_ext_" +
                           (nlap < 10 ? "0" : "") + std::to_string(nlap) + ".cpp";
    FILE* fc = std::fopen(cpp_path.c_str(), "w");
    if (!fc) {
        throw std::runtime_error(std::string("gen_table: cannot open ") + cpp_path);
    }
    std::fprintf(fc, "// AUTO-GENERATED by tools/gen_table -- do not edit.\n");
    std::fprintf(fc, "// Defines EXT_TABLE_%02d for nlap=%d.\n", nlap, nlap);
    std::fprintf(fc, "#include \"data_ext.hpp\"\n");
    std::fprintf(fc, "namespace minimax_cpppy {\n");
    std::fprintf(fc, "namespace detail {\n");
    std::fprintf(fc, "namespace data {\n");
    std::fprintf(fc, "#include \"data_ext_%02d.hpp\"\n", nlap);
    std::fprintf(fc, "} // namespace data\n");
    std::fprintf(fc, "} // namespace detail\n");
    std::fprintf(fc, "} // namespace minimax_cpppy\n");
    std::fclose(fc);
    std::fprintf(stderr, "  wrote %s\n", cpp_path.c_str());
}

// Write src/core/data_ext_counts.cpp with the real per-nlap node counts.
static void writeCounts(const char* outdir, const int* counts)
{
    std::string path = std::string(outdir) + "/data_ext_counts.cpp";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        throw std::runtime_error(std::string("gen_table: cannot open ") + path);
    }
    std::fprintf(f, "// AUTO-GENERATED by tools/gen_table -- do not edit.\n");
    std::fprintf(f, "#include \"data_ext_counts.hpp\"\n");
    std::fprintf(f, "namespace minimax_cpppy { namespace detail { namespace data {\n");
    std::fprintf(f, "const int EXT_COUNTS[30] = {");
    for (int i = 0; i < 30; ++i) std::fprintf(f, "%d%s", counts[i], i < 29 ? "," : "");
    std::fprintf(f, "};\n} } }\n");
    std::fclose(f);
    std::fprintf(stderr, "  wrote %s\n", path.c_str());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: gen_table <output_dir>\n");
        return 1;
    }
    const char* outdir = argv[1];

    int counts[30] = {};

    // Direct per-node seeding: the dense (3000-point) ext table compiled into
    // this binary supplies a near-machine-precision seed (incl. extrema) at
    // any ratio, so every node on the new small grid converges independently.
    // No neighbor-to-neighbor marching or copy-on-failure is needed.
    for (int nlap = 1; nlap <= 30; ++nlap) {
        std::vector<double> nodes = minimax_cpppy::detail::buildLog10Grid(nlap, gridParamsFor(nlap));
        const int count = static_cast<int>(nodes.size());
        std::vector<Entry> entries(count);

        std::fprintf(stderr, "nlap=%d: %d grid nodes\n", nlap, count);

        for (int j = 0; j < count; ++j) {
            const double ratio = std::pow(10.0, nodes[j]);
            double e[53] = {}, w[53] = {}, ext[105] = {}, err = 0.0;
            bool ok = false;

            if (seedFromExtTable(nlap, ratio, e, w, err, ext))
                ok = compute(nlap, ratio, e, w, err, entries[j], ext);

            if (!ok && seedFromExtTable(nlap, ratio, e, w, err, ext))   // retry without extrema hint
                ok = compute(nlap, ratio, e, w, err, entries[j]);

            if (!ok && seedFromLogspace(nlap, ratio, e, w, err))
                ok = compute(nlap, ratio, e, w, err, entries[j]);

            // Degenerate over-resolved regime (e.g. nlap>=5 at ratio~1): the
            // approximation is already exact to ~machine zero, so Remez fails to
            // make progress / converge. The dense-table interpolated seed IS the
            // near-exact solution there, so store it directly. Guard on a tiny
            // seed error so a genuine resolved-region failure still hard-errors.
            // The 1e-6 guard is deliberately looser than the downstream
            // over-resolved gate (1e-8 actual quadrature residual, see
            // check_against_reference.cpp); that gate is the real backstop and
            // fails the build if any stored seed isn't truly near-exact.
            if (!ok && seedFromExtTable(nlap, ratio, e, w, err, ext) && err < 1e-6) {
                entries[j].range  = ratio;
                entries[j].errmax = err;
                std::memcpy(entries[j].expon,   e,   sizeof(entries[j].expon));
                std::memcpy(entries[j].weight,  w,   sizeof(entries[j].weight));
                std::memcpy(entries[j].extrema, ext, sizeof(entries[j].extrema));
                ok = true;
            }

            if (!ok) {
                std::fprintf(stderr, "ERROR nlap=%d node=%d ratio=%.3e\n", nlap, j, ratio);
                return 1;
            }
        }

        writeFile(outdir, nlap, entries, count);
        counts[nlap - 1] = count;
        std::fprintf(stderr, "nlap=%d: %d nodes written\n", nlap, count);
    }

    writeCounts(outdir, counts);

    std::fprintf(stderr, "Done.\n");
    return 0;
}
