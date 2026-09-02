#include "test_helpers.hpp"
#include "minimax_cpppy/minimax.hpp"
#include "molecular_orbital_data.hpp"
#include <cmath>
#include <iostream>

static constexpr int NLAP_LIST[] = {5, 7, 10, 15};
static constexpr int N_NLAP      = 4;

static void test_molecule_accuracy(const mol_data::MoleculeData& mol)
{
    const int     no   = mol.n_occ;
    const int     nv   = mol.n_virt;
    const double* occ  = mol.occ;
    const double* virt = mol.virt;

    // sorted arrays: xmin = 2*virt[0] - 2*occ[no-1], xmax = 2*virt[nv-1] - 2*occ[0]
    const double xmin = 2.0 * virt[0]      - 2.0 * occ[no - 1];
    const double xmax = 2.0 * virt[nv - 1] - 2.0 * occ[0];
    MINIMAX_REQUIRE(xmin > 0.0);

    double prev_errmax = 1e99;
    for (int ni = 0; ni < N_NLAP; ++ni) {
        const int nlap = NLAP_LIST[ni];
        auto r = minimax_cpppy::laplaceMinimax(nlap, xmin, xmax);

        double max_err = 0.0;
        for (int i = 0; i < no; ++i) {
            for (int j = i; j < no; ++j) {
                for (int a = 0; a < nv; ++a) {
                    for (int b = a; b < nv; ++b) {
                        const double x = virt[a] + virt[b] - occ[i] - occ[j];
                        double approx = 0.0;
                        for (int k = 0; k < nlap; ++k)
                            approx += r.weight[k] * std::exp(-r.expon[k] * x);
                        max_err = std::max(max_err, std::abs(1.0 / x - approx));
                    }
                }
            }
        }

        std::cout << "    " << mol.name
                  << " nlap=" << nlap
                  << " max_err=" << max_err
                  << " errmax=" << r.errmax << "\n";
        MINIMAX_REQUIRE(max_err <= 1.5 * r.errmax);
        MINIMAX_REQUIRE(r.errmax < prev_errmax);
        prev_errmax = r.errmax;
    }
}

MINIMAX_TEST(all_denominators_positive) {
    for (int m = 0; m < mol_data::N_MOLECULES; ++m) {
        const auto& mol   = mol_data::MOLECULES[m];
        const double xmin = 2.0 * mol.virt[0] - 2.0 * mol.occ[mol.n_occ - 1];
        MINIMAX_REQUIRE(xmin > 0.0);
    }
}

MINIMAX_TEST(ne_accuracy)      { test_molecule_accuracy(mol_data::MOLECULES[0]); }
MINIMAX_TEST(h2o_accuracy)     { test_molecule_accuracy(mol_data::MOLECULES[1]); }
MINIMAX_TEST(nh3_accuracy)     { test_molecule_accuracy(mol_data::MOLECULES[2]); }
MINIMAX_TEST(co2_accuracy)     { test_molecule_accuracy(mol_data::MOLECULES[3]); }
MINIMAX_TEST(benzene_accuracy) { test_molecule_accuracy(mol_data::MOLECULES[4]); }

int main() { MINIMAX_RUN_TESTS(); }
