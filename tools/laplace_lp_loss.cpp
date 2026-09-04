// tools/laplace_lp_loss.cpp
#include "minimax_cpppy/laplace_lp.hpp"
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " nlap ymin ymax norm_p\n";
        return 1;
    }
    int nlap = std::atoi(argv[1]);
    double ymin = std::atof(argv[2]);
    double ymax = std::atof(argv[3]);
    int normP = std::atoi(argv[4]);
    // normP>=1: odd normP / normP=1 minimise the |eta|^p loss; even normP is unchanged.
    auto r = minimax_cpppy::laplaceLp(nlap, ymin, ymax, normP, 2, std::cerr);
    std::cout << "# k,expon,weight\n";
    for (int k = 0; k < nlap; ++k)
        std::cout << k << "," << r.expon[k] << "," << r.weight[k] << "\n";
    std::cout << "# L_p norm = " << r.errmax << "\n";
    return 0;
}
