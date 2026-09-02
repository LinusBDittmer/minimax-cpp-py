/**
 * @file log_pretty.hpp
 * @brief Formatting primitives for verbose logging: run banner, citation
 *        placeholder block, and fixed-width table cell formatters.
 *
 * Used by the three public entry points (src/laplace.cpp, src/laplace_lp.cpp)
 * for the banner/citation, and by newton.hpp/remez.hpp/paraopt.hpp for the
 * per-iteration convergence tables (verbose level 3 / INFO). Formats into a
 * local buffer via snprintf rather than std::ios manipulators so it never
 * mutates the format flags/precision of the caller-supplied std::ostream,
 * which may be a shared stream (e.g. std::cout in a test).
 */
#pragma once
#ifndef MINIMAX_CPPPY_LOG_PRETTY_HPP
#define MINIMAX_CPPPY_LOG_PRETTY_HPP

#include <cstdio>
#include <ostream>
#include <string>

namespace minimax_cpppy {
namespace detail {

inline std::string fmtCell(double value, int width, int prec) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%*.*e", width, prec, value);
    return std::string(buf);
}

inline std::string fmtCellInt(long long value, int width) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%*lld", width, value);
    return std::string(buf);
}

inline std::string fmtHeaderCell(const char* label, int width) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%*s", width, label);
    return std::string(buf);
}

inline std::string ruleCell(int width) {
    return std::string(static_cast<size_t>(width < 0 ? 0 : width), '-');
}

inline void printRunBanner(std::ostream& os, const char* fnName,
                            int nlap, double ymin, double ymax) {
    char range[64];
    std::snprintf(range, sizeof(range), "[%.2e, %.2e]", ymin, ymax);
    char ratio[32];
    std::snprintf(ratio, sizeof(ratio), "%.1e", ymax / ymin);
    os << "=== " << fnName << "  nlap=" << nlap
       << "  range=" << range << "  ratio=" << ratio << " ===\n";
}

inline void printCitationBlock(std::ostream& os) {
    os << "----------------------------------------\n"
          " Please cite:\n"
          "   Helmich-Paris, B.; Visscher, L. Improvements on the minimax\n"
          "   algorithm for the Laplace transformation of orbital energy\n"
          "   denominators. J. Comput. Phys. 2016, 321, 927-931.\n"
          "----------------------------------------\n";
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_LOG_PRETTY_HPP
