// tests/test_log_pretty.cpp
#include "test_helpers.hpp"
#include "core/log_pretty.hpp"
#include <sstream>
#include <string>

using namespace minimax_cpppy::detail;

MINIMAX_TEST(fmt_cell_int_right_aligns) {
    MINIMAX_REQUIRE(fmtCellInt(3, 4) == "   3");
    MINIMAX_REQUIRE(fmtCellInt(0, 4) == "   0");
    MINIMAX_REQUIRE(fmtCellInt(12345, 4) == "12345");  // widens instead of truncating
}

MINIMAX_TEST(fmt_cell_formats_scientific) {
    std::string s = fmtCell(3.2010e-04, 12, 4);
    MINIMAX_REQUIRE(s.find("3.2010e-04") != std::string::npos);
    MINIMAX_REQUIRE(s.size() >= 12);
}

MINIMAX_TEST(fmt_header_cell_right_aligns) {
    MINIMAX_REQUIRE(fmtHeaderCell("iter", 4) == "iter");
    MINIMAX_REQUIRE(fmtHeaderCell("f", 4) == "   f");
}

MINIMAX_TEST(rule_cell_is_dashes_of_given_width) {
    MINIMAX_REQUIRE(ruleCell(4) == "----");
    MINIMAX_REQUIRE(ruleCell(8) == "--------");
}

MINIMAX_TEST(run_banner_contains_fields) {
    std::ostringstream oss;
    printRunBanner(oss, "testFn", 7, 1.0, 100.0);
    std::string out = oss.str();
    MINIMAX_REQUIRE(out.find("testFn") != std::string::npos);
    MINIMAX_REQUIRE(out.find("nlap=7") != std::string::npos);
    MINIMAX_REQUIRE(out.find("ratio=1.0e+02") != std::string::npos);
    MINIMAX_REQUIRE(out.back() == '\n');
}

MINIMAX_TEST(citation_block_has_reference) {
    std::ostringstream oss;
    printCitationBlock(oss);
    MINIMAX_REQUIRE(
        oss.str().find("Helmich-Paris") != std::string::npos);
}

int main() { MINIMAX_RUN_TESTS(); }
