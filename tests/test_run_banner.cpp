// tests/test_run_banner.cpp
#include "test_helpers.hpp"
#include "minimax_cpppy/minimax.hpp"
#include <sstream>
#include <string>

MINIMAX_TEST(banner_prints_at_verbose_1_no_citation) {
    std::ostringstream oss;
    (void)minimax_cpppy::laplaceMinimax(4, 1.0, 10.0, /*verbose=*/1, oss);
    std::string out = oss.str();
    MINIMAX_REQUIRE(out.find("=== laplaceMinimax") != std::string::npos);
    MINIMAX_REQUIRE(out.find("nlap=4") != std::string::npos);
    MINIMAX_REQUIRE(out.find("Please cite") == std::string::npos);
}

MINIMAX_TEST(citation_prints_at_verbose_3_after_solver_output) {
    std::ostringstream oss;
    (void)minimax_cpppy::laplaceMinimax(4, 1.0, 10.0, /*verbose=*/3, oss);
    std::string out = oss.str();
    MINIMAX_REQUIRE(out.find("=== laplaceMinimax") != std::string::npos);
    size_t citePos = out.find("Please cite");
    MINIMAX_REQUIRE(citePos != std::string::npos);
    size_t lastRemezPos = out.rfind("[Remez");
    MINIMAX_REQUIRE(lastRemezPos != std::string::npos);
    MINIMAX_REQUIRE(citePos > lastRemezPos);
}

MINIMAX_TEST(remez_table_header_and_columns_present) {
    std::ostringstream oss;
    (void)minimax_cpppy::laplaceMinimax(4, 1.0, 10.0, /*verbose=*/3, oss);
    std::string out = oss.str();
    MINIMAX_REQUIRE(out.find("[Remez]\n") != std::string::npos);
    MINIMAX_REQUIRE(out.find("errmax") != std::string::npos);
    MINIMAX_REQUIRE(out.find("NR_iters") != std::string::npos);
}

MINIMAX_TEST(paraopt_table_header_and_columns_present) {
    std::ostringstream oss;
    (void)minimax_cpppy::laplaceMinimax(4, 1.0, 10.0, /*verbose=*/3, oss);
    std::string out = oss.str();
    MINIMAX_REQUIRE(out.find("[ParaOpt]\n") != std::string::npos);
    MINIMAX_REQUIRE(out.find("stepNorm") != std::string::npos);
    MINIMAX_REQUIRE(out.find("||F||^2") != std::string::npos);
}

MINIMAX_TEST(no_output_at_verbose_0) {
    std::ostringstream oss;
    (void)minimax_cpppy::laplaceMinimax(4, 1.0, 10.0, /*verbose=*/0, oss);
    MINIMAX_REQUIRE(oss.str().empty());
}

int main() { MINIMAX_RUN_TESTS(); }
