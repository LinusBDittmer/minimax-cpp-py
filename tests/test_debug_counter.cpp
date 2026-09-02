#include "test_helpers.hpp"
#include "minimax_cpppy/minimax.hpp"
#include <iostream>

// All tests in this file require Debug mode (counter only exists there).
// In Release builds the test binary is compiled but CTest marks it DISABLED,
// so the main() below is never invoked.  The #ifndef guard is a belt-and-
// suspenders safety net in case someone runs the binary directly.
#ifndef MINIMAX_CPPPY_DEBUG_MODE__

int main() {
    std::cout << "test_debug_counter: skipped (not a Debug build)\n";
    return 0;
}

#else // MINIMAX_CPPPY_DEBUG_MODE__ is defined

MINIMAX_TEST(counter_starts_at_zero_after_reset) {
    minimax_cpppy::debug::reset_functional1_call_count();
    MINIMAX_REQUIRE(minimax_cpppy::debug::get_functional1_call_count() == 0);
}

MINIMAX_TEST(counter_positive_after_laplaceMinimax) {
    minimax_cpppy::debug::reset_functional1_call_count();
    [[maybe_unused]] auto r = minimax_cpppy::laplaceMinimax(3, 1.0, 100.0);
    long long count = minimax_cpppy::debug::get_functional1_call_count();
    std::cout << "    functional1 calls (nlap=3): " << count << "\n";
    MINIMAX_REQUIRE(count > 0);
}

MINIMAX_TEST(counter_deterministic_same_args) {
    minimax_cpppy::debug::reset_functional1_call_count();
    minimax_cpppy::laplaceMinimax(5, 1.0, 1000.0);
    long long first = minimax_cpppy::debug::get_functional1_call_count();
    std::cout << "    functional1 calls (nlap=5, run 1): " << first << "\n";

    minimax_cpppy::debug::reset_functional1_call_count();
    minimax_cpppy::laplaceMinimax(5, 1.0, 1000.0);
    long long second = minimax_cpppy::debug::get_functional1_call_count();
    std::cout << "    functional1 calls (nlap=5, run 2): " << second << "\n";

    MINIMAX_REQUIRE(first == second);
}

MINIMAX_TEST(counter_accumulates_across_calls) {
    minimax_cpppy::debug::reset_functional1_call_count();
    minimax_cpppy::laplaceMinimax(3, 1.0, 100.0);
    long long after_one = minimax_cpppy::debug::get_functional1_call_count();
    MINIMAX_REQUIRE(after_one > 0);

    minimax_cpppy::laplaceMinimax(3, 1.0, 100.0);
    long long after_two = minimax_cpppy::debug::get_functional1_call_count();

    MINIMAX_REQUIRE(after_two == 2 * after_one);
}

MINIMAX_TEST(reset_zeroes_counter) {
    minimax_cpppy::laplaceMinimax(3, 1.0, 100.0);
    minimax_cpppy::debug::reset_functional1_call_count();
    MINIMAX_REQUIRE(minimax_cpppy::debug::get_functional1_call_count() == 0);
}

int main() { MINIMAX_RUN_TESTS(); }

#endif // MINIMAX_CPPPY_DEBUG_MODE__
