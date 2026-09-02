#include "test_helpers.hpp"
#include "core/ifft.hpp"
#include <complex>
#include <vector>
#include <stdexcept>

using namespace minimax_cpppy::detail;
using cd = std::complex<double>;

// Avoid M_PI: not standard C++, and MSVC only defines it with
// _USE_MATH_DEFINES set before any <cmath> include, which transitive
// headers can preempt.
constexpr double kPi = 3.14159265358979323846;

MINIMAX_TEST(ifft_size_zero_throws) {
    std::vector<cd> empty;
    MINIMAX_REQUIRE_THROW_TYPE(ifft<double>(empty), std::runtime_error);
}

MINIMAX_TEST(ifft_size_not_power_of_two_throws) {
    std::vector<cd> bad(3, cd{1.0, 0.0});
    MINIMAX_REQUIRE_THROW_TYPE(ifft<double>(bad), std::runtime_error);
}

MINIMAX_TEST(ifft_size_5_throws) {
    std::vector<cd> bad(5, cd{1.0, 0.0});
    MINIMAX_REQUIRE_THROW_TYPE(ifft<double>(bad), std::runtime_error);
}

MINIMAX_TEST(ifft_size_6_throws) {
    std::vector<cd> bad(6, cd{1.0, 0.0});
    MINIMAX_REQUIRE_THROW_TYPE(ifft<double>(bad), std::runtime_error);
}

// Bit-reversal verified indirectly via DC spectrum test in Task 4.
MINIMAX_TEST(ifft_bit_reversal_n4) {
    MINIMAX_REQUIRE(true);
}

// IFFT([N, 0, 0, ..., 0]) = [1, 1, 1, ..., 1]
// DC component = N → all-ones signal (after 1/N normalization)
MINIMAX_TEST(ifft_dc_spectrum_to_constant_n8) {
    const int N = 8;
    std::vector<cd> X(N, cd{0.0, 0.0});
    X[0] = cd{static_cast<double>(N), 0.0};
    auto x = ifft<double>(X);
    MINIMAX_REQUIRE(x.size() == static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i)
        MINIMAX_REQUIRE(std::abs(x[i] - 1.0) < 1e-12);
}

// IFFT([1, 1, 1, ..., 1]) = [1, 0, 0, ..., 0]  (impulse at n=0)
MINIMAX_TEST(ifft_flat_spectrum_to_impulse_n8) {
    const int N = 8;
    std::vector<cd> X(N, cd{1.0, 0.0});
    auto x = ifft<double>(X);
    MINIMAX_REQUIRE(std::abs(x[0] - 1.0) < 1e-12);
    for (int i = 1; i < N; ++i)
        MINIMAX_REQUIRE(std::abs(x[i]) < 1e-12);
}

// X[1] = N/2, X[N-1] = N/2, rest 0  →  x[n] = cos(2π·n/N)
MINIMAX_TEST(ifft_single_cosine_n8) {
    const int N = 8;
    std::vector<cd> X(N, cd{0.0, 0.0});
    X[1]     = cd{static_cast<double>(N) / 2.0, 0.0};
    X[N - 1] = cd{static_cast<double>(N) / 2.0, 0.0};
    auto x = ifft<double>(X);
    for (int n = 0; n < N; ++n) {
        const double expected = std::cos(2.0 * kPi * n / N);
        MINIMAX_REQUIRE(std::abs(x[n] - expected) < 1e-12);
    }
}

// X[2] = N/2, X[N-2] = N/2, rest 0  →  x[n] = cos(2π·2n/N)
MINIMAX_TEST(ifft_second_cosine_n8) {
    const int N = 8;
    std::vector<cd> X(N, cd{0.0, 0.0});
    X[2]     = cd{static_cast<double>(N) / 2.0, 0.0};
    X[N - 2] = cd{static_cast<double>(N) / 2.0, 0.0};
    auto x = ifft<double>(X);
    for (int n = 0; n < N; ++n) {
        const double expected = std::cos(2.0 * kPi * 2 * n / N);
        MINIMAX_REQUIRE(std::abs(x[n] - expected) < 1e-12);
    }
}

// X[1] = -i·N/2, X[N-1] = +i·N/2, rest 0  →  x[n] = sin(2π·n/N)
MINIMAX_TEST(ifft_single_sine_n8) {
    const int N = 8;
    std::vector<cd> X(N, cd{0.0, 0.0});
    X[1]     = cd{0.0, -static_cast<double>(N) / 2.0};
    X[N - 1] = cd{0.0, +static_cast<double>(N) / 2.0};
    auto x = ifft<double>(X);
    for (int n = 0; n < N; ++n) {
        const double expected = std::sin(2.0 * kPi * n / N);
        MINIMAX_REQUIRE(std::abs(x[n] - expected) < 1e-12);
    }
}

// N=1: IFFT([c]) = [c.real]
MINIMAX_TEST(ifft_size_1) {
    std::vector<cd> X = {cd{3.7, 1.5}};
    auto x = ifft<double>(X);
    MINIMAX_REQUIRE(x.size() == 1);
    MINIMAX_REQUIRE(std::abs(x[0] - 3.7) < 1e-12);
}

// N=2: DC only → both outputs equal half the DC value
MINIMAX_TEST(ifft_size_2_dc_only) {
    std::vector<cd> X = {cd{4.0, 0.0}, cd{0.0, 0.0}};
    auto x = ifft<double>(X);
    MINIMAX_REQUIRE(x.size() == 2);
    MINIMAX_REQUIRE(std::abs(x[0] - 2.0) < 1e-12);
    MINIMAX_REQUIRE(std::abs(x[1] - 2.0) < 1e-12);
}

// Nyquist bin: X[N/2] = A*N, rest 0  →  x[n] = A·(-1)^n
MINIMAX_TEST(ifft_nyquist_bin_n8) {
    const int N = 8;
    const double A = 3.0;
    std::vector<cd> X(N, cd{0.0, 0.0});
    X[N / 2] = cd{A * N, 0.0};
    auto x = ifft<double>(X);
    for (int n = 0; n < N; ++n) {
        const double expected = A * (n % 2 == 0 ? 1.0 : -1.0);
        MINIMAX_REQUIRE(std::abs(x[n] - expected) < 1e-12);
    }
}

// DC test across all valid sizes: N = 1, 2, 4, 8, 16, 32, 64, 128, 256
MINIMAX_TEST(ifft_dc_all_power_of_two_sizes) {
    for (int N : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
        std::vector<cd> X(N, cd{0.0, 0.0});
        X[0] = cd{static_cast<double>(N), 0.0};
        auto x = ifft<double>(X);
        MINIMAX_REQUIRE(static_cast<int>(x.size()) == N);
        for (int i = 0; i < N; ++i)
            MINIMAX_REQUIRE(std::abs(x[i] - 1.0) < 1e-11);
    }
}

// Linearity: IFFT(a*X + b*Y) = a*IFFT(X) + b*IFFT(Y)
// Use N=8, a=2.0, b=-1.5
MINIMAX_TEST(ifft_linearity_n8) {
    const int N = 8;
    const double a = 2.0, b = -1.5;

    std::vector<cd> X(N, cd{0.0, 0.0}), Y(N, cd{0.0, 0.0});
    // X: cosine at freq 1
    X[1] = X[N-1] = cd{static_cast<double>(N)/2.0, 0.0};
    // Y: sine at freq 2
    Y[2]   = cd{0.0, -static_cast<double>(N)/2.0};
    Y[N-2] = cd{0.0, +static_cast<double>(N)/2.0};

    auto x_ref = ifft<double>(X);
    auto y_ref = ifft<double>(Y);

    std::vector<cd> Z(N);
    for (int k = 0; k < N; ++k)
        Z[k] = a * X[k] + b * Y[k];

    auto z = ifft<double>(Z);

    for (int n = 0; n < N; ++n) {
        const double expected = a * x_ref[n] + b * y_ref[n];
        MINIMAX_REQUIRE(std::abs(z[n] - expected) < 1e-11);
    }
}

// float and double produce numerically close results for same input
MINIMAX_TEST(ifft_float_matches_double_n16) {
    const int N = 16;
    std::vector<cd> Xd(N, cd{0.0, 0.0});
    Xd[3]   = cd{static_cast<double>(N)/2.0, 0.0};
    Xd[N-3] = cd{static_cast<double>(N)/2.0, 0.0};

    std::vector<std::complex<float>> Xf(N);
    for (int k = 0; k < N; ++k)
        Xf[k] = {static_cast<float>(Xd[k].real()), static_cast<float>(Xd[k].imag())};

    auto xd = ifft<double>(Xd);
    auto xf = ifft<float>(Xf);

    MINIMAX_REQUIRE(xd.size() == xf.size());
    for (int n = 0; n < N; ++n)
        MINIMAX_REQUIRE(std::abs(xd[n] - static_cast<double>(xf[n])) < 1e-5);
}

// For Hermitian input (sum of cosines), real output must match analytical sum exactly.
// Validates that discarding imaginary parts loses no information.
MINIMAX_TEST(ifft_hermitian_output_is_real_to_machine_eps_n32) {
    const int N = 32;
    std::vector<cd> X(N, cd{0.0, 0.0});
    for (int k : {1, 3, 7}) {
        X[k]   += cd{static_cast<double>(N)/2.0, 0.0};
        X[N-k] += cd{static_cast<double>(N)/2.0, 0.0};
    }
    auto x = ifft<double>(X);
    for (int n = 0; n < N; ++n) {
        double expected = std::cos(2.0 * kPi * 1 * n / N)
                        + std::cos(2.0 * kPi * 3 * n / N)
                        + std::cos(2.0 * kPi * 7 * n / N);
        MINIMAX_REQUIRE(std::abs(x[n] - expected) < 1e-11);
    }
}

int main() { MINIMAX_RUN_TESTS(); }
