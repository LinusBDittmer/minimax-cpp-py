/**
 * @file ifft.hpp
 * @brief Radix-2 inverse FFT (Cooley-Tukey DIT) for real output.
 *
 * Provides a single template function `ifft<T>` that maps a Hermitian complex
 * spectrum back to a real signal.  Used by `buildDensityArrays` (density.hpp)
 * to recover the pair-denominator KDE density from its characteristic function.
 */
#pragma once
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace minimax_cpppy {
namespace detail {

namespace {

/** @brief Return true iff n is a positive power of two. */
inline bool is_power_of_two(std::size_t n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief In-place bit-reversal permutation for radix-2 FFT/IFFT.
 *
 * Rearranges elements of x so that element at index i moves to the index
 * obtained by bit-reversing i.  Required before the Cooley-Tukey butterfly
 * pass in decimation-in-time (DIT) algorithms.
 *
 * @param x Complex array to permute in place; length must be a power of two.
 */
template<typename T>
void bit_reverse_permute(std::vector<std::complex<T>>& x) noexcept {
    const std::size_t n = x.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }
}

/**
 * @brief Unnormalized in-place radix-2 DIT inverse FFT (Cooley-Tukey).
 *
 * Twiddle factors use +2π/len (conjugate of the forward FFT convention)
 * to implement the inverse direction.  The caller is responsible for the
 * 1/N normalisation.
 *
 * @param x Complex array to transform in place; length must be a power of two.
 */
template<typename T>
void ifft_inplace(std::vector<std::complex<T>>& x) {
    const std::size_t n = x.size();
    bit_reverse_permute(x);
    static constexpr double two_pi = 2.0 * 3.14159265358979323846264338327950288;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const T angle = static_cast<T>(two_pi) / static_cast<T>(len);
        const std::complex<T> w_base(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<T> w(static_cast<T>(1), static_cast<T>(0));
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<T> u = x[i + k];
                const std::complex<T> v = x[i + k + len / 2] * w;
                x[i + k]           = u + v;
                x[i + k + len / 2] = u - v;
                w *= w_base;
            }
        }
    }
}

} // anonymous namespace

/**
 * @brief Inverse FFT: Hermitian complex spectrum → real signal.
 *
 * Normalization: 1/N.  Imaginary parts of output discarded (Hermitian input assumed).
 * Input size must be a positive power of two.
 *
 * @throws std::runtime_error if spectrum.size() is 0 or not a power of two.
 */
template<typename T>
std::vector<T> ifft(std::vector<std::complex<T>> spectrum) {
    const std::size_t n = spectrum.size();
    if (!is_power_of_two(n))
        throw std::runtime_error(
            "ifft: size must be a positive power of two, got " + std::to_string(n));

    ifft_inplace(spectrum);

    const T inv_n = static_cast<T>(1) / static_cast<T>(n);
    std::vector<T> result(n);
    for (std::size_t i = 0; i < n; ++i)
        result[i] = spectrum[i].real() * inv_n;
    return result;
}

} // namespace detail
} // namespace minimax_cpppy
