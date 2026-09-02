#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax_test {

struct TestCase {
    std::string           name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline int runAll() {
    int passed = 0, failed = 0;
    for (auto& tc : registry()) {
        try {
            tc.fn();
            std::cout << "  PASS  " << tc.name << "\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "  FAIL  " << tc.name << "\n        " << e.what() << "\n";
            ++failed;
        } catch (...) {
            std::cout << "  FAIL  " << tc.name << "\n        (unknown exception)\n";
            ++failed;
        }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

} // namespace minimax_test

#define MINIMAX_TEST(name)                                                      \
    static void _minimax_test_body_##name();                                    \
    static ::minimax_test::Registrar _minimax_reg_##name(                       \
        #name, _minimax_test_body_##name);                                      \
    static void _minimax_test_body_##name()

#define MINIMAX_REQUIRE(cond)                                                   \
    do {                                                                        \
        if (!(cond)) {                                                          \
            throw std::runtime_error(                                           \
                std::string("REQUIRE failed: " #cond " [") +                   \
                __FILE__ + ":" + std::to_string(__LINE__) + "]");              \
        }                                                                       \
    } while (0)

#define MINIMAX_REQUIRE_THROW_TYPE(expr, ExcType)                               \
    do {                                                                        \
        bool _caught = false;                                                   \
        try { expr; } catch (const ExcType&) { _caught = true; }               \
        catch (...) {}                                                          \
        if (!_caught) {                                                         \
            throw std::runtime_error(                                           \
                std::string("REQUIRE_THROW_TYPE<" #ExcType ">: wrong or no "   \
                            "exception from: " #expr " [") +                   \
                __FILE__ + ":" + std::to_string(__LINE__) + "]");              \
        }                                                                       \
    } while (0)

#define MINIMAX_REQUIRE_NO_THROW(expr)                                          \
    do {                                                                        \
        try { expr; }                                                           \
        catch (const std::exception& _e) {                                      \
            throw std::runtime_error(                                           \
                std::string("REQUIRE_NO_THROW: unexpected exception: ") +       \
                _e.what() + " from: " #expr " [" + __FILE__ + ":" +           \
                std::to_string(__LINE__) + "]");                                \
        }                                                                       \
    } while (0)

#define MINIMAX_REQUIRE_CLOSE(a, b, tol)                                        \
    do {                                                                        \
        double _a = (a), _b = (b), _t = (tol);                                 \
        double _ref = std::abs(_b) > 1e-300 ? std::abs(_b) : 1.0;              \
        double _rel = std::abs(_a - _b) / _ref;                                \
        if (_rel > _t) {                                                        \
            throw std::runtime_error(                                           \
                std::string("REQUIRE_CLOSE failed: |") + std::to_string(_a) +  \
                " - " + std::to_string(_b) + "| / ref(" +                      \
                std::to_string(_ref) + ") = " + std::to_string(_rel) +         \
                " > " + std::to_string(_t) +                                   \
                " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]");       \
        }                                                                       \
    } while (0)

#define MINIMAX_RUN_TESTS() return ::minimax_test::runAll()
