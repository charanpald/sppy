#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>
namespace sppy {
// Integer arithmetic follows NumPy's fixed-width modular behavior without
// invoking signed C++ overflow. Floating/complex arithmetic stays in its dtype.
template <class T> T product_value(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        U bits = U(uint64_t(U(a)) * uint64_t(U(b)));
        T value;
        std::memcpy(&value, &bits, sizeof(T));
        return value;
    } else {
        return a * b;
    }
}
template <class T> T sum_value(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        U bits = U(uint64_t(U(a)) + uint64_t(U(b)));
        T value;
        std::memcpy(&value, &bits, sizeof(T));
        return value;
    } else {
        return a + b;
    }
}

} // namespace sppy
