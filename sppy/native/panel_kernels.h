#pragma once
#include <complex>
#include <cstddef>
namespace sppy {
bool panel_simd_available();
#define SPPY_PANEL_DECL(T)                                                                         \
    bool panel_gemm(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, const T *, std::ptrdiff_t,     \
                    const T *, std::ptrdiff_t, T *, std::ptrdiff_t);
SPPY_PANEL_DECL(float)
SPPY_PANEL_DECL(double)
SPPY_PANEL_DECL(std::complex<float>)
SPPY_PANEL_DECL(std::complex<double>)
#undef SPPY_PANEL_DECL
template <class T>
bool panel_gemm(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, const T *, std::ptrdiff_t,
                const T *, std::ptrdiff_t, T *, std::ptrdiff_t) {
    return false;
}

// These assign fresh output buffers. A false return leaves output unspecified;
// callers recompute through Eigen, including for nonfinite vector results.
#define SPPY_COMPLEX_DECL(T)                                                                       \
    bool complex_csr_vector(std::ptrdiff_t, const T *, const int *, const int *, const T *, T *);  \
    bool complex_sparse_block(std::ptrdiff_t, std::ptrdiff_t, bool, const T *, const int *,        \
                              const int *, const T *, T *);
SPPY_COMPLEX_DECL(std::complex<float>)
SPPY_COMPLEX_DECL(std::complex<double>)
#undef SPPY_COMPLEX_DECL
} // namespace sppy
