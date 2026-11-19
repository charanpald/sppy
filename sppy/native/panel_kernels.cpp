#include "panel_kernels.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SPPY_VECTOR_TARGET __attribute__((target("avx2,fma")))
#define SPPY_VECTOR_AVAILABLE 1
#else
#define SPPY_VECTOR_AVAILABLE 0
#endif
namespace sppy {
bool panel_simd_available() {
#if SPPY_VECTOR_AVAILABLE
    static const bool enabled = [] {
        const char *disabled = std::getenv("SPPY_DISABLE_SIMD");
        const bool forced = disabled && disabled[0] && !(disabled[0] == '0' && disabled[1] == '\0');
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && !forced;
    }();
    return enabled;
#else
    return false;
#endif
}
#if SPPY_VECTOR_AVAILABLE
template <class T> struct V;
template <> struct V<double> {
    using P = __m256d;
    static constexpr int size = 4;
    SPPY_VECTOR_TARGET static P zero() { return _mm256_setzero_pd(); }
    SPPY_VECTOR_TARGET static P load(const double *p) { return _mm256_loadu_pd(p); }
    SPPY_VECTOR_TARGET static void store(double *p, P v) { _mm256_storeu_pd(p, v); }
    SPPY_VECTOR_TARGET static P set(double x) { return _mm256_set1_pd(x); }
    SPPY_VECTOR_TARGET static P set(std::complex<double> x) {
        return _mm256_setr_pd(x.real(), x.imag(), x.real(), x.imag());
    }
    SPPY_VECTOR_TARGET static P add(P a, P b) { return _mm256_add_pd(a, b); }
    SPPY_VECTOR_TARGET static P madd(P a, P b, P c) { return _mm256_fmadd_pd(a, b, c); }
    SPPY_VECTOR_TARGET static P cmadd(P a, P b, P c) {
        P re = _mm256_movedup_pd(a), im = _mm256_permute_pd(a, 15);
        P swap = _mm256_permute_pd(b, 5);
        return add(c, _mm256_fmaddsub_pd(re, b, _mm256_mul_pd(im, swap)));
    }
};
template <> struct V<float> {
    using P = __m256;
    static constexpr int size = 8;
    SPPY_VECTOR_TARGET static P zero() { return _mm256_setzero_ps(); }
    SPPY_VECTOR_TARGET static P load(const float *p) { return _mm256_loadu_ps(p); }
    SPPY_VECTOR_TARGET static void store(float *p, P v) { _mm256_storeu_ps(p, v); }
    SPPY_VECTOR_TARGET static P set(float x) { return _mm256_set1_ps(x); }
    SPPY_VECTOR_TARGET static P set(std::complex<float> x) {
        return _mm256_setr_ps(x.real(), x.imag(), x.real(), x.imag(), x.real(), x.imag(), x.real(),
                              x.imag());
    }
    SPPY_VECTOR_TARGET static P add(P a, P b) { return _mm256_add_ps(a, b); }
    SPPY_VECTOR_TARGET static P madd(P a, P b, P c) { return _mm256_fmadd_ps(a, b, c); }
    SPPY_VECTOR_TARGET static P cmadd(P a, P b, P c) {
        P re = _mm256_moveldup_ps(a), im = _mm256_movehdup_ps(a);
        P swap = _mm256_permute_ps(b, 0xb1);
        return add(c, _mm256_fmaddsub_ps(re, b, _mm256_mul_ps(im, swap)));
    }
};
template <class T> struct Real {
    using type = T;
    static constexpr bool complex = false;
};
template <class T> struct Real<std::complex<T>> {
    using type = T;
    static constexpr bool complex = true;
};
template <int Columns, class T>
SPPY_VECTOR_TARGET void panel_tile(std::ptrdiff_t rows, std::ptrdiff_t depth, const T *a,
                                   std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb, T *out,
                                   std::ptrdiff_t ldc) {
    using R = typename Real<T>::type;
    using Vec = V<R>;
    using P = typename Vec::P;
    constexpr int lanes = Vec::size / (Real<T>::complex ? 2 : 1);
    std::ptrdiff_t row = 0;
    for (; row + lanes <= rows; row += lanes) {
        P sums[Columns];
        for (int c = 0; c < Columns; ++c)
            sums[c] = Vec::zero();
        for (std::ptrdiff_t k = 0; k < depth; ++k) {
            P av = Vec::load(reinterpret_cast<const R *>(a + row + k * lda));
            for (int c = 0; c < Columns; ++c) {
                P bv = Vec::set(b[k + c * ldb]);
                if constexpr (Real<T>::complex)
                    sums[c] = Vec::cmadd(av, bv, sums[c]);
                else
                    sums[c] = Vec::madd(av, bv, sums[c]);
            }
        }
        for (int c = 0; c < Columns; ++c) {
            R *dst = reinterpret_cast<R *>(out + row + c * ldc);
            Vec::store(dst, Vec::add(Vec::load(dst), sums[c]));
        }
    }
    for (; row < rows; ++row)
        for (int c = 0; c < Columns; ++c) {
            T sum = T(0);
            for (std::ptrdiff_t k = 0; k < depth; ++k)
                sum += a[row + k * lda] * b[k + c * ldb];
            out[row + c * ldc] += sum;
        }
}
template <class T>
SPPY_VECTOR_TARGET void panel_run(std::ptrdiff_t rows, std::ptrdiff_t cols, std::ptrdiff_t depth,
                                  const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                                  T *out, std::ptrdiff_t ldc) {
    std::ptrdiff_t c = 0;
    for (; c + 4 <= cols; c += 4)
        panel_tile<4>(rows, depth, a, lda, b + c * ldb, ldb, out + c * ldc, ldc);
    switch (cols - c) {
    case 3:
        panel_tile<3>(rows, depth, a, lda, b + c * ldb, ldb, out + c * ldc, ldc);
        break;
    case 2:
        panel_tile<2>(rows, depth, a, lda, b + c * ldb, ldb, out + c * ldc, ldc);
        break;
    case 1:
        panel_tile<1>(rows, depth, a, lda, b + c * ldb, ldb, out + c * ldc, ldc);
        break;
    }
}
template <class T>
SPPY_VECTOR_TARGET bool csr_vector_run(std::ptrdiff_t rows, const T *values, const int *indices,
                                       const int *pointers, const T *rhs, T *out) {
    using R = typename Real<T>::type;
    using Vec = V<R>;
    using P = typename Vec::P;
    constexpr int lanes = Vec::size / 2;
    for (std::ptrdiff_t row = 0; row < rows; ++row) {
        int k = pointers[row], end = pointers[row + 1];
        P sum = Vec::zero();
        for (; end - k >= lanes; k += lanes) {
            R gathered[Vec::size];
            for (int c = 0; c < lanes; ++c) {
                const T value = rhs[indices[k + c]];
                gathered[2 * c] = value.real();
                gathered[2 * c + 1] = value.imag();
            }
            sum = Vec::cmadd(Vec::load(reinterpret_cast<const R *>(values + k)),
                             Vec::load(gathered), sum);
        }
        R partial[Vec::size];
        Vec::store(partial, sum);
        T total(0);
        for (int c = 0; c < lanes; ++c)
            total += T(partial[2 * c], partial[2 * c + 1]);
        for (; k < end; ++k)
            total += values[k] * rhs[indices[k]];
        if (!std::isfinite(total.real()) || !std::isfinite(total.imag()))
            return false;
        out[row] = total;
    }
    return true;
}
template <bool RowMajor, class T>
SPPY_VECTOR_TARGET void sparse_block_run(std::ptrdiff_t rows, std::ptrdiff_t cols, const T *values,
                                         const int *indices, const int *pointers, const T *rhs,
                                         T *out) {
    using R = typename Real<T>::type;
    using Vec = V<R>;
    using P = typename Vec::P;
    constexpr int lanes = Vec::size / 2, blocks = 8 / lanes;
    if constexpr (RowMajor) {
        for (std::ptrdiff_t row = 0; row < rows; ++row) {
            P sums[blocks];
            for (int c = 0; c < blocks; ++c)
                sums[c] = Vec::zero();
            for (int k = pointers[row]; k < pointers[row + 1]; ++k) {
                P value = Vec::set(values[k]);
                for (int c = 0; c < blocks; ++c)
                    sums[c] = Vec::cmadd(value,
                                         Vec::load(reinterpret_cast<const R *>(
                                             rhs + std::ptrdiff_t(indices[k]) * 8 + c * lanes)),
                                         sums[c]);
            }
            for (int c = 0; c < blocks; ++c)
                Vec::store(reinterpret_cast<R *>(out + row * 8 + c * lanes), sums[c]);
        }
    } else {
        std::fill_n(out, rows * 8, T(0));
        for (std::ptrdiff_t col = 0; col < cols; ++col) {
            P input[blocks];
            for (int c = 0; c < blocks; ++c)
                input[c] = Vec::load(reinterpret_cast<const R *>(rhs + col * 8 + c * lanes));
            for (int k = pointers[col]; k < pointers[col + 1]; ++k) {
                P value = Vec::set(values[k]);
                T *destination = out + std::ptrdiff_t(indices[k]) * 8;
                for (int c = 0; c < blocks; ++c) {
                    R *dst = reinterpret_cast<R *>(destination + c * lanes);
                    Vec::store(dst, Vec::cmadd(value, input[c], Vec::load(dst)));
                }
            }
        }
    }
}

#endif
template <class T>
bool dispatch(std::ptrdiff_t rows, std::ptrdiff_t cols, std::ptrdiff_t depth, const T *a,
              std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb, T *out, std::ptrdiff_t ldc) {
#if SPPY_VECTOR_AVAILABLE
    if (panel_simd_available()) {
        panel_run(rows, cols, depth, a, lda, b, ldb, out, ldc);
        return true;
    }
#endif
    return false;
}

template <class T>
bool vector_dispatch(std::ptrdiff_t rows, const T *values, const int *indices, const int *pointers,
                     const T *rhs, T *out) {
#if SPPY_VECTOR_AVAILABLE
    if (panel_simd_available())
        return csr_vector_run(rows, values, indices, pointers, rhs, out);
#endif
    return false;
}
template <class T>
bool block_dispatch(std::ptrdiff_t rows, std::ptrdiff_t cols, bool row_major, const T *values,
                    const int *indices, const int *pointers, const T *rhs, T *out) {
#if SPPY_VECTOR_AVAILABLE
    if (panel_simd_available()) {
        if (row_major)
            sparse_block_run<true>(rows, cols, values, indices, pointers, rhs, out);
        else
            sparse_block_run<false>(rows, cols, values, indices, pointers, rhs, out);
        return true;
    }
#endif
    return false;
}
#define SPPY_COMPLEX_DEF(T)                                                                        \
    bool complex_csr_vector(std::ptrdiff_t rows, const T *v, const int *i, const int *p,           \
                            const T *rhs, T *out) {                                                \
        return vector_dispatch(rows, v, i, p, rhs, out);                                           \
    }                                                                                              \
    bool complex_sparse_block(std::ptrdiff_t rows, std::ptrdiff_t cols, bool row_major,            \
                              const T *v, const int *i, const int *p, const T *rhs, T *out) {      \
        return block_dispatch(rows, cols, row_major, v, i, p, rhs, out);                           \
    }
SPPY_COMPLEX_DEF(std::complex<float>)
SPPY_COMPLEX_DEF(std::complex<double>)
#undef SPPY_COMPLEX_DEF
#define DEF(T)                                                                                     \
    bool panel_gemm(std::ptrdiff_t r, std::ptrdiff_t c, std::ptrdiff_t k, const T *a,              \
                    std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb, T *out,                    \
                    std::ptrdiff_t ldc) {                                                          \
        return dispatch(r, c, k, a, lda, b, ldb, out, ldc);                                        \
    }
DEF(float) DEF(double) DEF(std::complex<float>) DEF(std::complex<double>)
#undef DEF
} // namespace sppy
