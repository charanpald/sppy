#pragma once
#include "panel_kernels.h"
#include <Eigen/Sparse>
// Thin, untransposed panels use runtime-selected kernels. Dynamic-stride
// dispatch retains Eigen's original implementation for unsupported CPUs,
// shapes, conjugation/scaling, and parallel GEMM protocols.
namespace Eigen {
namespace internal {
template <class Index, class T>
struct general_matrix_matrix_product<Index, T, ColMajor, false, T, ColMajor, false, ColMajor, 1> {
    using Traits = gebp_traits<T, T>;
    using ResScalar = T;
    static void run(Index rows, Index cols, Index depth, const T *lhs, Index lhsStride,
                    const T *rhs, Index rhsStride, T *res, Index resIncr, Index resStride, T alpha,
                    level3_blocking<T, T> &blocking, GemmParallelInfo<Index> *info = nullptr) {
        if (!info && resIncr == 1 && rows >= 4 && depth >= 4 && cols >= 2 && cols <= 32 &&
            rows <= 4096 && depth <= 4096 && rows * depth <= (1 << 20) && alpha == T(1) &&
            sppy::panel_gemm(rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res, resStride))
            return;
        general_matrix_matrix_product<Index, T, ColMajor, false, T, ColMajor, false, ColMajor,
                                      Dynamic>::run(rows, cols, depth, lhs, lhsStride, rhs,
                                                    rhsStride, res, resIncr, resStride, alpha,
                                                    blocking, info);
    }
};
} // namespace internal
} // namespace Eigen
