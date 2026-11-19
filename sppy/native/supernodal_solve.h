#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <type_traits>
namespace sppy {
// Solve an entire dense RHS block directly over Eigen 5's supernodal factor
// views. Contiguous real/imaginary rows expose SIMD across RHS columns without
// materializing CSR copies of L/U or delaying work until a warm solve.

template <int Columns, class Solver, class B>
auto super_solve_impl(const Solver &solver, const B &rhs) {
    using T = typename B::Scalar;
    using R = typename Eigen::NumTraits<T>::Real;
    using D = Eigen::Matrix<R, Eigen::Dynamic, Columns, Eigen::RowMajor>;
    using C = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    C permuted = solver.rowsPermutation() * rhs;
    D real = permuted.real(), imag;
    if constexpr (Eigen::NumTraits<T>::IsComplex)
        imag = permuted.imag();
    const auto view = solver.matrixU();
    const auto &block = view.m_mapL;
    const auto &upper = view.m_mapU;
    using Block = std::decay_t<decltype(block)>;
    using Upper = std::decay_t<decltype(upper)>;
    auto update = [&](int row, int column, T value) {
        Eigen::Block<D, 1, Columns> destination(real, row, 0, 1, real.cols()),
            source(real, column, 0, 1, real.cols());
        if constexpr (Eigen::NumTraits<T>::IsComplex) {
            Eigen::Block<D, 1, Columns> destination_imag(imag, row, 0, 1, imag.cols()),
                source_imag(imag, column, 0, 1, imag.cols());
            destination -= value.real() * source - value.imag() * source_imag;
            destination_imag -= value.real() * source_imag + value.imag() * source;
        } else
            destination -= value * source;
    };
    const int n = int(solver.rows());
    for (int col = 0; col < n; ++col)
        for (typename Block::InnerIterator it(block, col); it; ++it)
            if (it.row() > col)
                update(int(it.row()), col, it.value());
    for (int col = n - 1; col >= 0; --col) {
        const int super = block.colToSup()[col];
        const int first = block.supToCol()[super];
        const T diagonal = block.valuePtr()[block.colIndexPtr()[col] + col - first];
        if constexpr (Eigen::NumTraits<T>::IsComplex) {
            for (int k = 0; k < real.cols(); ++k) {
                T value = T(real(col, k), imag(col, k)) / diagonal;
                real(col, k) = value.real();
                imag(col, k) = value.imag();
            }
        } else {
            real.row(col) /= diagonal;
        }
        for (typename Block::InnerIterator it(block, col); it; ++it)
            if (it.row() < col)
                update(int(it.row()), col, it.value());
        for (typename Upper::InnerIterator it(upper, col); it; ++it)
            update(int(it.row()), col, it.value());
    }
    C result(rhs.rows(), rhs.cols());
    result.real() = real;
    if constexpr (Eigen::NumTraits<T>::IsComplex)
        result.imag() = imag;
    return C(solver.colsPermutation().inverse() * result);
}
template <class Solver, class B> auto super_solve(const Solver &solver, const B &rhs) {
    if (rhs.cols() == 8)
        return super_solve_impl<8>(solver, rhs);
    return super_solve_impl<Eigen::Dynamic>(solver, rhs);
}
} // namespace sppy
