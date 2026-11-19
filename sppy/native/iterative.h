#pragma once
#include "matrix_storage.h"
#include <Eigen/IterativeLinearSolvers>
namespace sppy {
template <class T, class Solver, class S>
py::tuple run_iterative(const S &a, const ColDense<T> &b, const ColDense<T> &guess, int maxiter,
                        double tolerance, double drop, int fill, double shift) {
    Solver solver;
    solver.setMaxIterations(maxiter);
    solver.setTolerance(tolerance);
    using P = typename Solver::Preconditioner;
    if constexpr (std::is_same_v<P, Eigen::IncompleteLUT<T>>) {
        solver.preconditioner().setDroptol(drop);
        solver.preconditioner().setFillfactor(fill);
    }
    if constexpr (std::is_same_v<P, Eigen::IncompleteCholesky<T>>)
        solver.preconditioner().setInitialShift(shift);
    ColDense<T> x;
    int status;
    Eigen::Index iterations = 0;
    double error = 0;
    {
        py::gil_scoped_release release;
        solver.compute(a);
        status = solver.info();
        if (status == Eigen::Success) {
            x.resize(a.cols(), b.cols());
            for (int j = 0; j < b.cols(); ++j) {
                x.col(j) = solver.solveWithGuess(b.col(j), guess.col(j));
                status = std::max(status, int(solver.info()));
                iterations = std::max(iterations, solver.iterations());
                error = std::max(error, double(solver.error()));
            }
        }
    }
    if (status != Eigen::Success && x.size() == 0)
        throw std::runtime_error("Preconditioner computation failed");
    return py::make_tuple(array_copy(x), status, iterations, error);
}
template <class T, class S>
py::tuple iterate_storage(const S &a, Array<T> rhs, Array<T> initial, std::string method,
                          std::string preconditioner, int maxiter, double tolerance, double drop,
                          int fill, double shift) {
    const auto r = a.rows(), c = a.cols();
    auto b = dense_copy<T>(rhs);
    auto guess = dense_copy<T>(initial);
    require(r > 0 && c > 0 && b.rows() == r && guess.rows() == c && guess.cols() == b.cols(),
            "Invalid iterative system dimensions");
    require(maxiter > 0 && std::isfinite(tolerance) && tolerance > 0,
            "Invalid iteration limit or tolerance");
    require(std::isfinite(drop) && drop >= 0 && fill > 0 && std::isfinite(shift) && shift >= 0,
            "Invalid preconditioner options");
    for (int i = 0; i < a.nonZeros(); ++i)
        require(Eigen::numext::isfinite(a.valuePtr()[i]), "Matrix contains nonfinite values");
    require(b.allFinite() && guess.allFinite(), "Right-hand side and guess must be finite");
    if (method == "cg") {
        require(r == c, "CG requires a square matrix");
        if (preconditioner == "identity")
            return run_iterative<T, Eigen::ConjugateGradient<S, Eigen::Lower | Eigen::Upper,
                                                             Eigen::IdentityPreconditioner>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
        if (preconditioner == "diagonal")
            return run_iterative<T, Eigen::ConjugateGradient<S, Eigen::Lower | Eigen::Upper,
                                                             Eigen::DiagonalPreconditioner<T>>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
        if (preconditioner == "incomplete_cholesky")
            return run_iterative<T, Eigen::ConjugateGradient<S, Eigen::Lower | Eigen::Upper,
                                                             Eigen::IncompleteCholesky<T>>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
    } else if (method == "bicgstab") {
        require(r == c, "BiCGSTAB requires a square matrix");
        if (preconditioner == "identity")
            return run_iterative<T, Eigen::BiCGSTAB<S, Eigen::IdentityPreconditioner>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
        if (preconditioner == "diagonal")
            return run_iterative<T, Eigen::BiCGSTAB<S, Eigen::DiagonalPreconditioner<T>>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
        if (preconditioner == "incomplete_lut")
            return run_iterative<T, Eigen::BiCGSTAB<S, Eigen::IncompleteLUT<T>>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
    } else if (method == "lscg") {
        if (preconditioner == "identity")
            return run_iterative<
                T, Eigen::LeastSquaresConjugateGradient<S, Eigen::IdentityPreconditioner>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
        if (preconditioner == "least_square_diagonal")
            return run_iterative<T, Eigen::LeastSquaresConjugateGradient<
                                        S, Eigen::LeastSquareDiagonalPreconditioner<T>>>(
                a, b, guess, maxiter, tolerance, drop, fill, shift);
    }
    throw py::value_error("Unsupported solver/preconditioner combination");
}
template <class T>
py::tuple iterate(int r, int c, Array<int> ri, Array<int> ci, Array<T> values, Array<T> rhs,
                  Array<T> initial, std::string method, std::string preconditioner, int maxiter,
                  double tolerance, double drop, int fill, double shift) {
    return iterate_storage<T>(sparse_copy<T>(r, c, ri, ci, values), rhs, initial, method,
                              preconditioner, maxiter, tolerance, drop, fill, shift);
}
template <class T, class S>
auto triangular_solve_storage(const S &a, Array<T> rhs, bool lower, bool unit) {
    const auto r = a.rows(), c = a.cols();
    auto b = dense_copy<T>(rhs);
    require(r == c && r == b.rows(), "Invalid triangular system dimensions");
    require(b.allFinite(), "Right-hand side must be finite");
    for (int j = 0; j < a.nonZeros(); ++j)
        require(Eigen::numext::isfinite(a.valuePtr()[j]), "Matrix contains nonfinite values");
    if (!unit)
        for (int i = 0; i < r; ++i)
            require(a.coeff(i, i) != T(0), "Singular triangular diagonal");
    ColDense<T> x;
    {
        py::gil_scoped_release release;
        if (lower) {
            if (unit)
                x = a.template triangularView<Eigen::UnitLower>().solve(b);
            else
                x = a.template triangularView<Eigen::Lower>().solve(b);
        } else {
            if (unit)
                x = a.template triangularView<Eigen::UnitUpper>().solve(b);
            else
                x = a.template triangularView<Eigen::Upper>().solve(b);
        }
    }
    return array_copy(x);
}
template <class T> void bind_iterative(py::module_ &m, const char *name) {
    m.def((std::string("iterate_") + name).c_str(), &iterate<T>);
    m.def((std::string("iterate_matrix_") + name).c_str(),
          [](py::handle backend, Array<T> rhs, Array<T> initial, std::string method,
             std::string preconditioner, int maxiter, double tolerance, double drop, int fill,
             double shift) {
              return iterate_storage<T>(solver_matrix<T, Eigen::RowMajor>(backend), rhs, initial,
                                        method, preconditioner, maxiter, tolerance, drop, fill,
                                        shift);
          });
    m.def((std::string("triangular_solve_matrix_") + name).c_str(),
          [](py::handle backend, Array<T> rhs, bool lower, bool unit) {
              return triangular_solve_storage<T>(solver_matrix<T, Eigen::RowMajor>(backend), rhs,
                                                 lower, unit);
          });
    m.def((std::string("triangular_solve_") + name).c_str(),
          [](int r, int c, Array<int> ri, Array<int> ci, Array<T> values, Array<T> rhs, bool lower,
             bool unit) {
              return triangular_solve_storage<T>(sparse_copy<T>(r, c, ri, ci, values), rhs, lower,
                                                 unit);
          });
}
} // namespace sppy
