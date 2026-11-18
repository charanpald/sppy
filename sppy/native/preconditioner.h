#pragma once
#include "common.h"
#include <Eigen/IterativeLinearSolvers>
#include <functional>
namespace sppy {
template <class T> class Preconditioner {
    std::function<ColDense<T>(const ColDense<T> &)> apply;
    int dimension;
    template <class P> void create(const Sparse<T> &a, double drop, int fill, double shift) {
        auto p = std::make_shared<P>();
        if constexpr (std::is_same_v<P, Eigen::IncompleteLUT<T>>) {
            p->setDroptol(drop);
            p->setFillfactor(fill);
        }
        if constexpr (std::is_same_v<P, Eigen::IncompleteCholesky<T>>)
            p->setInitialShift(shift);
        {
            py::gil_scoped_release release;
            p->compute(a);
        }
        if (p->info() != Eigen::Success)
            throw std::runtime_error("Preconditioner computation failed");
        apply = [p](const ColDense<T> &b) -> ColDense<T> { return p->solve(b); };
    }

  public:
    Preconditioner(int r, int c, Array<int> ri, Array<int> ci, Array<T> values, std::string method,
                   double drop, int fill, double shift)
        : dimension(c) {
        auto a = sparse_copy<T>(r, c, ri, ci, values);
        require(r > 0 && c > 0, "Preconditioners require nonempty matrices");
        require(std::isfinite(drop) && drop >= 0 && fill > 0 && std::isfinite(shift) && shift >= 0,
                "Invalid preconditioner options");
        for (int i = 0; i < a.nonZeros(); ++i)
            require(Eigen::numext::isfinite(a.valuePtr()[i]), "Matrix contains nonfinite values");
        require(method == "least_square_diagonal" || r == c,
                "This preconditioner requires a square matrix");
        if (method == "identity")
            create<Eigen::IdentityPreconditioner>(a, drop, fill, shift);
        else if (method == "diagonal")
            create<Eigen::DiagonalPreconditioner<T>>(a, drop, fill, shift);
        else if (method == "least_square_diagonal")
            create<Eigen::LeastSquareDiagonalPreconditioner<T>>(a, drop, fill, shift);
        else if (method == "incomplete_lut")
            create<Eigen::IncompleteLUT<T>>(a, drop, fill, shift);
        else if (method == "incomplete_cholesky")
            create<Eigen::IncompleteCholesky<T>>(a, drop, fill, shift);
        else
            throw py::value_error("Unknown preconditioner");
    }
    auto solve(Array<T> rhs) const {
        auto b = dense_copy<T>(rhs);
        require(b.rows() == dimension, "Preconditioner RHS shape mismatch");
        require(b.allFinite(), "Right-hand side must be finite");
        ColDense<T> x;
        {
            py::gil_scoped_release release;
            x = apply(b);
        }
        return array_copy(x);
    }
};
template <class T> void bind_preconditioner(py::module_ &m, const char *name) {
    py::class_<Preconditioner<T>>(m, (std::string("Preconditioner_") + name).c_str())
        .def(py::init<int, int, Array<int>, Array<int>, Array<T>, std::string, double, int,
                      double>())
        .def("solve", &Preconditioner<T>::solve);
}
} // namespace sppy
