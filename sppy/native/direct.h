#pragma once
#include "matrix_storage.h"
#include "supernodal_solve.h"
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
namespace sppy {
// Narrow-band inputs have little dense panel work. Halve Eigen's panel
// workspace there while retaining its default blocking for general matrices.
template <class S, class Ordering> class PanelSparseLU : public Eigen::SparseLU<S, Ordering> {
    using Base = Eigen::SparseLU<S, Ordering>;

  public:
    void analyzePattern(const S &a) {
        bool narrow = a.rows() >= 256;
        for (int col = 0; narrow && col < a.cols(); ++col)
            for (typename S::InnerIterator it(a, col); it; ++it)
                if (std::abs(it.row() - it.col()) > 32) {
                    narrow = false;
                    break;
                }
        this->m_perfv.panel_size = narrow ? 8 : 16;
        Base::analyzePattern(a);
    }
};
template <class T> struct DirectBase {
    virtual ~DirectBase() = default;
    virtual void analyze(const Sparse<T> &) = 0;
    virtual void factorize(const Sparse<T> &) = 0;
    virtual int info() const = 0;
    virtual ColDense<T> solve(const ColDense<T> &) const = 0;
    virtual py::object factor(const std::string &) const = 0;
    virtual T determinant() const = 0;
    virtual int rank() const = 0;
    virtual Sparse<T> solve_sparse(const Sparse<T> &) const = 0;
    virtual py::tuple slogdet() const = 0;
    virtual ColDense<T> apply_q(const ColDense<T> &, bool) const = 0;
};
template <class T, class Solver, int Kind> struct DirectModel : DirectBase<T> {
    mutable Solver solver;
    DirectModel(double offset, double scale, double pivot) {
        if constexpr (Kind == 0 || Kind == 1)
            solver.setShift(offset, scale);
        if constexpr (Kind == 2 || Kind == 3)
            if (pivot >= 0)
                solver.setPivotThreshold(pivot);
    }
    void analyze(const Sparse<T> &a) override { solver.analyzePattern(a); }
    void factorize(const Sparse<T> &a) override { solver.factorize(a); }
    int info() const override { return solver.info(); }
    ColDense<T> solve(const ColDense<T> &b) const override {
        if constexpr (Kind == 2) {
            if (b.cols() >= 4)
                return super_solve(solver, b);
        }
        return solver.solve(b);
    }
    py::object factor(const std::string &name) const override {
        if constexpr (Kind == 0 || Kind == 1) {
            if (name == "L") {
                Sparse<T> a = solver.matrixL();
                return export_sparse(a);
            }
            if (name == "U") {
                Sparse<T> a = solver.matrixU();
                return export_sparse(a);
            }
            if (name == "P") {
                py::array_t<int> a(solver.permutationP().size());
                std::copy_n(solver.permutationP().indices().data(), a.size(), a.mutable_data());
                return a;
            }
            if constexpr (Kind == 1)
                if (name == "D")
                    return array_copy(solver.vectorD());
        }
        if constexpr (Kind == 2) {
            if (name == "L")
                return export_sparse(Sparse<T>(solver.matrixL().toSparse()));
            if (name == "U")
                return export_sparse(Sparse<T>(solver.matrixU().toSparse()));
            if (name == "P" || name == "Q") {
                auto p = name == "P" ? solver.rowsPermutation().indices().eval()
                                     : solver.colsPermutation().indices().eval();
                py::array_t<int> a(p.size());
                std::copy_n(p.data(), p.size(), a.mutable_data());
                return a;
            }
        }
        if constexpr (Kind == 3) {
            if (name == "R")
                return export_sparse(Sparse<T>(solver.matrixR()));
            if (name == "P") {
                auto p = solver.colsPermutation().indices().eval();
                py::array_t<int> a(p.size());
                std::copy_n(p.data(), p.size(), a.mutable_data());
                return a;
            }
        }
        throw py::value_error("Factor is unavailable for this decomposition");
    }
    T determinant() const override {
        if constexpr (Kind != 3)
            return solver.determinant();
        throw py::value_error("Determinant is unavailable for QR");
    }
    Sparse<T> solve_sparse(const Sparse<T> &b) const override { return solver.solve(b); }
    py::tuple slogdet() const override {
        T phase = T(1);
        double logabs = 0;
        if constexpr (Kind == 0) {
            Sparse<T> l = solver.matrixL();
            for (int i = 0; i < l.rows(); ++i)
                logabs += 2 * std::log(std::abs(l.coeff(i, i)));
        } else if constexpr (Kind == 1) {
            auto d = solver.vectorD();
            for (int i = 0; i < d.size(); ++i) {
                double magnitude = std::abs(d[i]);
                phase *= d[i] / T(magnitude);
                logabs += std::log(magnitude);
            }
        } else if constexpr (Kind == 2) {
            Sparse<T> u = solver.matrixU().toSparse();
            phase =
                T(solver.rowsPermutation().determinant() * solver.colsPermutation().determinant());
            for (int i = 0; i < u.rows(); ++i) {
                double magnitude = std::abs(u.coeff(i, i));
                phase *= u.coeff(i, i) / T(magnitude);
                logabs += std::log(magnitude);
            }
        } else
            throw py::value_error("Log determinant is unavailable for QR");
        return py::make_tuple(phase, logabs);
    }
    int rank() const override {
        if constexpr (Kind == 3)
            return solver.rank();
        throw py::value_error("Rank is only available for QR");
    }
    ColDense<T> apply_q(const ColDense<T> &b, bool adjoint) const override {
        if constexpr (Kind == 3) {
            if (adjoint)
                return solver.matrixQ().adjoint() * b;
            return solver.matrixQ() * b;
        }
        throw py::value_error("Q is only available for QR");
    }
};
template <class T> class Factor {
    std::unique_ptr<DirectBase<T>> solver;
    Sparse<T> pattern;
    bool analyzed = false, ready = false;
    std::string method, ordering;
    double offset, scale, pivot;
    template <class Ordering> void create() {
        if (method == "llt")
            solver = std::make_unique<
                DirectModel<T, Eigen::SimplicialLLT<Sparse<T>, Eigen::Lower, Ordering>, 0>>(
                offset, scale, pivot);
        else if (method == "ldlt")
            solver = std::make_unique<
                DirectModel<T, Eigen::SimplicialLDLT<Sparse<T>, Eigen::Lower, Ordering>, 1>>(
                offset, scale, pivot);
        else if (method == "lu")
            solver = std::make_unique<DirectModel<T, PanelSparseLU<Sparse<T>, Ordering>, 2>>(
                offset, scale, pivot);
        else if (method == "qr")
            solver = std::make_unique<DirectModel<T, Eigen::SparseQR<Sparse<T>, Ordering>, 3>>(
                offset, scale, pivot);
        else
            throw py::value_error("Unknown direct method");
    }
    void check_shape(const Sparse<T> &a) const {
        require(a.rows() > 0 && a.cols() > 0, "Factorization requires nonempty dimensions");
        require(ordering != "amd" || a.rows() == a.cols(),
                "AMD factorization ordering requires a square matrix");
        require(method == "qr" || a.rows() == a.cols(),
                "This factorization requires a square matrix");
        for (int i = 0; i < a.nonZeros(); ++i)
            require(Eigen::numext::isfinite(a.valuePtr()[i]), "Matrix contains nonfinite values");
    }

  public:
    Factor(std::string method, std::string ordering, double offset, double scale, double pivot)
        : method(std::move(method)), ordering(ordering), offset(offset), scale(scale),
          pivot(pivot) {
        if (ordering == "natural")
            create<Eigen::NaturalOrdering<int>>();
        else if (ordering == "amd")
            create<Eigen::AMDOrdering<int>>();
        else if (ordering == "colamd")
            create<Eigen::COLAMDOrdering<int>>();
        else
            throw py::value_error("Unknown ordering");
    }
    void analyze(int r, int c, Array<int> ri, Array<int> ci, Array<T> values) {
        analyze_storage(sparse_copy<T>(r, c, ri, ci, values));
    }
    void analyze_storage(Sparse<T> a) {
        check_shape(a);
        ready = false;
        analyzed = false;
        {
            py::gil_scoped_release release;
            solver->analyze(a);
        }
        pattern = std::move(a);
        analyzed = true;
    }
    int factorize(int r, int c, Array<int> ri, Array<int> ci, Array<T> values) {
        return factorize_storage(sparse_copy<T>(r, c, ri, ci, values));
    }
    int factorize_storage(const Sparse<T> &a) {
        check_shape(a);
        ready = false;
        require(analyzed, "Call analyze_pattern before factorize");
        require(a.rows() == pattern.rows() && a.cols() == pattern.cols() &&
                    a.nonZeros() == pattern.nonZeros(),
                "Sparsity pattern changed; analyze again");
        require(std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.outerSize() + 1,
                           pattern.outerIndexPtr()) &&
                    std::equal(a.innerIndexPtr(), a.innerIndexPtr() + a.nonZeros(),
                               pattern.innerIndexPtr()),
                "Sparsity pattern changed; analyze again");
        {
            py::gil_scoped_release release;
            solver->factorize(a);
        }
        ready = solver->info() == Eigen::Success;
        return solver->info();
    }
    void analyze_matrix(py::handle backend) { analyze_storage(solver_matrix<T>(backend)); }
    int factorize_matrix(py::handle backend) {
        return factorize_storage(solver_matrix<T>(backend));
    }
    int compute_matrix(py::handle backend) {
        auto a = solver_matrix<T>(backend);
        check_shape(a);
        ready = analyzed = false;
        {
            py::gil_scoped_release release;
            solver->analyze(a);
            solver->factorize(a);
        }
        pattern = std::move(a);
        analyzed = true;
        ready = solver->info() == Eigen::Success;
        return solver->info();
    }
    auto solve(Array<T> values) const {
        require(ready, "Factorization is not successful");
        auto b = dense_copy<T>(values);
        require(b.rows() == pattern.rows(), "Right-hand side shape mismatch");
        ColDense<T> x;
        {
            py::gil_scoped_release release;
            x = solver->solve(b);
        }
        if (!x.allFinite())
            throw std::runtime_error("Solve produced nonfinite values");
        return array_copy(x);
    }
    py::tuple solve_sparse(int r, int c, Array<int> ri, Array<int> ci, Array<T> values) const {
        require(ready, "Factorization is not successful");
        auto b = sparse_copy<T>(r, c, ri, ci, values);
        require(r == pattern.rows(), "Right-hand side shape mismatch");
        Sparse<T> x;
        {
            py::gil_scoped_release release;
            x = solver->solve_sparse(b);
        }
        return export_sparse(x);
    }
    py::tuple slogdet() const {
        require(ready, "Factorization is not successful");
        return solver->slogdet();
    }
    py::object factor(std::string name) const {
        require(ready, "Factorization is not successful");
        return solver->factor(name);
    }
    T determinant() const {
        require(ready, "Factorization is not successful");
        return solver->determinant();
    }
    int rank() const {
        require(ready, "Factorization is not successful");
        return solver->rank();
    }
    auto apply_q(Array<T> values, bool adjoint) const {
        require(ready, "Factorization is not successful");
        auto b = dense_copy<T>(values);
        require(b.rows() == pattern.rows(), "Q operand shape mismatch");
        return array_copy(solver->apply_q(b, adjoint));
    }
};
template <class T> void bind_direct(py::module_ &m, const char *name) {
    using F = Factor<T>;
    py::class_<F>(m, (std::string("Factor_") + name).c_str())
        .def(py::init<std::string, std::string, double, double, double>())
        .def("analyze", &F::analyze)
        .def("analyze_matrix", &F::analyze_matrix)
        .def("factorize_matrix", &F::factorize_matrix)
        .def("compute_matrix", &F::compute_matrix)
        .def("factorize", &F::factorize)
        .def("solve", &F::solve)
        .def("solve_sparse", &F::solve_sparse)
        .def("slogdet", &F::slogdet)
        .def("factor", &F::factor)
        .def("determinant", &F::determinant)
        .def_property_readonly("rank", &F::rank)
        .def("apply_q", &F::apply_q);
    m.def((std::string("ordering_") + name).c_str(),
          [](int r, int c, Array<int> ri, Array<int> ci, Array<T> v, std::string method) {
              auto a = sparse_copy<T>(r, c, ri, ci, v);
              Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> p;
              if (method == "amd") {
                  require(r == c, "AMD requires a square matrix");
                  Eigen::AMDOrdering<int>{}(a, p);
              } else if (method == "colamd")
                  Eigen::COLAMDOrdering<int>{}(a, p);
              else if (method == "natural") {
                  p.resize(c);
                  p.setIdentity();
              } else
                  throw py::value_error("Unknown ordering");
              py::array_t<int> result(p.size());
              std::copy_n(p.indices().data(), p.size(), result.mutable_data());
              return result;
          });
}
} // namespace sppy
