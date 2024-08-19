

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#if defined(_MSC_VER)
#  pragma warning(disable: 4996) 
#endif

#include <Eigen/Cholesky>

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;



template <typename M> void reset_ref(M &x) {
for (int i = 0; i < x.rows(); i++) for (int j = 0; j < x.cols(); j++)
x(i, j) = 11 + 10*i + j;
}

Eigen::MatrixXd &get_cm() {
static Eigen::MatrixXd *x;
if (!x) {
x = new Eigen::MatrixXd(3, 3);
reset_ref(*x);
}
return *x;
}
MatrixXdR &get_rm() {
static MatrixXdR *x;
if (!x) {
x = new MatrixXdR(3, 3);
reset_ref(*x);
}
return *x;
}
void reset_refs() {
reset_ref(get_cm());
reset_ref(get_rm());
}

double get_elem(Eigen::Ref<const Eigen::MatrixXd> m) { return m(2, 1); };


template <typename MatrixArgType> Eigen::MatrixXd adjust_matrix(MatrixArgType m) {
Eigen::MatrixXd ret(m);
for (int c = 0; c < m.cols(); c++) for (int r = 0; r < m.rows(); r++)
ret(r, c) += 10*r + 100*c;
return ret;
}

struct CustomOperatorNew {
CustomOperatorNew() = default;

Eigen::Matrix4d a = Eigen::Matrix4d::Zero();
Eigen::Matrix4d b = Eigen::Matrix4d::Identity();

EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

TEST_SUBMODULE(eigen, m) {
using FixedMatrixR = Eigen::Matrix<float, 5, 6, Eigen::RowMajor>;
using FixedMatrixC = Eigen::Matrix<float, 5, 6>;
using DenseMatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DenseMatrixC = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using FourRowMatrixC = Eigen::Matrix<float, 4, Eigen::Dynamic>;
using FourColMatrixC = Eigen::Matrix<float, Eigen::Dynamic, 4>;
using FourRowMatrixR = Eigen::Matrix<float, 4, Eigen::Dynamic>;
using FourColMatrixR = Eigen::Matrix<float, Eigen::Dynamic, 4>;
using SparseMatrixR = Eigen::SparseMatrix<float, Eigen::RowMajor>;
using SparseMatrixC = Eigen::SparseMatrix<float>;

m.attr("have_eigen") = true;

m.def("double_col", [](const Eigen::VectorXf &x) -> Eigen::VectorXf { return 2.0f * x; });
m.def("double_row", [](const Eigen::RowVectorXf &x) -> Eigen::RowVectorXf { return 2.0f * x; });
m.def("double_complex", [](const Eigen::VectorXcf &x) -> Eigen::VectorXcf { return 2.0f * x; });
m.def("double_threec", [](py::EigenDRef<Eigen::Vector3f> x) { x *= 2; });
m.def("double_threer", [](py::EigenDRef<Eigen::RowVector3f> x) { x *= 2; });
m.def("double_mat_cm", [](Eigen::MatrixXf x) -> Eigen::MatrixXf { return 2.0f * x; });
m.def("double_mat_rm", [](DenseMatrixR x) -> DenseMatrixR { return 2.0f * x; });

m.def("cholesky1", [](Eigen::Ref<MatrixXdR> x) -> Eigen::MatrixXd { return x.llt().matrixL(); });
m.def("cholesky2", [](const Eigen::Ref<const MatrixXdR> &x) -> Eigen::MatrixXd { return x.llt().matrixL(); });
m.def("cholesky3", [](const Eigen::Ref<MatrixXdR> &x) -> Eigen::MatrixXd { return x.llt().matrixL(); });
m.def("cholesky4", [](Eigen::Ref<const MatrixXdR> x) -> Eigen::MatrixXd { return x.llt().matrixL(); });

auto add_rm = [](Eigen::Ref<MatrixXdR> x, int r, int c, double v) { x(r,c) += v; };
auto add_cm = [](Eigen::Ref<Eigen::MatrixXd> x, int r, int c, double v) { x(r,c) += v; };

m.def("add_rm", add_rm); 
m.def("add_cm", add_cm); 
m.def("add1", add_rm);
m.def("add1", add_cm);
m.def("add2", add_cm);
m.def("add2", add_rm);
m.def("add_any", [](py::EigenDRef<Eigen::MatrixXd> x, int r, int c, double v) { x(r,c) += v; });

m.def("get_cm_ref", []() { return Eigen::Ref<Eigen::MatrixXd>(get_cm()); });
m.def("get_rm_ref", []() { return Eigen::Ref<MatrixXdR>(get_rm()); });
m.def("get_cm_const_ref", []() { return Eigen::Ref<const Eigen::MatrixXd>(get_cm()); });
m.def("get_rm_const_ref", []() { return Eigen::Ref<const MatrixXdR>(get_rm()); });

m.def("reset_refs", reset_refs); 

m.def("incr_matrix", [](Eigen::Ref<Eigen::MatrixXd> m, double v) {
m += Eigen::MatrixXd::Constant(m.rows(), m.cols(), v);
return m;
}, py::return_value_policy::reference);

m.def("incr_matrix_any", [](py::EigenDRef<Eigen::MatrixXd> m, double v) {
m += Eigen::MatrixXd::Constant(m.rows(), m.cols(), v);
return m;
}, py::return_value_policy::reference);

m.def("even_rows", [](py::EigenDRef<Eigen::MatrixXd> m) {
return py::EigenDMap<Eigen::MatrixXd>(
m.data(), (m.rows() + 1) / 2, m.cols(),
py::EigenDStride(m.outerStride(), 2 * m.innerStride()));
}, py::return_value_policy::reference);

m.def("even_cols", [](py::EigenDRef<Eigen::MatrixXd> m) {
return py::EigenDMap<Eigen::MatrixXd>(
m.data(), m.rows(), (m.cols() + 1) / 2,
py::EigenDStride(2 * m.outerStride(), m.innerStride()));
}, py::return_value_policy::reference);

m.def("diagonal", [](const Eigen::Ref<const Eigen::MatrixXd> &x) { return x.diagonal(); });
m.def("diagonal_1", [](const Eigen::Ref<const Eigen::MatrixXd> &x) { return x.diagonal<1>(); });
m.def("diagonal_n", [](const Eigen::Ref<const Eigen::MatrixXd> &x, int index) { return x.diagonal(index); });

m.def("block", [](const Eigen::Ref<const Eigen::MatrixXd> &x, int start_row, int start_col, int block_rows, int block_cols) {
return x.block(start_row, start_col, block_rows, block_cols);
});

class ReturnTester {
Eigen::MatrixXd mat = create();
public:
ReturnTester() { print_created(this); }
~ReturnTester() { print_destroyed(this); }
static Eigen::MatrixXd create() { return Eigen::MatrixXd::Ones(10, 10); }
static const Eigen::MatrixXd createConst() { return Eigen::MatrixXd::Ones(10, 10); }
Eigen::MatrixXd &get() { return mat; }
Eigen::MatrixXd *getPtr() { return &mat; }
const Eigen::MatrixXd &view() { return mat; }
const Eigen::MatrixXd *viewPtr() { return &mat; }
Eigen::Ref<Eigen::MatrixXd> ref() { return mat; }
Eigen::Ref<const Eigen::MatrixXd> refConst() { return mat; }
Eigen::Block<Eigen::MatrixXd> block(int r, int c, int nrow, int ncol) { return mat.block(r, c, nrow, ncol); }
Eigen::Block<const Eigen::MatrixXd> blockConst(int r, int c, int nrow, int ncol) const { return mat.block(r, c, nrow, ncol); }
py::EigenDMap<Eigen::Matrix2d> corners() { return py::EigenDMap<Eigen::Matrix2d>(mat.data(),
py::EigenDStride(mat.outerStride() * (mat.outerSize()-1), mat.innerStride() * (mat.innerSize()-1))); }
py::EigenDMap<const Eigen::Matrix2d> cornersConst() const { return py::EigenDMap<const Eigen::Matrix2d>(mat.data(),
py::EigenDStride(mat.outerStride() * (mat.outerSize()-1), mat.innerStride() * (mat.innerSize()-1))); }
};
using rvp = py::return_value_policy;
py::class_<ReturnTester>(m, "ReturnTester")
.def(py::init<>())
.def_static("create", &ReturnTester::create)
.def_static("create_const", &ReturnTester::createConst)
.def("get", &ReturnTester::get, rvp::reference_internal)
.def("get_ptr", &ReturnTester::getPtr, rvp::reference_internal)
.def("view", &ReturnTester::view, rvp::reference_internal)
.def("view_ptr", &ReturnTester::view, rvp::reference_internal)
.def("copy_get", &ReturnTester::get)   
.def("copy_view", &ReturnTester::view) 
.def("ref", &ReturnTester::ref) 
.def("ref_const", &ReturnTester::refConst) 
.def("ref_safe", &ReturnTester::ref, rvp::reference_internal)
.def("ref_const_safe", &ReturnTester::refConst, rvp::reference_internal)
.def("copy_ref", &ReturnTester::ref, rvp::copy)
.def("copy_ref_const", &ReturnTester::refConst, rvp::copy)
.def("block", &ReturnTester::block)
.def("block_safe", &ReturnTester::block, rvp::reference_internal)
.def("block_const", &ReturnTester::blockConst, rvp::reference_internal)
.def("copy_block", &ReturnTester::block, rvp::copy)
.def("corners", &ReturnTester::corners, rvp::reference_internal)
.def("corners_const", &ReturnTester::cornersConst, rvp::reference_internal)
;

m.def("incr_diag", [](int k) {
Eigen::DiagonalMatrix<int, Eigen::Dynamic> m(k);
for (int i = 0; i < k; i++) m.diagonal()[i] = i+1;
return m;
});

m.def("symmetric_lower", [](const Eigen::MatrixXi &m) {
return m.selfadjointView<Eigen::Lower>();
});
m.def("symmetric_upper", [](const Eigen::MatrixXi &m) {
return m.selfadjointView<Eigen::Upper>();
});

Eigen::MatrixXf mat(5, 6);
mat << 0,  3,  0,  0,  0, 11,
22, 0,  0,  0, 17, 11,
7,  5,  0,  1,  0, 11,
0,  0,  0,  0,  0, 11,
0,  0, 14,  0,  8, 11;

m.def("fixed_r", [mat]() -> FixedMatrixR { return FixedMatrixR(mat); });
m.def("fixed_r_const", [mat]() -> const FixedMatrixR { return FixedMatrixR(mat); });
m.def("fixed_c", [mat]() -> FixedMatrixC { return FixedMatrixC(mat); });
m.def("fixed_copy_r", [](const FixedMatrixR &m) -> FixedMatrixR { return m; });
m.def("fixed_copy_c", [](const FixedMatrixC &m) -> FixedMatrixC { return m; });
m.def("fixed_mutator_r", [](Eigen::Ref<FixedMatrixR>) {});
m.def("fixed_mutator_c", [](Eigen::Ref<FixedMatrixC>) {});
m.def("fixed_mutator_a", [](py::EigenDRef<FixedMatrixC>) {});
m.def("dense_r", [mat]() -> DenseMatrixR { return DenseMatrixR(mat); });
m.def("dense_c", [mat]() -> DenseMatrixC { return DenseMatrixC(mat); });
m.def("dense_copy_r", [](const DenseMatrixR &m) -> DenseMatrixR { return m; });
m.def("dense_copy_c", [](const DenseMatrixC &m) -> DenseMatrixC { return m; });
m.def("sparse_r", [mat]() -> SparseMatrixR { return Eigen::SparseView<Eigen::MatrixXf>(mat); });
m.def("sparse_c", [mat]() -> SparseMatrixC { return Eigen::SparseView<Eigen::MatrixXf>(mat); });
m.def("sparse_copy_r", [](const SparseMatrixR &m) -> SparseMatrixR { return m; });
m.def("sparse_copy_c", [](const SparseMatrixC &m) -> SparseMatrixC { return m; });
m.def("partial_copy_four_rm_r", [](const FourRowMatrixR &m) -> FourRowMatrixR { return m; });
m.def("partial_copy_four_rm_c", [](const FourColMatrixR &m) -> FourColMatrixR { return m; });
m.def("partial_copy_four_cm_r", [](const FourRowMatrixC &m) -> FourRowMatrixC { return m; });
m.def("partial_copy_four_cm_c", [](const FourColMatrixC &m) -> FourColMatrixC { return m; });

m.def("cpp_copy", [](py::handle m) { return m.cast<Eigen::MatrixXd>()(1, 0); });
m.def("cpp_ref_c", [](py::handle m) { return m.cast<Eigen::Ref<Eigen::MatrixXd>>()(1, 0); });
m.def("cpp_ref_r", [](py::handle m) { return m.cast<Eigen::Ref<MatrixXdR>>()(1, 0); });
m.def("cpp_ref_any", [](py::handle m) { return m.cast<py::EigenDRef<Eigen::MatrixXd>>()(1, 0); });


m.def("get_elem", &get_elem);
m.def("get_elem_nocopy", [](Eigen::Ref<const Eigen::MatrixXd> m) -> double { return get_elem(m); },
py::arg().noconvert());
m.def("get_elem_rm_nocopy", [](Eigen::Ref<const Eigen::Matrix<long, -1, -1, Eigen::RowMajor>> &m) -> long { return m(2, 1); },
py::arg().noconvert());

m.def("iss738_f1", &adjust_matrix<const Eigen::Ref<const Eigen::MatrixXd> &>, py::arg().noconvert());
m.def("iss738_f2", &adjust_matrix<const Eigen::Ref<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> &>, py::arg().noconvert());

m.def("iss1105_col", [](Eigen::VectorXd) { return true; });
m.def("iss1105_row", [](Eigen::RowVectorXd) { return true; });

m.def("matrix_multiply", [](const py::EigenDRef<const Eigen::MatrixXd> A, const py::EigenDRef<const Eigen::MatrixXd> B)
-> Eigen::MatrixXd {
if (A.cols() != B.rows()) throw std::domain_error("Nonconformable matrices!");
return A * B;
}, py::arg("A"), py::arg("B"));

py::class_<CustomOperatorNew>(m, "CustomOperatorNew")
.def(py::init<>())
.def_readonly("a", &CustomOperatorNew::a)
.def_readonly("b", &CustomOperatorNew::b);

m.def("get_elem_direct", [](Eigen::Ref<const Eigen::VectorXd> v) {
py::module::import("numpy").attr("ones")(10);
return v(5);
});
m.def("get_elem_indirect", [](std::vector<Eigen::Ref<const Eigen::VectorXd>> v) {
py::module::import("numpy").attr("ones")(10);
return v[0](5);
});
}
