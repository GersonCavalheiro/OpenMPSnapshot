#include "gaussian_curvature.h"
#include "internal_angles.h"
#include "PI.h"
#include <iostream>
template <typename DerivedV, typename DerivedF, typename DerivedK>
IGL_INLINE void igl::gaussian_curvature(
const Eigen::PlainObjectBase<DerivedV>& V,
const Eigen::PlainObjectBase<DerivedF>& F,
Eigen::PlainObjectBase<DerivedK> & K)
{
using namespace Eigen;
using namespace std;
Matrix<
typename DerivedV::Scalar,
DerivedF::RowsAtCompileTime,
DerivedF::ColsAtCompileTime> A;
internal_angles(V,F,A);
K.resize(V.rows(),1);
K.setConstant(V.rows(),1,2.*PI);
assert(A.rows() == F.rows());
assert(A.cols() == F.cols());
assert(K.rows() == V.rows());
assert(F.maxCoeff() < V.rows());
assert(K.cols() == 1);
const int Frows = F.rows();
for(int f = 0;f<Frows;f++)
{
for(int j = 0; j < 3;j++)
{
K(F(f,j),0) -=  A(f,j);
}
}
}

#ifdef IGL_STATIC_LIBRARY
template void igl::gaussian_curvature<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::gaussian_curvature<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#endif
