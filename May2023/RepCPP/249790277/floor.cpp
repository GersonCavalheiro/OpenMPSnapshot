#include "floor.h"
#include <cmath>
#include <iostream>

template < typename DerivedX, typename DerivedY>
IGL_INLINE void igl::floor(
const Eigen::PlainObjectBase<DerivedX>& X,
Eigen::PlainObjectBase<DerivedY>& Y)
{
using namespace std;
typedef typename DerivedX::Scalar Scalar;
Y = X.unaryExpr([](const Scalar &x)->Scalar{return std::floor(x);}).template cast<typename DerivedY::Scalar >();
}

#ifdef IGL_STATIC_LIBRARY
template void igl::floor<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::floor<Eigen::Array<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::PlainObjectBase<Eigen::Array<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
template void igl::floor<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
#endif
