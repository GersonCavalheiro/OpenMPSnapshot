#include "ceil.h"
#include <cmath>

template < typename DerivedX, typename DerivedY>
IGL_INLINE void igl::ceil(
const Eigen::PlainObjectBase<DerivedX>& X,
Eigen::PlainObjectBase<DerivedY>& Y)
{
using namespace std;
typedef typename DerivedX::Scalar Scalar;
Y = X.unaryExpr([](const Scalar &x)->Scalar{return std::ceil(x);}).template cast<typename DerivedY::Scalar >();
}

#ifdef IGL_STATIC_LIBRARY
template void igl::ceil<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#endif
