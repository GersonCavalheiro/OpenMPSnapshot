#include "cumsum.h"
#include <numeric>
#include <iostream>

template <typename DerivedX, typename DerivedY>
IGL_INLINE void igl::cumsum(
const Eigen::MatrixBase<DerivedX > & X,
const int dim,
Eigen::PlainObjectBase<DerivedY > & Y)
{
using namespace Eigen;
using namespace std;
Y.resizeLike(X);
int num_outer = (dim == 1 ? X.cols() : X.rows() );
int num_inner = (dim == 1 ? X.rows() : X.cols() );
if(dim == 1)
{
#pragma omp parallel for
for(int o = 0;o<num_outer;o++)
{
typename DerivedX::Scalar sum = 0;
for(int i = 0;i<num_inner;i++)
{
sum += X(i,o);
Y(i,o) = sum;
}
}
}else
{
for(int i = 0;i<num_inner;i++)
{
for(int o = 0;o<num_outer;o++)
{
if(i == 0)
{
Y(o,i) = X(o,i);
}else
{
Y(o,i) = Y(o,i-1) + X(o,i);
}
}
}
}
}

#ifdef IGL_STATIC_LIBRARY
template void igl::cumsum<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::cumsum<Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void igl::cumsum<Eigen::Matrix<double, 3, 1, 0, 3, 1>, Eigen::Matrix<double, 3, 1, 0, 3, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, 3, 1, 0, 3, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, 3, 1, 0, 3, 1> >&);
template void igl::cumsum<Eigen::Matrix<unsigned long, 2, 1, 0, 2, 1>, Eigen::Matrix<unsigned long, 2, 1, 0, 2, 1> >(Eigen::MatrixBase<Eigen::Matrix<unsigned long, 2, 1, 0, 2, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<unsigned long, 2, 1, 0, 2, 1> >&);
template void igl::cumsum<Eigen::Matrix<unsigned long, -1, 1, 0, -1, 1>, Eigen::Matrix<unsigned long, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<unsigned long, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<unsigned long, -1, 1, 0, -1, 1> >&);
template void igl::cumsum<Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
#ifdef WIN32
template void igl::cumsum<class Eigen::Matrix<unsigned __int64, -1, 1, 0, -1, 1>, class Eigen::Matrix<unsigned __int64, -1, 1, 0, -1, 1>>(class Eigen::MatrixBase<class Eigen::Matrix<unsigned __int64, -1, 1, 0, -1, 1>> const &, int, class Eigen::PlainObjectBase<class Eigen::Matrix<unsigned __int64, -1, 1, 0, -1, 1>> &);
template void igl::cumsum<class Eigen::Matrix<unsigned __int64, 2, 1, 0, 2, 1>, class Eigen::Matrix<unsigned __int64, 2, 1, 0, 2, 1>>(class Eigen::MatrixBase<class Eigen::Matrix<unsigned __int64, 2, 1, 0, 2, 1>> const &, int, class Eigen::PlainObjectBase<class Eigen::Matrix<unsigned __int64, 2, 1, 0, 2, 1>> &);
#endif
#endif
