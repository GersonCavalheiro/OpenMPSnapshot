#ifndef __DENSE_CHOLESKY_H__
#define __DENSE_CHOLESKY_H__

#include "../matrix-formats/matrix-formats.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
template <int ver,typename T=void> typename std::enable_if<ver==0,void>::type cholesky(MatrixHandler<T,size_t>& R) {
for(size_t k=0;k<R.n();++k) {
for(size_t j=k+1;j<R.n();++j)
for(size_t i=j;i<R.n();++i)
R(j,i)-=R(k,i)*(R(k,j))/R(k,k);
T tmp=__sqrt__(__abs__(R(k,k)));
for(size_t j=k;j<R.n();++j)
R(k,j)=R(k,j)/tmp;
}
for(size_t i=0;i<R.n();++i)
for(size_t j=0;j<R.n();++j)
if(i>j)
R(i,j)=0;
}
template <int ver,typename T=void> typename std::enable_if<ver==1,void>::type cholesky(MatrixHandler<T,size_t>& R) {
for(size_t k=0;k<R.n();++k) {
#pragma omp parallel for num_threads(4)
for(size_t j=k+1;j<R.n();++j)
for(size_t i=j;i<R.n();++i)
R(j,i)-=R(k,i)*R(k,j)/R(k,k);
T tmp=__sqrt__(__abs__(R(k,k)));
#pragma omp parallel for num_threads(4)
for(size_t j=k;j<R.n();++j)
R(k,j)=R(k,j)/tmp;
}
for(size_t i=0;i<R.n();++i)
for(size_t j=0;j<R.n();++j)
if(i>j)
R(i,j)=0;
}
}
}
}
#endif
