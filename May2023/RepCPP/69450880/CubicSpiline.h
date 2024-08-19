



#ifndef CUBICSPILINE_H_
#define CUBICSPILINE_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/external/hydra_thrust/copy.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/binary_search.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <math.h>
#include <algorithm>

namespace hydra {





template<size_t N, unsigned int ArgIndex=0>
class CubicSpiline: public BaseFunctor<CubicSpiline<N, ArgIndex>, GReal_t , 0>
{

public:


CubicSpiline() = default;

template<typename Iterator1, typename Iterator2>
CubicSpiline( Iterator1 xbegin, Iterator2 ybegin ):
BaseFunctor<CubicSpiline<N, ArgIndex>, GReal_t , 0>()
{
hydra_thrust::copy( ybegin, ybegin+N,  fD );
hydra_thrust::copy( xbegin, xbegin+N,  fX);

}

__hydra_host__ __hydra_device__
CubicSpiline(CubicSpiline<N, ArgIndex> const& other ):
BaseFunctor<CubicSpiline<N, ArgIndex>, double, 0>(other)
{
#pragma unroll
for(size_t i =0; i< N; i++){

fD[i] = other.GetD()[i];
fX[i] = other.GetX()[i];
}
}

__hydra_host__ __hydra_device__ inline
CubicSpiline<N>& operator=(CubicSpiline<N, ArgIndex> const& other )
{
if(this == &other) return *this;

BaseFunctor<CubicSpiline<N, ArgIndex>, double, 0>::operator=(other);

#pragma unroll
for(size_t i =0; i< N; i++){

fD[i] = other.GetD()[i];
fX[i] = other.GetX()[i];
}
return *this;
}

__hydra_host__ __hydra_device__
inline const GReal_t* GetD() const {
return fD;
}

__hydra_host__ __hydra_device__
inline void SetD(unsigned int i, GReal_t value)  {
fD[i]=value;
}

__hydra_host__ __hydra_device__
inline const GReal_t* GetX() const {
return fX;
}

__hydra_host__ __hydra_device__
inline void SetX(unsigned int i, GReal_t value)  {
fX[i]=value;
}

template<typename T>
__hydra_host__ __hydra_device__
inline double Evaluate(unsigned int n, T*x)  const {

GReal_t X  = x[ArgIndex];

GReal_t r = X<=fX[0]?fD[0]: X>=fX[N-1] ? fD[N-1] :spiline( X);

return  CHECK_VALUE( r, "r=%f",r) ;
}

template<typename T>
__hydra_host__ __hydra_device__
inline double Evaluate(T x)  const {

GReal_t X  = hydra::get<ArgIndex>(x); 

GReal_t r = X<=fX[0]?fD[0]: X>=fX[N-1] ? fD[N-1] :spiline(X);

return  CHECK_VALUE( r, "r=%f",r) ;
}

private:

__hydra_host__ __hydra_device__
inline double spiline( const double x) const
{
using hydra_thrust::min;

const size_t i = hydra_thrust::distance(fX,
hydra_thrust::lower_bound(hydra_thrust::seq, fX, fX +N, x));

const double y_i = fD[i], y_ip = fD[i+1],y_ipp = fD[i+2], y_im = fD[i-1] ;

const double x_i = fX[i], x_ip = fX[i+1],x_ipp = fX[i+2], x_im = fX[i-1] ;

const double  h_i  = x_ip -x_i;
const double  h_ip = x_ipp -x_ip;
const double  h_im = x_i  -x_im;

const double  s_i  = (y_ip - y_i)/h_i;
const double  s_ip = (y_ipp - y_ip)/h_ip;
const double  s_im = (y_i - y_im)/h_im;

const double p_i  = i==0 ? ( s_i*(1 + h_i/(h_i + h_ip)) - s_ip*h_i/(h_i + h_ip) ):
i==N-2 ? ( s_i*(1 + h_i/(h_i + h_im)) - s_im*h_i/(h_i + h_im) )
: (s_im*h_i + s_i*h_im)/(h_i+ h_im);

const double p_ip = (s_i*h_ip + s_ip*h_i)/(h_ip+ h_i);



const double c_i =  i==0  ? (copysign(1.0, p_i ) + copysign(1.0, s_i ))
*min( fabs(s_i) , 0.5*fabs(p_i) ):
i==N-2 ? (copysign(1.0, p_i ) + copysign(1.0, s_i ))
*min( fabs(s_i) , 0.5*fabs(p_i) ):
(copysign(1.0, s_im ) + copysign(1.0, s_i ))
*min(min(fabs(s_im), fabs(s_i)), 0.5*fabs(p_i) );

const double c_ip =  (copysign(1.0, s_i ) + copysign(1.0, s_ip ))
*min(min(fabs(s_ip), fabs(s_i)), 0.5*fabs(p_ip) );

const double b_i =  (-2*c_i - c_ip - 3*s_i)/h_i;

const double a_i = (c_i + c_ip - 2*s_i)/(h_i*h_i);

const double X = (x-fX[i]);

return X*( X*(a_i*X + b_i) + c_i) + y_i;
}

GReal_t fX[N];
GReal_t fD[N];

};

}  



#endif 
