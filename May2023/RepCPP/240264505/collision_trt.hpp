#ifndef LBT_COLLISION_TRT
#define LBT_COLLISION_TRT



#include <algorithm>
#include <array>
#include <cmath>
#if __has_include (<omp.h>)
#include <omp.h>
#endif

#include "../../continuum/continuum.hpp"
#include "../../general/memory_alignment.hpp"
#include "../population.hpp"
#include "collision.hpp"



template <unsigned int NX, unsigned int NY, unsigned int NZ, template <typename T> class LT, typename T, unsigned int NPOP>
class TRT: public CollisionOperator<NX,NY,NZ,LT,T,NPOP,TRT<NX,NY,NZ,LT,T,NPOP>>
{
using CO = CollisionOperator<NX,NY,NZ,LT,T,NPOP,TRT<NX,NY,NZ,LT,T,NPOP>>;

public:
TRT() = delete;
TRT& operator = (TRT&) = delete;
TRT(TRT&&) = delete;
TRT& operator = (TRT&&) = delete;
TRT(TRT const&) = delete;


TRT(std::shared_ptr<Population<NX,NY,NZ,LT,T,NPOP>> population, std::shared_ptr<Continuum<NX,NY,NZ,T>> continuum,
T const Re, T const U, unsigned int const L, unsigned int const p = 0, T const lambda = 0.25) noexcept:
CO(population, continuum, p), 
nu_(U*static_cast<T>(L) / Re),
lambda_(lambda),
tau_p_(nu_/(LT<T>::CS*LT<T>::CS) + 1.0/ 2.0), omega_p_(1.0/tau_p_),
omega_m_((tau_p_ - 1.0/2.0) / (lambda_ + 1.0/2.0*( tau_p_ - 1.0/2.0)))
{
return;
}


template<timestep TS>
void implementation(bool const isSave) noexcept;

protected:
T const nu_;
T const lambda_;
T const tau_p_;
T const omega_p_;
T const omega_m_;
};


template <unsigned int NX, unsigned int NY, unsigned int NZ, template <typename T> class LT, typename T, unsigned int NPOP> template<timestep TS>
void TRT<NX,NY,NZ,LT,T,NPOP>::implementation(bool const isSave) noexcept
{
#pragma omp parallel for
for(std::int64_t block = 0; block < CO::NUM_BLOCKS_; ++block)
{
unsigned int const z_start = CO::BLOCK_SIZE_ * (block / (CO::NUM_BLOCKS_X_*CO::NUM_BLOCKS_Y_));
unsigned int const   z_end = std::min(z_start + CO::BLOCK_SIZE_, NZ);

for(unsigned int z = z_start; z < z_end; ++z)
{
std::array<unsigned int,3> const z_n = { (NZ + z - 1) % NZ, z, (z + 1) % NZ };

unsigned int const y_start = CO::BLOCK_SIZE_*((block % (CO::NUM_BLOCKS_X_*CO::NUM_BLOCKS_Y_)) / CO::NUM_BLOCKS_X_);
unsigned int const   y_end = std::min(y_start + CO::BLOCK_SIZE_, NY);

for(unsigned int y = y_start; y < y_end; ++y)
{
std::array<unsigned int,3> const y_n = { (NY + y - 1) % NY, y, (y + 1) % NY };

unsigned int const x_start = CO::BLOCK_SIZE_*(block % CO::NUM_BLOCKS_X_);
unsigned int const   x_end = std::min(x_start + CO::BLOCK_SIZE_, NX);

for(unsigned int x = x_start; x < x_end; ++x)
{
std::array<unsigned int,3> const x_n = { (NX + x - 1) % NX, x, (x + 1) % NX };

alignas(CACHE_LINE) T f[LT<T>::ND] = {0.0};

#pragma GCC unroll (2)
for(unsigned int n = 0; n <= 1; ++n)
{
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED)
#else
#pragma GCC unroll (16)
#endif
for(unsigned int d = 0; d < LT<T>::HSPEED; ++d)
{
unsigned int const curr = n*LT<T>::OFF + d;
f[curr] = LT<T>::MASK[curr]*CO::population_->A[CO::population_->template indexRead<TS>(x_n,y_n,z_n,n,d,CO::p_)];
}
}

T rho = 0.0;
T u   = 0.0;
T v   = 0.0;
T w   = 0.0;
#pragma GCC unroll (2)
for(unsigned int n = 0; n <= 1; ++n)
{
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED)
#else
#pragma GCC unroll (16)
#endif
for(unsigned int d = 0; d < LT<T>::HSPEED; ++d)
{
unsigned int const curr = n*LT<T>::OFF + d;
rho += f[curr];
u   += f[curr]*LT<T>::DX[curr];
v   += f[curr]*LT<T>::DY[curr];
w   += f[curr]*LT<T>::DZ[curr];
}
}
u /= rho;
v /= rho;
w /= rho;

if (isSave == true)
{
CO::continuum_->operator()(x, y, z, 0) = rho;
CO::continuum_->operator()(x, y, z, 1) = u;
CO::continuum_->operator()(x, y, z, 2) = v;
CO::continuum_->operator()(x, y, z, 3) = w;
}

alignas(CACHE_LINE) T feq[LT<T>::ND] = {0.0};

T const uu = - 1.0/(2.0*LT<T>::CS*LT<T>::CS)*(u*u + v*v + w*w);

#pragma GCC unroll (2)
for(unsigned int n = 0; n <= 1; ++n)
{
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED-1)
#else
#pragma GCC unroll (15)
#endif
for(unsigned int d = 0; d < LT<T>::HSPEED; ++d)
{
unsigned int const curr = n*LT<T>::OFF + d;
T const cu = 1.0/(LT<T>::CS*LT<T>::CS)*(u*LT<T>::DX[curr] + v*LT<T>::DY[curr] + w*LT<T>::DZ[curr]);
feq[curr] = LT<T>::W[curr]*(rho + rho*(cu*(1.0 + 0.5*cu) + uu));
}
}

alignas(CACHE_LINE) T fp[LT<T>::OFF] = {0.0};
alignas(CACHE_LINE) T fm[LT<T>::OFF] = {0.0};

#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED-1)
#else
#pragma GCC unroll (15)
#endif
for(unsigned int d = 1; d < LT<T>::HSPEED; ++d)
{
fp[d] = 0.5*(f[d] + f[LT<T>::OFF + d] - (feq[d] + feq[LT<T>::OFF + d]));
fm[d] = 0.5*(f[d] - f[LT<T>::OFF + d] - (feq[d] - feq[LT<T>::OFF + d]));
}

CO::population_->A[CO::population_->template indexWrite<TS>(x_n,y_n,z_n,0,0,CO::p_)] = f[0] + omega_p_*(feq[0] - f[0]);
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED-1)
#else
#pragma GCC unroll (15)
#endif
for(unsigned int d = 1; d < LT<T>::HSPEED; ++d)
{
CO::population_->A[CO::population_->template indexWrite<TS>(x_n,y_n,z_n,0,d,CO::p_)] = f[d] - omega_p_*fp[d] - omega_m_*fm[d];
}
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED-1)
#else
#pragma GCC unroll (15)
#endif
for(unsigned int d = 1; d < LT<T>::HSPEED; ++d)
{
CO::population_->A[CO::population_->template indexWrite<TS>(x_n,y_n,z_n,1,d,CO::p_)] = f[LT<T>::OFF + d] - omega_p_*fp[d] + omega_m_*fm[d];
}
}
}
}
}

return;
}

#endif 
