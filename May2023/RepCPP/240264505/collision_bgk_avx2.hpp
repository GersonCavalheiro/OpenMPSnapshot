#ifndef LBT_COLLISION_BGK_AVX2
#define LBT_COLLISION_BGK_AVX2



#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <type_traits>
#if __has_include (<omp.h>)
#include <omp.h>
#endif

#include "../../continuum/continuum.hpp"
#include "../../general/memory_alignment.hpp"
#include "../population.hpp"
#include "collision.hpp"


#ifdef __AVX2__

#include <immintrin.h>

#define AVX2_SIZE          sizeof(__m256d)
#define AVX2_REG_SIZE      sizeof(__m256d)/sizeof(double)



static inline double __attribute__((always_inline)) _mm256_reduce_add_pd(__m256d const _a) noexcept
{
__m256d const _sum = _mm256_hadd_pd(_a, _a);

return ((double*)&_sum)[0] + ((double*)&_sum)[2];
}



template <unsigned int NX, unsigned int NY, unsigned int NZ, template <typename T> class LT, typename T, unsigned int NPOP>
class BGK_AVX2: public CollisionOperator<NX,NY,NZ,LT,T,NPOP,BGK_AVX2<NX,NY,NZ,LT,T,NPOP>>
{
using CO = CollisionOperator<NX,NY,NZ,LT,T,NPOP,BGK_AVX2<NX,NY,NZ,LT,T,NPOP>>;

public:
BGK_AVX2() = delete;
BGK_AVX2& operator = (BGK_AVX2&) = delete;
BGK_AVX2(BGK_AVX2&&) = delete;
BGK_AVX2& operator = (BGK_AVX2&&) = delete;
BGK_AVX2(BGK_AVX2 const&) = delete;


BGK_AVX2(std::shared_ptr<Population<NX,NY,NZ,LT,T,NPOP>> population, std::shared_ptr<Continuum<NX,NY,NZ,T>> continuum,
T const Re, T const U, unsigned int const L, unsigned int const p = 0) noexcept:
CO(population, continuum, p), 
nu_(U*static_cast<T>(L) / Re),
tau_(nu_/(LT<T>::CS*LT<T>::CS) + 1.0/ 2.0), omega_(1.0/tau_)
{
static_assert(LT<T>::ND % 4 == 0);
static_assert(std::is_same<T,double>::value == true);

return;
}


template<timestep TS>
void implementation(bool const isSave) noexcept;

protected:
T const nu_;
T const tau_;
T const omega_;
};


template <unsigned int NX, unsigned int NY, unsigned int NZ, template <typename T> class LT, typename T, unsigned int NPOP> template<timestep TS>
void BGK_AVX2<NX,NY,NZ,LT,T,NPOP>::implementation(bool const isSave) noexcept
{
#pragma omp parallel for default(none) shared(CO::continuum_,CO::population_) firstprivate(isSave,CO::p_) schedule(static,1)
for(unsigned int block = 0; block < CO::NUM_BLOCKS_; ++block)
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

alignas(CACHE_LINE) double f[LT<T>::ND] = {0.0};

#pragma GCC unroll (2)
for(unsigned int n = 0; n <= 1; ++n)
{
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::OFF)
#else
#pragma GCC unroll (16)
#endif
for(unsigned int d = 0; d < LT<T>::OFF; ++d)
{
f[n*LT<T>::OFF + d] = CO::population_->A[CO::population_-> template indexRead<TS>(x_n,y_n,z_n,n,d,CO::p_)];
}
}

__m256d _rho = _mm256_setzero_pd();
__m256d _u   = _mm256_setzero_pd();
__m256d _v   = _mm256_setzero_pd();
__m256d _w   = _mm256_setzero_pd();

for (size_t i = 0; i < LT<T>::ND; i += AVX2_REG_SIZE)
{
_rho = _mm256_add_pd(_mm256_load_pd(&f[i]), _rho);
_u   = _mm256_fmadd_pd(_mm256_load_pd(&LT<T>::DX[i]), _mm256_load_pd(&f[i]), _u);
_v   = _mm256_fmadd_pd(_mm256_load_pd(&LT<T>::DY[i]), _mm256_load_pd(&f[i]), _v);
_w   = _mm256_fmadd_pd(_mm256_load_pd(&LT<T>::DZ[i]), _mm256_load_pd(&f[i]), _w);
}

double const rho = _mm256_reduce_add_pd(_rho);
double const u   = _mm256_reduce_add_pd(_u)/rho;
double const v   = _mm256_reduce_add_pd(_v)/rho;
double const w   = _mm256_reduce_add_pd(_w)/rho;

if (isSave == true)
{
CO::continuum_->operator()(x, y, z, 0) = rho;
CO::continuum_->operator()(x, y, z, 1) = u;
CO::continuum_->operator()(x, y, z, 2) = v;
CO::continuum_->operator()(x, y, z, 3) = w;
}

alignas(CACHE_LINE) double feq[LT<T>::ND] = {0.0};

__m256d const _uu = _mm256_set1_pd(-1.0/(2.0*LT<T>::CS*LT<T>::CS)*(u*u + v*v + w*w));
_rho = _mm256_set1_pd(rho);
_u   = _mm256_set1_pd(u);
_v   = _mm256_set1_pd(v);
_w   = _mm256_set1_pd(w);

for (size_t i = 0; i < LT<T>::ND; i += AVX2_REG_SIZE)
{
__m256d _cu = _mm256_mul_pd(_mm256_load_pd(&LT<T>::DX[i]), _u);
_cu = _mm256_fmadd_pd(_mm256_load_pd(&LT<T>::DY[i]), _v, _cu);
_cu = _mm256_fmadd_pd(_mm256_load_pd(&LT<T>::DZ[i]), _w, _cu);
_cu = _mm256_mul_pd(_cu, _mm256_set1_pd(1.0/(LT<T>::CS*LT<T>::CS)));

__m256d _res = _mm256_fmadd_pd(_mm256_set1_pd(0.5), _cu, _mm256_set1_pd(1.0));
_res = _mm256_fmadd_pd(_cu, _res, _uu);

_res = _mm256_fmadd_pd(_res, _rho, _rho);
_res = _mm256_mul_pd(_mm256_load_pd(&LT<T>::W[i]), _res);
_mm256_store_pd(&feq[i], _res);
}

for (size_t i = 0; i < LT<T>::ND; i += AVX2_REG_SIZE)
{
__m256d _res = _mm256_sub_pd(_mm256_load_pd(&feq[i]), _mm256_load_pd(&f[i]));
_res = _mm256_fmadd_pd(_mm256_set1_pd(omega_), _res, _mm256_load_pd(&f[i]));
_mm256_store_pd(&f[i], _mm256_mul_pd(_mm256_load_pd(&LT<T>::MASK[i]), _res));
}

#pragma GCC unroll (2)
for(unsigned int n = 0; n <= 1; ++n)
{
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::OFF)
#else
#pragma GCC unroll (16)
#endif
for(unsigned int d = 0; d < LT<T>::OFF; ++d)
{
size_t const curr = n*LT<T>::OFF + d;
CO::population_->A[CO::population_-> template indexWrite<TS>(x_n,y_n,z_n,n,d,CO::p_)] = f[curr];
}
}
}
}
}
}

return;
}

#endif 

#endif 
