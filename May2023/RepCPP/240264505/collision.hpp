#ifndef LBT_COLLISION
#define LBT_COLLISION



#include <array>
#include <memory>

#include "../../general/constexpr_math.hpp"
#include "../../continuum/continuum.hpp"
#include "../population.hpp"



template <template <typename T> class LT, typename T, unsigned int NPOP>
class CollisionOperator
{
public:
CollisionOperator() = delete;
CollisionOperator& operator = (CollisionOperator&) = delete;
CollisionOperator(CollisionOperator&&) = delete;
CollisionOperator& operator = (CollisionOperator&&) = delete;
CollisionOperator(CollisionOperator const&) = delete;


template<timestep TS>
void initialise(T const RHO_0, T const U_0, T const V_0, T const W_0) noexcept;

protected:

CollisionOperator(std::shared_ptr<Population<LT,T,NPOP>> population, std::shared_ptr<Continuum<T>> continuum,
unsigned int const p = 0) noexcept:
population_(population), continuum_(continuum), p_(p)
{
return;
}


void initialiseContinuum_(T const RHO_0, T const U_0, T const V_0, T const W_0) noexcept;


template<timestep TS>
void initialisePopulationFromContinuum_() noexcept;

std::shared_ptr<Population<LT,T,NPOP>> population_;
std::shared_ptr<Continuum<T>>          continuum_;
unsigned int const p_;

static constexpr std::int32_t   BLOCK_SIZE_ = (LT<T>::DIM == 3) ? 32 : 96;                      
static constexpr std::int32_t NUM_BLOCKS_Z_ = cem::ceil(static_cast<double>(NZ) / BLOCK_SIZE_); 
static constexpr std::int32_t NUM_BLOCKS_Y_ = cem::ceil(static_cast<double>(NY) / BLOCK_SIZE_);
static constexpr std::int32_t NUM_BLOCKS_X_ = cem::ceil(static_cast<double>(NX) / BLOCK_SIZE_);
static constexpr std::int32_t   NUM_BLOCKS_ = NUM_BLOCKS_X_*NUM_BLOCKS_Y_*NUM_BLOCKS_Z_;        
};


template <template <typename T> class LT, typename T, unsigned int NPOP>
template<timestep TS>
void CollisionOperator<LT,T,NPOP>::initialise(T const RHO_0, T const U_0, T const V_0, T const W_0) noexcept
{
initialiseContinuum_(RHO_0, U_0, V_0, W_0);
initialisePopulationFromContinuum_<TS>();

return;
}

template <template <typename T> class LT, typename T, unsigned int NPOP>
void CollisionOperator<LT,T,NPOP>::initialiseContinuum_(T const RHO_0, T const U_0, T const V_0, T const W_0) noexcept
{
for(std::int32_t block = 0; block < NUM_BLOCKS_; ++block)
{
std::int32_t const z_start = BLOCK_SIZE_ * (block / (NUM_BLOCKS_X_*NUM_BLOCKS_Y_));
std::int32_t const   z_end = std::min(z_start + BLOCK_SIZE_, NZ);

for(std::int32_t z = z_start; z < z_end; ++z)
{
std::int32_t const y_start = BLOCK_SIZE_*((block % (NUM_BLOCKS_X_*NUM_BLOCKS_Y_)) / NUM_BLOCKS_X_);
std::int32_t const   y_end = std::min(y_start + BLOCK_SIZE_, NY);

for(std::int32_t y = y_start; y < y_end; ++y)
{
std::int32_t const x_start = BLOCK_SIZE_*(block % NUM_BLOCKS_X_);
std::int32_t const   x_end = std::min(x_start + BLOCK_SIZE_, NX);

for(std::int32_t x = x_start; x < x_end; ++x)
{
continuum_->operator()(x, y, z, 0) = RHO_0;
continuum_->operator()(x, y, z, 1) = U_0;
continuum_->operator()(x, y, z, 2) = V_0;
continuum_->operator()(x, y, z, 3) = W_0;
}
}
}
}

return;
}

template <template <typename T> class LT, typename T, unsigned int NPOP>
template<timestep TS>
void CollisionOperator<LT,T,NPOP>::initialisePopulationFromContinuum_() noexcept
{
for(std::int32_t block = 0; block < NUM_BLOCKS_; ++block)
{
std::int32_t const z_start = BLOCK_SIZE_ * (block / (NUM_BLOCKS_X_*NUM_BLOCKS_Y_));
std::int32_t const   z_end = std::min(z_start + BLOCK_SIZE_, NZ);

for(std::int32_t z = z_start; z < z_end; ++z)
{
std::array<std::int32_t,3> const z_n = { (NZ + z - 1) % NZ, z, (z + 1) % NZ };

std::int32_t const y_start = BLOCK_SIZE_*((block % (NUM_BLOCKS_X_*NUM_BLOCKS_Y_)) / NUM_BLOCKS_X_);
std::int32_t const   y_end = std::min(y_start + BLOCK_SIZE_, NY);

for(std::int32_t y = y_start; y < y_end; ++y)
{
std::array<std::int32_t,3> const y_n = { (NY + y - 1) % NY, y, (y + 1) % NY };

std::int32_t const x_start = BLOCK_SIZE_*(block % NUM_BLOCKS_X_);
std::int32_t const   x_end = std::min(x_start + BLOCK_SIZE_, NX);

for(std::int32_t x = x_start; x < x_end; ++x)
{
std::array<std::int32_t,3> const x_n = { (NX + x - 1) % NX, x, (x + 1) % NX };

T const rho = continuum_->operator()(x, y, z, 0);
T const u   = continuum_->operator()(x, y, z, 1);
T const v   = continuum_->operator()(x, y, z, 2);
T const w   = continuum_->operator()(x, y, z, 3);

T const uu = - 1.0/(2.0*LT<T>::CS*LT<T>::CS)*(u*u + v*v + w*w);

#pragma GCC unroll (2)
for(std::int32_t n = 0; n <= 1; ++n)
{
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED)
#else
#pragma GCC unroll (16)
#endif
for(std::int32_t d = 0; d < LT<T>::OFF; ++d)
{
std::int32_t const curr = n*LT<T>::OFF + d;
T const cu = 1.0/(LT<T>::CS*LT<T>::CS)*(u*LT<T>::DX[curr] + v*LT<T>::DY[curr] + w*LT<T>::DZ[curr]);
population_->A[population_->template indexRead<TS>(x_n,y_n,z_n,n,d,p_)] = LT<T>::MASK[curr]*
LT<T>::W[curr]*(rho + rho*(cu*(1.0 + 0.5*cu) + uu));
}
}
}
}
}
}

return;
}

#endif 
