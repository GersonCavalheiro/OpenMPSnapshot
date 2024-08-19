#ifndef LBT_BOUNDARY_BOUNCEBACK
#define LBT_BOUNDARY_BOUNCEBACK



#include <array>
#include <memory>
#if __has_include (<omp.h>)
#include <omp.h>
#endif

#include "../population.hpp"
#include "boundary.hpp"



template <unsigned int NX, unsigned int NY, unsigned int NZ, template <typename T> class LT, typename T, unsigned int NPOP>
class HalfwayBounceBack: public BoundaryCondition<NX,NY,NZ,LT,T,NPOP,HalfwayBounceBack<NX,NY,NZ,LT,T,NPOP>>
{
using BC = BoundaryCondition<NX,NY,NZ,LT,T,NPOP,HalfwayBounceBack<NX,NY,NZ,LT,T,NPOP>>;

public:
HalfwayBounceBack() = delete;


HalfwayBounceBack(std::shared_ptr<Population<NX,NY,NZ,LT,T,NPOP>> population, std::vector<boundary::Element<T>> const& boundaryElements,
unsigned int const p = 0) noexcept:
BC(population, boundaryElements, p)
{
return;
}


template <timestep TS>
void implementationBeforeCollisionOperator() noexcept;


template <timestep TS>
void implementationAfterCollisionOperator() noexcept;
};

template <unsigned int NX, unsigned int NY, unsigned int NZ, template <typename T> class LT, typename T, unsigned int NPOP> template <timestep TS>
void HalfwayBounceBack<NX,NY,NZ,LT,T,NPOP>::implementationBeforeCollisionOperator() noexcept
{
return;
}

template <unsigned int NX, unsigned int NY, unsigned int NZ, template <typename T> class LT, typename T, unsigned int NPOP> template <timestep TS>
void HalfwayBounceBack<NX,NY,NZ,LT,T,NPOP>::implementationAfterCollisionOperator() noexcept
{
#pragma omp parallel for
for(std::int64_t i = 0; i < BC::boundaryElements_.size(); ++i)
{
auto const& boundaryElement = BC::boundaryElements_[i];
std::array<unsigned int,3> const x_n = { (NX + boundaryElement.x - 1) % NX, boundaryElement.x, (boundaryElement.x + 1) % NX };
std::array<unsigned int,3> const y_n = { (NY + boundaryElement.y - 1) % NY, boundaryElement.y, (boundaryElement.y + 1) % NY };
std::array<unsigned int,3> const z_n = { (NZ + boundaryElement.z - 1) % NZ, boundaryElement.z, (boundaryElement.z + 1) % NZ };

#pragma GCC unroll (2)
for(unsigned int n = 0; n <= 1; ++n)
{
#if defined(__ICC) || defined(__ICL)
#pragma unroll (LT<T>::HSPEED-1)
#else
#pragma GCC unroll (15)
#endif
for(unsigned int d = 1; d < LT<T>::HSPEED; ++d)
{
BC::population_->A[BC::population_-> template indexWrite<TS>(x_n, y_n, z_n, !n, d, BC::p_)] =
BC::population_->A[BC::population_-> template indexRead<!TS>(x_n, y_n, z_n, n, d, BC::p_)];
}
}
}

return;
}

#endif 
