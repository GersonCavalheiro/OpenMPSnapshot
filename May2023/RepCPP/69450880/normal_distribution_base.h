



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/random/uniform_real_distribution.h>
#include <limits>
#include <cmath>

namespace hydra_thrust
{
namespace random
{
namespace detail
{

template<typename RealType>
class normal_distribution_nvcc
{
protected:
template<typename UniformRandomNumberGenerator>
__host__ __device__
RealType sample(UniformRandomNumberGenerator &urng, const RealType mean, const RealType stddev)
{
typedef typename UniformRandomNumberGenerator::result_type uint_type;
const uint_type urng_range = UniformRandomNumberGenerator::max - UniformRandomNumberGenerator::min;

const RealType S1 = static_cast<RealType>(1) / urng_range;
const RealType S2 = S1 / 2;

RealType S3 = static_cast<RealType>(-1.4142135623730950488016887242097); 

uint_type u = urng() - UniformRandomNumberGenerator::min;

if(u > (urng_range / 2))
{
u = urng_range - u;
S3 = -S3;
}

RealType p = u*S1 + S2;

return mean + stddev * S3 * erfcinv(2 * p);
}

__host__ __device__
void reset() {}
};

template<typename RealType>
class normal_distribution_portable
{
protected:
normal_distribution_portable()
: m_r1(), m_r2(), m_cached_rho(), m_valid(false)
{}

normal_distribution_portable(const normal_distribution_portable &other)
: m_r1(other.m_r1), m_r2(other.m_r2), m_cached_rho(other.m_cached_rho), m_valid(other.m_valid)
{}

void reset()
{
m_valid = false;
}

template<typename UniformRandomNumberGenerator>
__host__ __device__
RealType sample(UniformRandomNumberGenerator &urng, const RealType mean, const RealType stddev)
{
using std::sqrt; using std::log; using std::sin; using std::cos;

if(!m_valid)
{
uniform_real_distribution<RealType> u01;
m_r1 = u01(urng);
m_r2 = u01(urng);
m_cached_rho = sqrt(-RealType(2) * log(RealType(1)-m_r2));

m_valid = true;
}
else
{
m_valid = false;
}

const RealType pi = RealType(3.14159265358979323846);

RealType result = m_cached_rho * (m_valid ?
cos(RealType(2)*pi*m_r1) :
sin(RealType(2)*pi*m_r1));

return mean + stddev * result;
}

private:
RealType m_r1, m_r2, m_cached_rho;
bool m_valid;
};

template<typename RealType>
struct normal_distribution_base
{
#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
typedef normal_distribution_nvcc<RealType> type;
#else
typedef normal_distribution_portable<RealType> type;
#endif
};

} 
} 
} 

