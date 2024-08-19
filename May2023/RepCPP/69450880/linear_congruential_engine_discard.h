

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>
#include <hydra/detail/external/hydra_thrust/random/detail/mod.h>

namespace hydra_thrust
{

namespace random
{

namespace detail
{


template<typename UIntType, UIntType a, unsigned long long c, UIntType m>
struct linear_congruential_engine_discard_implementation
{
__host__ __device__
static void discard(UIntType &state, unsigned long long z)
{
for(; z > 0; --z)
{
state = detail::mod<UIntType,a,c,m>(state);
}
}
}; 


template<hydra_thrust::detail::uint32_t a, hydra_thrust::detail::uint32_t m>
struct linear_congruential_engine_discard_implementation<hydra_thrust::detail::uint32_t,a,0,m>
{
__host__ __device__
static void discard(hydra_thrust::detail::uint32_t &state, unsigned long long z)
{
const hydra_thrust::detail::uint32_t modulus = m;

unsigned long long multiplier = a;
unsigned long long multiplier_to_z = 1;

while(z > 0)
{
if(z & 1)
{
multiplier_to_z = (multiplier_to_z * multiplier) % modulus;
}

z >>= 1;
multiplier = (multiplier * multiplier) % modulus;
}

state = static_cast<hydra_thrust::detail::uint32_t>((multiplier_to_z * state) % modulus);
}
}; 


struct linear_congruential_engine_discard
{
template<typename LinearCongruentialEngine>
__host__ __device__
static void discard(LinearCongruentialEngine &lcg, unsigned long long z)
{
typedef typename LinearCongruentialEngine::result_type result_type;
const result_type c = LinearCongruentialEngine::increment;
const result_type a = LinearCongruentialEngine::multiplier;
const result_type m = LinearCongruentialEngine::modulus;

(void) c;
(void) a;
(void) m;

linear_congruential_engine_discard_implementation<result_type,a,c,m>::discard(lcg.m_x, z);
}
}; 


} 

} 

} 

