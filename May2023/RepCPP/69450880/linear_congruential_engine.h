




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <iostream>
#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>
#include <hydra/detail/external/hydra_thrust/random/detail/random_core_access.h>
#include <hydra/detail/external/hydra_thrust/random/detail/linear_congruential_engine_discard.h>

namespace hydra_thrust
{

namespace random
{




template<typename UIntType, UIntType a, UIntType c, UIntType m>
class linear_congruential_engine
{
public:


typedef UIntType result_type;



static const result_type multiplier = a;


static const result_type increment = c;


static const result_type modulus = m;


static const result_type min = c == 0u ? 1u : 0u;


static const result_type max = m - 1u;


static const result_type default_seed = 1u;



__host__ __device__
explicit linear_congruential_engine(result_type s = default_seed);


__host__ __device__
void seed(result_type s = default_seed);



__host__ __device__
result_type operator()(void);


__host__ __device__
void discard(unsigned long long z);


private:
result_type m_x;

static void transition(result_type &state);

friend struct hydra_thrust::random::detail::random_core_access;

friend struct hydra_thrust::random::detail::linear_congruential_engine_discard;

__host__ __device__
bool equal(const linear_congruential_engine &rhs) const;

template<typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

template<typename CharT, typename Traits>
std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);


}; 



template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_>
__host__ __device__
bool operator==(const linear_congruential_engine<UIntType_,a_,c_,m_> &lhs,
const linear_congruential_engine<UIntType_,a_,c_,m_> &rhs);



template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_>
__host__ __device__
bool operator!=(const linear_congruential_engine<UIntType_,a_,c_,m_> &lhs,
const linear_congruential_engine<UIntType_,a_,c_,m_> &rhs);



template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
const linear_congruential_engine<UIntType_,a_,c_,m_> &e);



template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
linear_congruential_engine<UIntType_,a_,c_,m_> &e);









typedef linear_congruential_engine<hydra_thrust::detail::uint32_t, 16807, 0, 2147483647> minstd_rand0;



typedef linear_congruential_engine<hydra_thrust::detail::uint32_t, 48271, 0, 2147483647> minstd_rand;



} 

using random::linear_congruential_engine;
using random::minstd_rand;
using random::minstd_rand0;

} 

#include <hydra/detail/external/hydra_thrust/random/detail/linear_congruential_engine.inl>

