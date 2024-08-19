



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/random/detail/random_core_access.h>

#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>
#include <cstddef> 
#include <iostream>

namespace hydra_thrust
{

namespace random
{





template<typename UIntType, size_t w, size_t s, size_t r>
class subtract_with_carry_engine
{

private:
static const UIntType modulus = UIntType(1) << w;


public:


typedef UIntType result_type;



static const size_t word_size = w;


static const size_t short_lag = s;


static const size_t long_lag = r;


static const result_type min = 0;


static const result_type max = modulus - 1;


static const result_type default_seed = 19780503u;



__host__ __device__
explicit subtract_with_carry_engine(result_type value = default_seed);


__host__ __device__
void seed(result_type value = default_seed);



__host__ __device__
result_type operator()(void);


__host__ __device__
void discard(unsigned long long z);


private:
result_type m_x[long_lag];
unsigned int m_k;
int m_carry;

friend struct hydra_thrust::random::detail::random_core_access;

__host__ __device__
bool equal(const subtract_with_carry_engine &rhs) const;

template<typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

template<typename CharT, typename Traits>
std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);


}; 



template<typename UIntType_, size_t w_, size_t s_, size_t r_>
__host__ __device__
bool operator==(const subtract_with_carry_engine<UIntType_,w_,s_,r_> &lhs,
const subtract_with_carry_engine<UIntType_,w_,s_,r_> &rhs);



template<typename UIntType_, size_t w_, size_t s_, size_t r_>
__host__ __device__
bool operator!=(const subtract_with_carry_engine<UIntType_,w_,s_,r_>&lhs,
const subtract_with_carry_engine<UIntType_,w_,s_,r_>&rhs);



template<typename UIntType_, size_t w_, size_t s_, size_t r_,
typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
const subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);



template<typename UIntType_, size_t w_, size_t s_, size_t r_,
typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);









typedef subtract_with_carry_engine<hydra_thrust::detail::uint32_t, 24, 10, 24> ranlux24_base;




typedef subtract_with_carry_engine<hydra_thrust::detail::uint64_t, 48,  5, 12> ranlux48_base;



} 

using random::subtract_with_carry_engine;
using random::ranlux24_base;
using random::ranlux48_base;

} 

#include <hydra/detail/external/hydra_thrust/random/detail/subtract_with_carry_engine.inl>

