




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/detail/integer_traits.h>
#include <hydra/detail/external/hydra_thrust/random/detail/random_core_access.h>
#include <iostream>

namespace hydra_thrust
{

namespace random
{




template<typename IntType = int>
class uniform_int_distribution
{
public:


typedef IntType result_type;


typedef hydra_thrust::pair<IntType,IntType> param_type;



__host__ __device__
explicit uniform_int_distribution(IntType a = 0, IntType b = hydra_thrust::detail::integer_traits<IntType>::const_max);


__host__ __device__
explicit uniform_int_distribution(const param_type &parm);


__host__ __device__
void reset(void);



template<typename UniformRandomNumberGenerator>
__host__ __device__
result_type operator()(UniformRandomNumberGenerator &urng);


template<typename UniformRandomNumberGenerator>
__host__ __device__
result_type operator()(UniformRandomNumberGenerator &urng, const param_type &parm);



__host__ __device__
result_type a(void) const;


__host__ __device__
result_type b(void) const;


__host__ __device__
param_type param(void) const;


__host__ __device__
void param(const param_type &parm);


__host__ __device__
result_type min HYDRA_THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;


__host__ __device__
result_type max HYDRA_THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;


private:
param_type m_param;

friend struct hydra_thrust::random::detail::random_core_access;

__host__ __device__
bool equal(const uniform_int_distribution &rhs) const;

template<typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

template<typename CharT, typename Traits>
std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

}; 



template<typename IntType>
__host__ __device__
bool operator==(const uniform_int_distribution<IntType> &lhs,
const uniform_int_distribution<IntType> &rhs);



template<typename IntType>
__host__ __device__
bool operator!=(const uniform_int_distribution<IntType> &lhs,
const uniform_int_distribution<IntType> &rhs);



template<typename IntType,
typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
const uniform_int_distribution<IntType> &d);



template<typename IntType,
typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
uniform_int_distribution<IntType> &d);





} 

using random::uniform_int_distribution;

} 

#include <hydra/detail/external/hydra_thrust/random/detail/uniform_int_distribution.inl>

