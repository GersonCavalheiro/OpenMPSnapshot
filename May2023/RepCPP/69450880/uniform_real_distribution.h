




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/random/detail/random_core_access.h>
#include <iostream>

namespace hydra_thrust
{

namespace random
{





template<typename RealType = double>
class uniform_real_distribution
{
public:


typedef RealType result_type;


typedef hydra_thrust::pair<RealType,RealType> param_type;



__host__ __device__
explicit uniform_real_distribution(RealType a = 0.0, RealType b = 1.0);


__host__ __device__
explicit uniform_real_distribution(const param_type &parm);


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
bool equal(const uniform_real_distribution &rhs) const;

template<typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

template<typename CharT, typename Traits>
std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

}; 



template<typename RealType>
__host__ __device__
bool operator==(const uniform_real_distribution<RealType> &lhs,
const uniform_real_distribution<RealType> &rhs);



template<typename RealType>
__host__ __device__
bool operator!=(const uniform_real_distribution<RealType> &lhs,
const uniform_real_distribution<RealType> &rhs);



template<typename RealType,
typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
const uniform_real_distribution<RealType> &d);



template<typename RealType,
typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
uniform_real_distribution<RealType> &d);





} 

using random::uniform_real_distribution;

} 

#include <hydra/detail/external/hydra_thrust/random/detail/uniform_real_distribution.inl>

