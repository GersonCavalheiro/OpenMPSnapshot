




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/random/detail/random_core_access.h>
#include <hydra/detail/external/hydra_thrust/random/detail/normal_distribution_base.h>
#include <iostream>

namespace hydra_thrust
{

namespace random
{





template<typename RealType = double>
class normal_distribution
: public detail::normal_distribution_base<RealType>::type
{
private:
typedef typename detail::normal_distribution_base<RealType>::type super_t;

public:


typedef RealType result_type;


typedef hydra_thrust::pair<RealType,RealType> param_type;



__host__ __device__
explicit normal_distribution(RealType mean = 0.0, RealType stddev = 1.0);


__host__ __device__
explicit normal_distribution(const param_type &parm);


__host__ __device__
void reset(void);



template<typename UniformRandomNumberGenerator>
__host__ __device__
result_type operator()(UniformRandomNumberGenerator &urng);


template<typename UniformRandomNumberGenerator>
__host__ __device__
result_type operator()(UniformRandomNumberGenerator &urng, const param_type &parm);



__host__ __device__
result_type mean(void) const;


__host__ __device__
result_type stddev(void) const;


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
bool equal(const normal_distribution &rhs) const;

template<typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

template<typename CharT, typename Traits>
std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

}; 



template<typename RealType>
__host__ __device__
bool operator==(const normal_distribution<RealType> &lhs,
const normal_distribution<RealType> &rhs);



template<typename RealType>
__host__ __device__
bool operator!=(const normal_distribution<RealType> &lhs,
const normal_distribution<RealType> &rhs);



template<typename RealType,
typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
const normal_distribution<RealType> &d);



template<typename RealType,
typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
normal_distribution<RealType> &d);





} 

using random::normal_distribution;

} 

#include <hydra/detail/external/hydra_thrust/random/detail/normal_distribution.inl>

