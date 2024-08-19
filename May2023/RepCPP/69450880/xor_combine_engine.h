



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/random/detail/xor_combine_engine_max.h>
#include <hydra/detail/external/hydra_thrust/random/detail/random_core_access.h>
#include <iostream>
#include <cstddef> 

namespace hydra_thrust
{

namespace random
{




template<typename Engine1, size_t s1,
typename Engine2, size_t s2=0u>
class xor_combine_engine
{
public:


typedef Engine1 base1_type;


typedef Engine2 base2_type;


typedef typename hydra_thrust::detail::eval_if<
(sizeof(typename base2_type::result_type) > sizeof(typename base1_type::result_type)),
hydra_thrust::detail::identity_<typename base2_type::result_type>,
hydra_thrust::detail::identity_<typename base1_type::result_type>
>::type result_type;


static const size_t shift1 = s1;


static const size_t shift2 = s2;


static const result_type min = 0;


static const result_type max =
detail::xor_combine_engine_max<
Engine1, s1, Engine2, s2, result_type
>::value;



__host__ __device__
xor_combine_engine(void);


__host__ __device__
xor_combine_engine(const base1_type &urng1, const base2_type &urng2);


__host__ __device__
xor_combine_engine(result_type s);


__host__ __device__
void seed(void);


__host__ __device__
void seed(result_type s);



__host__ __device__
result_type operator()(void);


__host__ __device__
void discard(unsigned long long z);



__host__ __device__
const base1_type &base1(void) const;


__host__ __device__
const base2_type &base2(void) const;


private:
base1_type m_b1;
base2_type m_b2;

friend struct hydra_thrust::random::detail::random_core_access;

__host__ __device__
bool equal(const xor_combine_engine &rhs) const;

template<typename CharT, typename Traits>
std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

template<typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;


}; 



template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_>
__host__ __device__
bool operator==(const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &lhs,
const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &rhs);



template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_>
__host__ __device__
bool operator!=(const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &lhs,
const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &rhs);



template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_,
typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &e);



template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_,
typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &e);





} 

using random::xor_combine_engine;

} 

#include <hydra/detail/external/hydra_thrust/random/detail/xor_combine_engine.inl>

