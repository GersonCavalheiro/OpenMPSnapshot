




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <iostream>
#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>
#include <hydra/detail/external/hydra_thrust/random/detail/random_core_access.h>

namespace hydra_thrust
{

namespace random
{




template<typename Engine, size_t p, size_t r>
class discard_block_engine
{
public:


typedef Engine base_type;


typedef typename base_type::result_type result_type;



static const size_t block_size = p;


static const size_t used_block = r;


static const result_type min = base_type::min;


static const result_type max = base_type::max;



__host__ __device__
discard_block_engine();


__host__ __device__
explicit discard_block_engine(const base_type &urng);


__host__ __device__
explicit discard_block_engine(result_type s);


__host__ __device__
void seed(void);


__host__ __device__
void seed(result_type s);



__host__ __device__
result_type operator()(void);


__host__ __device__
void discard(unsigned long long z);



__host__ __device__
const base_type &base(void) const;


private:
base_type m_e;
unsigned int m_n;

friend struct hydra_thrust::random::detail::random_core_access;

__host__ __device__
bool equal(const discard_block_engine &rhs) const;

template<typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

template<typename CharT, typename Traits>
std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

}; 



template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator==(const discard_block_engine<Engine,p,r> &lhs,
const discard_block_engine<Engine,p,r> &rhs);



template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator!=(const discard_block_engine<Engine,p,r> &lhs,
const discard_block_engine<Engine,p,r> &rhs);



template<typename Engine, size_t p, size_t r,
typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
const discard_block_engine<Engine,p,r> &e);



template<typename Engine, size_t p, size_t r,
typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
discard_block_engine<Engine,p,r> &e);



} 

using random::discard_block_engine;

} 

#include <hydra/detail/external/hydra_thrust/random/detail/discard_block_engine.inl>

