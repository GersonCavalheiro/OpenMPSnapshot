



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/memory.h>
#include <hydra/detail/external/hydra_thrust/detail/vector_base.h>
#include <vector>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{



template<typename T, typename Allocator = allocator<T> >
class vector
: public hydra_thrust::detail::vector_base<T,Allocator>
{

private:
typedef hydra_thrust::detail::vector_base<T,Allocator> super_t;


public:


typedef typename super_t::size_type  size_type;
typedef typename super_t::value_type value_type;



vector();


explicit vector(size_type n);


explicit vector(size_type n, const value_type &value);


vector(const vector &x);

#if __cplusplus >= 201103L

vector(vector &&x);
#endif


template<typename OtherT, typename OtherAllocator>
vector(const hydra_thrust::detail::vector_base<OtherT,OtherAllocator> &x);


template<typename OtherT, typename OtherAllocator>
vector(const std::vector<OtherT,OtherAllocator> &x);


template<typename InputIterator>
vector(InputIterator first, InputIterator last);



vector &operator=(const vector &x);

#if __cplusplus >= 201103L

vector &operator=(vector &&x);
#endif


template<typename OtherT, typename OtherAllocator>
vector &operator=(const std::vector<OtherT,OtherAllocator> &x);


template<typename OtherT, typename OtherAllocator>
vector &operator=(const hydra_thrust::detail::vector_base<OtherT,OtherAllocator> &x);
}; 

} 
} 

namespace tbb
{

using hydra_thrust::system::tbb::vector;

} 

} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/vector.inl>

