

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/vector.h>
#include <utility>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{

template<typename T, typename Allocator>
vector<T,Allocator>
::vector()
: super_t()
{}

template<typename T, typename Allocator>
vector<T,Allocator>
::vector(size_type n)
: super_t(n)
{}

template<typename T, typename Allocator>
vector<T,Allocator>
::vector(size_type n, const value_type &value)
: super_t(n,value)
{}

template<typename T, typename Allocator>
vector<T,Allocator>
::vector(const vector &x)
: super_t(x)
{}

#if __cplusplus >= 201103L
template<typename T, typename Allocator>
vector<T,Allocator>
::vector(vector &&x)
: super_t(std::move(x))
{}
#endif

template<typename T, typename Allocator>
template<typename OtherT, typename OtherAllocator>
vector<T,Allocator>
::vector(const hydra_thrust::detail::vector_base<OtherT,OtherAllocator> &x)
: super_t(x)
{}

template<typename T, typename Allocator>
template<typename OtherT, typename OtherAllocator>
vector<T,Allocator>
::vector(const std::vector<OtherT,OtherAllocator> &x)
: super_t(x)
{}

template<typename T, typename Allocator>
template<typename InputIterator>
vector<T,Allocator>
::vector(InputIterator first, InputIterator last)
: super_t(first,last)
{}

template<typename T, typename Allocator>
vector<T,Allocator> &
vector<T,Allocator>
::operator=(const vector &x)
{
super_t::operator=(x);
return *this;
}

#if __cplusplus >= 201103L
template<typename T, typename Allocator>
vector<T,Allocator> &
vector<T,Allocator>
::operator=(vector &&x)
{
super_t::operator=(std::move(x));
return *this;
}
#endif

template<typename T, typename Allocator>
template<typename OtherT, typename OtherAllocator>
vector<T,Allocator> &
vector<T,Allocator>
::operator=(const std::vector<OtherT,OtherAllocator> &x)
{
super_t::operator=(x);
return *this;
}

template<typename T, typename Allocator>
template<typename OtherT, typename OtherAllocator>
vector<T,Allocator> &
vector<T,Allocator>
::operator=(const hydra_thrust::detail::vector_base<OtherT,OtherAllocator> &x)
{
super_t::operator=(x);
return *this;
}

} 
} 
} 

