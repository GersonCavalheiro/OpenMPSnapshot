




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <hydra/detail/external/hydra_thrust/device_reference.h>
#include <hydra/detail/external/hydra_thrust/device_malloc.h>
#include <hydra/detail/external/hydra_thrust/device_free.h>
#include <limits>
#include <stdexcept>

namespace hydra_thrust
{

template<typename> class device_ptr;
template<typename T> device_ptr<T> device_malloc(const std::size_t n);




template<typename T>
class device_malloc_allocator
{
public:

typedef T                                 value_type;


typedef device_ptr<T>                     pointer;


typedef device_ptr<const T>               const_pointer;


typedef device_reference<T>               reference;


typedef device_reference<const T>         const_reference;


typedef std::size_t                       size_type;


typedef typename pointer::difference_type difference_type;


template<typename U>
struct rebind
{

typedef device_malloc_allocator<U> other;
}; 


__host__ __device__
inline device_malloc_allocator() {}


__host__ __device__
inline ~device_malloc_allocator() {}


__host__ __device__
inline device_malloc_allocator(device_malloc_allocator const&) {}


template<typename U>
__host__ __device__
inline device_malloc_allocator(device_malloc_allocator<U> const&) {}


__host__ __device__
inline pointer address(reference r) { return &r; }


__host__ __device__
inline const_pointer address(const_reference r) { return &r; }


__host__
inline pointer allocate(size_type cnt,
const_pointer = const_pointer(static_cast<T*>(0)))
{
if(cnt > this->max_size())
{
throw std::bad_alloc();
} 

return pointer(device_malloc<T>(cnt));
} 


__host__
inline void deallocate(pointer p, size_type cnt)
{
(void)(cnt);

device_free(p);
} 


inline size_type max_size() const
{
return (std::numeric_limits<size_type>::max)() / sizeof(T);
} 


__host__ __device__
inline bool operator==(device_malloc_allocator const&) const { return true; }


__host__ __device__
inline bool operator!=(device_malloc_allocator const &a) const {return !operator==(a); }
}; 



} 


