




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <hydra/detail/external/hydra_thrust/device_reference.h>
#include <hydra/detail/external/hydra_thrust/device_new.h>
#include <hydra/detail/external/hydra_thrust/device_delete.h>
#include <limits>
#include <stdexcept>

namespace hydra_thrust
{




template<typename T>
class device_new_allocator
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

typedef device_new_allocator<U> other;
}; 


__host__ __device__
inline device_new_allocator() {}


__host__ __device__
inline ~device_new_allocator() {}


__host__ __device__
inline device_new_allocator(device_new_allocator const&) {}


template<typename U>
__host__ __device__
inline device_new_allocator(device_new_allocator<U> const&) {}


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

return pointer(device_new<T>(cnt));
} 


__host__
inline void deallocate(pointer p, size_type cnt)
{
device_delete(p);
} 


__host__ __device__
inline size_type max_size() const
{
return std::numeric_limits<size_type>::max HYDRA_THRUST_PREVENT_MACRO_SUBSTITUTION () / sizeof(T);
} 


__host__ __device__
inline bool operator==(device_new_allocator const&) { return true; }


__host__ __device__
inline bool operator!=(device_new_allocator const &a) {return !operator==(a); }
}; 



} 

