


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>


namespace hydra_thrust
{
namespace detail
{
namespace util
{


template<typename T>
__host__ __device__
T *align_up(T * ptr, detail::uintptr_t bytes)
{
return (T *) ( bytes * (((detail::uintptr_t) ptr + (bytes - 1)) / bytes) );
}


template<typename T>
__host__ __device__
T *align_down(T * ptr, detail::uintptr_t bytes)
{
return (T *) ( bytes * (detail::uintptr_t(ptr) / bytes) );
}


template<typename T>
__host__ __device__
bool is_aligned(T * ptr, detail::uintptr_t bytes = sizeof(T))
{
return detail::uintptr_t(ptr) % bytes == 0;
}


} 
} 
} 

