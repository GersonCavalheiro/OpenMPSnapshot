

#pragma once

#include "detail/config.h"
#include "memory_resource.h"

namespace hydra_thrust
{
namespace mr
{

template<typename MR>
struct validator
{
#if __cplusplus >= 201103L
static_assert(
std::is_base_of<memory_resource<typename MR::pointer>, MR>::value,
"a type used as a memory resource must derive from memory_resource"
);
#endif
};

template<typename T, typename U>
struct validator2 : private validator<T>, private validator<U>
{
};

template<typename T>
struct validator2<T, T> : private validator<T>
{
};

} 
} 

