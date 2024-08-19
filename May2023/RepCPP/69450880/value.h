


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/actor.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{


template<typename Eval> struct actor;


template<typename T>
class value
{
public:

template<typename Env>
struct result
{
typedef T type;
};

__host__ __device__
value(const T &arg)
: m_val(arg)
{}

template<typename Env>
__host__ __device__
T eval(const Env &) const
{
return m_val;
}

private:
T m_val;
}; 

template<typename T>
__host__ __device__
actor<value<T> > val(const T &x)
{
return value<T>(x);
} 


} 
} 
} 

