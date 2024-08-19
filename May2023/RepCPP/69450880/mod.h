

#pragma once

namespace hydra_thrust
{

namespace random
{

namespace detail
{

template<typename T, T a, T c, T m, bool = (m == 0)>
struct static_mod
{
static const T q = m / a;
static const T r = m % a;

__host__ __device__
T operator()(T x) const
{
if(a == 1)
{
x %= m;
}
else
{
T t1 = a * (x % q);
T t2 = r * (x / q);
if(t1 >= t2)
{
x = t1 - t2;
}
else
{
x = m - t2 + t1;
}
}

if(c != 0)
{
const T d = m - x;
if(d > c)
{
x += c;
}
else
{
x = c - d;
}
}

return x;
}
}; 


template<typename T, T a, T c, T m>
struct static_mod<T,a,c,m,true>
{
__host__ __device__
T operator()(T x) const
{
return a * x + c;
}
}; 

template<typename T, T a, T c, T m>
__host__ __device__
T mod(T x)
{
static_mod<T,a,c,m> f;
return f(x);
} 

} 

} 

} 

