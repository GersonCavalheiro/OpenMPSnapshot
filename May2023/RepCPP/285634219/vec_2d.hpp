

#pragma once

#include <ostream>

#if defined(__NVCC__) || defined(__HIPCC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif


template <typename T>
struct alignas(2 * sizeof(T)) vec_2d {
using value_type = T;
value_type x;
value_type y;
};


template <typename T>
std::ostream& operator<<(std::ostream& os, vec_2d<T> const& vec)
{
return os << "(" << vec.x << "," << vec.y << ")";
}


template <typename T>
bool operator==(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
{
return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}


template <typename T>
vec_2d<T> HOST_DEVICE operator+(vec_2d<T> const& a, vec_2d<T> const& b)
{
return vec_2d<T>{a.x + b.x, a.y + b.y};
}


template <typename T>
vec_2d<T> HOST_DEVICE operator-(vec_2d<T> const& a, vec_2d<T> const& b)
{
return vec_2d<T>{a.x - b.x, a.y - b.y};
}


template <typename T>
vec_2d<T> HOST_DEVICE operator*(vec_2d<T> vec, T const& r)
{
return vec_2d<T>{vec.x * r, vec.y * r};
}


template <typename T>
vec_2d<T> HOST_DEVICE operator*(T const& r, vec_2d<T> vec)
{
return vec * r;
}


template <typename T>
T HOST_DEVICE dot(vec_2d<T> const& a, vec_2d<T> const& b)
{
return a.x * b.x + a.y * b.y;
}


template <typename T>
T HOST_DEVICE det(vec_2d<T> const& a, vec_2d<T> const& b)
{
return a.x * b.y - a.y * b.x;
}
