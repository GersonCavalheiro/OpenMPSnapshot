
#pragma once

#include "decs.hpp"

#include <memory>
#include <string>
#include <stdexcept>


template <typename T>
KOKKOS_INLINE_FUNCTION T clip(const T& n, const T& lower, const T& upper)
{
#if TRACE
#endif
return m::min(m::max(lower, n), upper);
}
template <typename T>
KOKKOS_INLINE_FUNCTION T bounce(const T& n, const T& lower, const T& upper)
{
return (n < lower) ? 2*lower - n : ( (n > upper) ? 2*upper - n : n );
}
template <typename T>
KOKKOS_INLINE_FUNCTION T excise(const T& n, const T& center, const T& range)
{
return (m::abs(n - center) > range) ? n : ( (n > center) ? center + range : center - range );
}

template <typename T>
KOKKOS_INLINE_FUNCTION T close_to(const T& x, const T& y, const Real& rel_tol=1e-8, const Real& abs_tol=1e-8)
{
return ((abs(x - y) / y) < rel_tol) || (abs(x) < abs_tol && abs(y) < abs_tol);
}

template <typename T>
KOKKOS_INLINE_FUNCTION void zero(T* a, const int& n)
{
memset(a, 0, n*sizeof(T));
}
template <typename T>
KOKKOS_INLINE_FUNCTION void gzero(T a[GR_DIM])
{
memset(a, 0, GR_DIM*sizeof(T));
}
template <typename T>
KOKKOS_INLINE_FUNCTION void zero2(T* a[], const int& n)
{
memset(&(a[0][0]), 0, n*sizeof(T));
}
template <typename T>
KOKKOS_INLINE_FUNCTION void gzero2(T a[GR_DIM][GR_DIM])
{
memset(&(a[0][0]), 0, GR_DIM*GR_DIM*sizeof(T));
}
