





#pragma once

namespace hydra_thrust
{

namespace detail
{

namespace mpl
{

namespace math
{

namespace detail
{

template <unsigned int N, unsigned int Cur>
struct log2
{
static const unsigned int value = log2<N / 2,Cur+1>::value;
};

template <unsigned int Cur>
struct log2<1, Cur>
{
static const unsigned int value = Cur;
};

template <unsigned int Cur>
struct log2<0, Cur>
{
};

} 


template <unsigned int N>
struct log2
{
static const unsigned int value = detail::log2<N,0>::value;
};


template <typename T, T lhs, T rhs>
struct min
{
static const T value = (lhs < rhs) ? lhs : rhs;
};


template <typename T, T lhs, T rhs>
struct max
{
static const T value = (!(lhs < rhs)) ? lhs : rhs;
};


template<typename result_type, result_type x, result_type y>
struct mul
{
static const result_type value = x * y;
};


template<typename result_type, result_type x, result_type y>
struct mod
{
static const result_type value = x % y;
};


template<typename result_type, result_type x, result_type y>
struct div
{
static const result_type value = x / y;
};


template<typename result_type, result_type x, result_type y>
struct geq
{
static const bool value = x >= y;
};


template<typename result_type, result_type x, result_type y>
struct lt
{
static const bool value = x < y;
};


template<typename result_type, result_type x, result_type y>
struct gt
{
static const bool value = x > y;
};


template<bool x, bool y>
struct or_
{
static const bool value = (x || y);
};


template<typename result_type, result_type x, result_type y>
struct bit_and
{
static const result_type value = x & y;
};


template<typename result_type, result_type x, result_type y>
struct plus
{
static const result_type value = x + y;
};


template<typename result_type, result_type x, result_type y>
struct minus
{
static const result_type value = x - y;
};


template<typename result_type, result_type x, result_type y>
struct equal
{
static const bool value = x == y;
};


template<typename result_type, result_type x>
struct is_odd
{
static const bool value = x & 1;
};


} 

} 

} 

} 

