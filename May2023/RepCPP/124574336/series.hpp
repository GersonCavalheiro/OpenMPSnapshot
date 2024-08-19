
#ifndef BOOST_MATH_TOOLS_SERIES_INCLUDED
#define BOOST_MATH_TOOLS_SERIES_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/config/no_tr1/cmath.hpp>
#include <boost/cstdint.hpp>
#include <boost/limits.hpp>
#include <boost/math/tools/config.hpp>

namespace boost{ namespace math{ namespace tools{

template <class Functor, class U, class V>
inline typename Functor::result_type sum_series(Functor& func, const U& factor, boost::uintmax_t& max_terms, const V& init_value) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING

typedef typename Functor::result_type result_type;

boost::uintmax_t counter = max_terms;

result_type result = init_value;
result_type next_term;
do{
next_term = func();
result += next_term;
}
while((abs(factor * result) < abs(next_term)) && --counter);

max_terms = max_terms - counter;

return result;
}

template <class Functor, class U>
inline typename Functor::result_type sum_series(Functor& func, const U& factor, boost::uintmax_t& max_terms) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
typename Functor::result_type init_value = 0;
return sum_series(func, factor, max_terms, init_value);
}

template <class Functor, class U>
inline typename Functor::result_type sum_series(Functor& func, int bits, boost::uintmax_t& max_terms, const U& init_value) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING
typedef typename Functor::result_type result_type;
result_type factor = ldexp(result_type(1), 1 - bits);
return sum_series(func, factor, max_terms, init_value);
}

template <class Functor>
inline typename Functor::result_type sum_series(Functor& func, int bits) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING
typedef typename Functor::result_type result_type;
boost::uintmax_t iters = (std::numeric_limits<boost::uintmax_t>::max)();
result_type init_val = 0;
return sum_series(func, bits, iters, init_val);
}

template <class Functor>
inline typename Functor::result_type sum_series(Functor& func, int bits, boost::uintmax_t& max_terms) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING
typedef typename Functor::result_type result_type;
result_type init_val = 0;
return sum_series(func, bits, max_terms, init_val);
}

template <class Functor, class U>
inline typename Functor::result_type sum_series(Functor& func, int bits, const U& init_value) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING
boost::uintmax_t iters = (std::numeric_limits<boost::uintmax_t>::max)();
return sum_series(func, bits, iters, init_value);
}
template <class Functor, class U, class V>
inline typename Functor::result_type checked_sum_series(Functor& func, const U& factor, boost::uintmax_t& max_terms, const V& init_value, V& norm) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING

typedef typename Functor::result_type result_type;

boost::uintmax_t counter = max_terms;

result_type result = init_value;
result_type next_term;
do {
next_term = func();
result += next_term;
norm += fabs(next_term);
} while ((abs(factor * result) < abs(next_term)) && --counter);

max_terms = max_terms - counter;

return result;
}


template <class Functor>
inline typename Functor::result_type kahan_sum_series(Functor& func, int bits) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING

typedef typename Functor::result_type result_type;

result_type factor = pow(result_type(2), bits);
result_type result = func();
result_type next_term, y, t;
result_type carry = 0;
do{
next_term = func();
y = next_term - carry;
t = result + y;
carry = t - result;
carry -= y;
result = t;
}
while(fabs(result) < fabs(factor * next_term));
return result;
}

template <class Functor>
inline typename Functor::result_type kahan_sum_series(Functor& func, int bits, boost::uintmax_t& max_terms) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(typename Functor::result_type) && noexcept(std::declval<Functor>()()))
{
BOOST_MATH_STD_USING

typedef typename Functor::result_type result_type;

boost::uintmax_t counter = max_terms;

result_type factor = ldexp(result_type(1), bits);
result_type result = func();
result_type next_term, y, t;
result_type carry = 0;
do{
next_term = func();
y = next_term - carry;
t = result + y;
carry = t - result;
carry -= y;
result = t;
}
while((fabs(result) < fabs(factor * next_term)) && --counter);

max_terms = max_terms - counter;

return result;
}

} 
} 
} 

#endif 

