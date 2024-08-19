

#ifndef BOOST_MATH_TOOLS_POLYNOMIAL_GCD_HPP
#define BOOST_MATH_TOOLS_POLYNOMIAL_GCD_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/polynomial.hpp>
#include <boost/integer/common_factor_rt.hpp>
#include <boost/type_traits/is_pod.hpp>


namespace boost{

namespace integer {

namespace gcd_detail {

template <class T>
struct gcd_traits;

template <class T>
struct gcd_traits<boost::math::tools::polynomial<T> >
{
inline static const boost::math::tools::polynomial<T>& abs(const boost::math::tools::polynomial<T>& val) { return val; }

static const method_type method = method_euclid;
};

}
}



namespace math{ namespace tools{



template <class T>
T content(polynomial<T> const &x)
{
return x ? boost::integer::gcd_range(x.data().begin(), x.data().end()).first : T(0);
}

template <class T>
polynomial<T> primitive_part(polynomial<T> const &x, T const &cont)
{
return x ? x / cont : polynomial<T>();
}


template <class T>
polynomial<T> primitive_part(polynomial<T> const &x)
{
return primitive_part(x, content(x));
}


template <class T>
T leading_coefficient(polynomial<T> const &x)
{
return x ? x.data().back() : T(0);
}


namespace detail
{

template <class T>
T reduce_to_primitive(polynomial<T> &u, polynomial<T> &v)
{
using boost::integer::gcd;
T const u_cont = content(u), v_cont = content(v);
u /= u_cont;
v /= v_cont;
return gcd(u_cont, v_cont);
}
}



template <class T>
typename enable_if_c< std::numeric_limits<T>::is_integer, polynomial<T> >::type
subresultant_gcd(polynomial<T> u, polynomial<T> v)
{
using std::swap;
BOOST_ASSERT(u || v);

if (!u)
return v;
if (!v)
return u;

typedef typename polynomial<T>::size_type N;

if (u.degree() < v.degree())
swap(u, v);

T const d = detail::reduce_to_primitive(u, v);
T g = 1, h = 1;
polynomial<T> r;
while (true)
{
BOOST_ASSERT(u.degree() >= v.degree());
r = u % v;
if (!r)
return d * primitive_part(v); 
if (r.degree() == 0)
return d * polynomial<T>(T(1)); 
N const delta = u.degree() - v.degree();
u = v;
v = r / (g * detail::integer_power(h, delta));
g = leading_coefficient(u);
T const tmp = detail::integer_power(g, delta);
if (delta <= N(1))
h = tmp * detail::integer_power(h, N(1) - delta);
else
h = tmp / detail::integer_power(h, delta - N(1));
}
}



template <typename T>
typename enable_if_c<std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_bounded, polynomial<T> >::type
gcd(polynomial<T> const &u, polynomial<T> const &v)
{
return subresultant_gcd(u, v);
}
template <typename T>
typename enable_if_c<std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_bounded, polynomial<T> >::type
gcd(polynomial<T> const &u, polynomial<T> const &v)
{
BOOST_STATIC_ASSERT_MSG(sizeof(v) == 0, "GCD on polynomials of bounded integers is disallowed due to the excessive growth in the size of intermediate terms.");
return subresultant_gcd(u, v);
}
template <typename T>
typename enable_if_c<!std::numeric_limits<T>::is_integer && (std::numeric_limits<T>::min_exponent != std::numeric_limits<T>::max_exponent) && !std::numeric_limits<T>::is_exact, polynomial<T> >::type
gcd(polynomial<T> const &u, polynomial<T> const &v)
{
return boost::integer::gcd_detail::Euclid_gcd(u, v);
}

}
using boost::math::tools::gcd;

}

namespace integer
{
using boost::math::tools::gcd;
}

} 

#endif
