
#ifndef BOOST_MATH_SF_CBRT_HPP
#define BOOST_MATH_SF_CBRT_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/rational.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/mpl/divides.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_convertible.hpp>

namespace boost{ namespace math{

namespace detail
{

struct big_int_type
{
operator boost::uintmax_t()const;
};

template <class T>
struct largest_cbrt_int_type
{
typedef typename mpl::if_c<
boost::is_convertible<big_int_type, T>::value,
boost::uintmax_t,
unsigned int
>::type type;
};

template <class T, class Policy>
T cbrt_imp(T z, const Policy& pol)
{
BOOST_MATH_STD_USING
static const T P[] = { 
static_cast<T>(0.37568269008611818),
static_cast<T>(1.3304968705558024),
static_cast<T>(-1.4897101632445036),
static_cast<T>(1.2875573098219835),
static_cast<T>(-0.6398703759826468),
static_cast<T>(0.13584489959258635),
};
static const T correction[] = {
static_cast<T>(0.62996052494743658238360530363911),  
static_cast<T>(0.79370052598409973737585281963615),  
static_cast<T>(1),
static_cast<T>(1.2599210498948731647672106072782),   
static_cast<T>(1.5874010519681994747517056392723),   
};
if((boost::math::isinf)(z) || (z == 0))
return z;
if(!(boost::math::isfinite)(z))
{
return policies::raise_domain_error("boost::math::cbrt<%1%>(%1%)", "Argument to function must be finite but got %1%.", z, pol);
}

int i_exp, sign(1);
if(z < 0)
{
z = -z;
sign = -sign;
}

T guess = frexp(z, &i_exp);
int original_i_exp = i_exp; 
guess = tools::evaluate_polynomial(P, guess);
int i_exp3 = i_exp / 3;

typedef typename largest_cbrt_int_type<T>::type shift_type;

BOOST_STATIC_ASSERT( ::std::numeric_limits<shift_type>::radix == 2);

if(abs(i_exp3) < std::numeric_limits<shift_type>::digits)
{
if(i_exp3 > 0)
guess *= shift_type(1u) << i_exp3;
else
guess /= shift_type(1u) << -i_exp3;
}
else
{
guess = ldexp(guess, i_exp3);
}
i_exp %= 3;
guess *= correction[i_exp + 2];
typedef typename policies::precision<T, Policy>::type prec;
typedef typename mpl::divides<prec, boost::integral_constant<int, 3> >::type prec3;
typedef typename mpl::plus<prec3, boost::integral_constant<int, 3> >::type new_prec;
typedef typename policies::normalise<Policy, policies::digits2<new_prec::value> >::type new_policy;
T eps = (new_prec::value > 3) ? policies::get_epsilon<T, new_policy>() : ldexp(T(1), -2 - tools::digits<T>() / 3);
T diff;

if(original_i_exp < std::numeric_limits<T>::max_exponent - 3)
{
do
{
T g3 = guess * guess * guess;
diff = (g3 + z + z) / (g3 + g3 + z);
guess *= diff;
}
while(fabs(1 - diff) > eps);
}
else
{
do
{
T g2 = guess * guess;
diff = (g2 - z / guess) / (2 * guess + z / g2);
guess -= diff;
}
while((guess * eps) < fabs(diff));
}

return sign * guess;
}

} 

template <class T, class Policy>
inline typename tools::promote_args<T>::type cbrt(T z, const Policy& pol)
{
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return static_cast<result_type>(detail::cbrt_imp(value_type(z), pol));
}

template <class T>
inline typename tools::promote_args<T>::type cbrt(T z)
{
return cbrt(z, policies::policy<>());
}

} 
} 

#endif 




