
#ifndef BOOST_SPIRIT_QI_NUMERIC_DETAIL_REAL_IMPL_HPP
#define BOOST_SPIRIT_QI_NUMERIC_DETAIL_REAL_IMPL_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <cmath>
#include <boost/limits.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/detail/pow10.hpp>
#include <boost/spirit/home/support/detail/sign.hpp>
#include <boost/integer.hpp>
#include <boost/assert.hpp>

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(push)
# pragma warning(disable: 4100)   
# pragma warning(disable: 4127)   
#endif

namespace boost { namespace spirit { namespace traits
{
using spirit::traits::pow10;

namespace detail
{
template <typename T, typename AccT>
void compensate_roundoff(T& n, AccT acc_n, mpl::true_)
{
int const comp = 10;
n = T((acc_n / comp) * comp);
n += T(acc_n % comp);
}

template <typename T, typename AccT>
void compensate_roundoff(T& n, AccT acc_n, mpl::false_)
{
n = acc_n;
}

template <typename T, typename AccT>
void compensate_roundoff(T& n, AccT acc_n)
{
compensate_roundoff(n, acc_n, is_integral<AccT>());
}
}

template <typename T, typename AccT>
inline bool
scale(int exp, T& n, AccT acc_n)
{
if (exp >= 0)
{
int const max_exp = std::numeric_limits<T>::max_exponent10;

if (is_floating_point<T>() && (exp > max_exp))
return false;
n = acc_n * pow10<T>(exp);
}
else
{
if (exp < std::numeric_limits<T>::min_exponent10)
{
int const min_exp = std::numeric_limits<T>::min_exponent10;
detail::compensate_roundoff(n, acc_n);
n /= pow10<T>(-min_exp);

exp += -min_exp;
if (is_floating_point<T>() && exp < min_exp)
return false;

n /= pow10<T>(-exp);
}
else
{
n = T(acc_n) / pow10<T>(-exp);
}
}
return true;
}

inline bool
scale(int , unused_type , unused_type )
{
return true;
}

template <typename T, typename AccT>
inline bool
scale(int exp, int frac, T& n, AccT acc_n)
{
return scale(exp - frac, n, acc_n);
}

inline bool
scale(int , int , unused_type )
{
return true;
}

inline float
negate(bool neg, float n)
{
return neg ? spirit::detail::changesign(n) : n;
}

inline double
negate(bool neg, double n)
{
return neg ? spirit::detail::changesign(n) : n;
}

inline long double
negate(bool neg, long double n)
{
return neg ? spirit::detail::changesign(n) : n;
}

template <typename T>
inline T
negate(bool neg, T const& n)
{
return neg ? -n : n;
}

inline unused_type
negate(bool , unused_type n)
{
return n;
}

template <typename T>
struct real_accumulator : mpl::identity<T> {};

template <>
struct real_accumulator<float>
: mpl::identity<uint_t<(sizeof(float)*CHAR_BIT)>::least> {};

template <>
struct real_accumulator<double>
: mpl::identity<uint_t<(sizeof(double)*CHAR_BIT)>::least> {};
}}}

namespace boost { namespace spirit { namespace qi  { namespace detail
{
BOOST_MPL_HAS_XXX_TRAIT_DEF(version)

template <typename T, typename RealPolicies>
struct real_impl
{
template <typename Iterator>
static std::size_t
ignore_excess_digits(Iterator& , Iterator const& , mpl::false_)
{
return 0;
}

template <typename Iterator>
static std::size_t
ignore_excess_digits(Iterator& first, Iterator const& last, mpl::true_)
{
return RealPolicies::ignore_excess_digits(first, last);
}

template <typename Iterator>
static std::size_t
ignore_excess_digits(Iterator& first, Iterator const& last)
{
typedef mpl::bool_<has_version<RealPolicies>::value> has_version;
return ignore_excess_digits(first, last, has_version());
}

template <typename Iterator, typename Attribute>
static bool
parse(Iterator& first, Iterator const& last, Attribute& attr,
RealPolicies const& p)
{
if (first == last)
return false;
Iterator save = first;

bool neg = p.parse_sign(first, last);

T n;

typename traits::real_accumulator<T>::type acc_n = 0;
bool got_a_number = p.parse_n(first, last, acc_n);
int excess_n = 0;

if (!got_a_number)
{
if (p.parse_nan(first, last, n) ||
p.parse_inf(first, last, n))
{
traits::assign_to(traits::negate(neg, n), attr);
return true;    
}

if (!p.allow_leading_dot)
{
first = save;
return false;
}
}
else
{
excess_n = static_cast<int>(ignore_excess_digits(first, last));
}

bool e_hit = false;
Iterator e_pos;
int frac_digits = 0;

if (p.parse_dot(first, last))
{
if (excess_n != 0)
{
ignore_excess_digits(first, last);
}
else if (p.parse_frac_n(first, last, acc_n, frac_digits))
{
BOOST_ASSERT(frac_digits >= 0);
}
else if (!got_a_number || !p.allow_trailing_dot)
{
first = save;
return false;
}

e_pos = first;
e_hit = p.parse_exp(first, last);
}
else
{
if (!got_a_number)
{
first = save;
return false;
}

e_pos = first;
e_hit = p.parse_exp(first, last);
if (p.expect_dot && !e_hit)
{
first = save;
return false;
}
}

if (e_hit)
{
int exp = 0;
if (p.parse_exp_n(first, last, exp))
{
if (!traits::scale(exp + excess_n, frac_digits, n, acc_n))
return false;
}
else
{
first = e_pos;
bool r = traits::scale(-frac_digits, n, acc_n);
BOOST_VERIFY(r);
}
}
else if (frac_digits)
{
bool r = traits::scale(-frac_digits, n, acc_n);
BOOST_VERIFY(r);
}
else
{
if (excess_n)
{
if (!traits::scale(excess_n, n, acc_n))
return false;
}
else
{
n = static_cast<T>(acc_n);
}
}

traits::assign_to(traits::negate(neg, n), attr);

return true;
}
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

}}}}

#endif
