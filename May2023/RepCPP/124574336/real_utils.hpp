
#if !defined(BOOST_SPIRIT_KARMA_REAL_UTILS_FEB_23_2007_0841PM)
#define BOOST_SPIRIT_KARMA_REAL_UTILS_FEB_23_2007_0841PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/config/no_tr1/cmath.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/limits.hpp>

#include <boost/spirit/home/support/char_class.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/detail/pow10.hpp>
#include <boost/spirit/home/support/detail/sign.hpp>
#include <boost/spirit/home/karma/detail/generate_to.hpp>
#include <boost/spirit/home/karma/detail/string_generate.hpp>
#include <boost/spirit/home/karma/numeric/detail/numeric_utils.hpp>

namespace boost { namespace spirit { namespace karma 
{ 
template <typename T>
struct real_policies;

template <typename T
, typename Policies = real_policies<T>
, typename CharEncoding = unused_type
, typename Tag = unused_type>
struct real_inserter
{
template <typename OutputIterator, typename U>
static bool
call (OutputIterator& sink, U n, Policies const& p = Policies())
{
if (traits::test_nan(n)) {
return p.template nan<CharEncoding, Tag>(
sink, n, p.force_sign(n));
}
else if (traits::test_infinite(n)) {
return p.template inf<CharEncoding, Tag>(
sink, n, p.force_sign(n));
}
return p.template call<real_inserter>(sink, n, p);
}

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)  
# pragma warning(push)
# pragma warning(disable: 4100)   
# pragma warning(disable: 4127)   
# pragma warning(disable: 4267)   
#endif 
template <typename OutputIterator, typename U>
static bool
call_n (OutputIterator& sink, U n, Policies const& p)
{
bool force_sign = p.force_sign(n);
bool sign_val = false;
int flags = p.floatfield(n);
if (traits::test_negative(n)) 
{
n = -n;
sign_val = true;
}


unsigned precision = p.precision(n);
if (std::numeric_limits<U>::digits10) 
{
precision = (std::min)(precision, 
(unsigned)std::numeric_limits<U>::digits10 + 1);
}

using namespace std;

U dim = 0;
if (0 == (Policies::fmtflags::fixed & flags) && !traits::test_zero(n))
{
dim = log10(n);
if (dim > 0) 
n /= spirit::traits::pow10<U>(traits::truncate_to_long::call(dim));
else if (n < 1.) {
long exp = traits::truncate_to_long::call(-dim);
if (exp != -dim)
++exp;
dim = static_cast<U>(-exp);
if (exp > std::numeric_limits<U>::max_exponent10)
{
n *= spirit::traits::pow10<U>(std::numeric_limits<U>::max_exponent10);
n *= spirit::traits::pow10<U>(exp - std::numeric_limits<U>::max_exponent10);
} else
n *= spirit::traits::pow10<U>(exp);
}
}

U integer_part;
U precexp = spirit::traits::pow10<U>(precision);
U fractional_part = modf(n, &integer_part);

fractional_part = floor(fractional_part * precexp + U(0.5));
if (fractional_part >= precexp) 
{
fractional_part = floor(fractional_part - precexp);
integer_part += 1;    
}

U long_int_part = floor(integer_part);
U long_frac_part = fractional_part;
unsigned prec = precision;
if (!p.trailing_zeros(n))
{
U frac_part_floor = long_frac_part;
if (0 != long_frac_part) {
while (0 != prec && 
0 == traits::remainder<10>::call(long_frac_part)) 
{
long_frac_part = traits::divide<10>::call(long_frac_part);
--prec;
}
}
else {
prec = 0;
}

if (precision != prec)
{
long_frac_part = frac_part_floor / 
spirit::traits::pow10<U>(precision-prec);
}
}

if ((force_sign || sign_val) &&
traits::test_zero(long_int_part) &&
traits::test_zero(long_frac_part))
{
sign_val = false;     
force_sign = false;
}

bool r = p.integer_part(sink, long_int_part, sign_val, force_sign);

r = r && p.dot(sink, long_frac_part, precision);

r = r && p.fraction_part(sink, long_frac_part, prec, precision);

if (r && 0 == (Policies::fmtflags::fixed & flags)) {
return p.template exponent<CharEncoding, Tag>(sink, 
traits::truncate_to_long::call(dim));
}
return r;
}

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif 

};
}}}

#endif

