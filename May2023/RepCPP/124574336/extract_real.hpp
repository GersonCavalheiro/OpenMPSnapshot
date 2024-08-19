
#if !defined(BOOST_SPIRIT_X3_EXTRACT_REAL_APRIL_18_2006_0901AM)
#define BOOST_SPIRIT_X3_EXTRACT_REAL_APRIL_18_2006_0901AM

#include <cmath>
#include <boost/limits.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/spirit/home/x3/support/unused.hpp>
#include <boost/spirit/home/x3/support/numeric_utils/pow10.hpp>
#include <boost/spirit/home/x3/support/numeric_utils/sign.hpp>
#include <boost/spirit/home/x3/support/traits/move_to.hpp>
#include <boost/assert.hpp>

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(push)
# pragma warning(disable: 4100)   
# pragma warning(disable: 4127)   
#endif

namespace boost { namespace spirit { namespace x3 { namespace extension
{
using x3::traits::pow10;

template <typename T>
inline bool
scale(int exp, T& n)
{
constexpr auto max_exp = std::numeric_limits<T>::max_exponent10;
constexpr auto min_exp = std::numeric_limits<T>::min_exponent10;

if (exp >= 0)
{
if (is_floating_point<T>() && exp > max_exp)
return false;
n *= pow10<T>(exp);
}
else
{
if (exp < min_exp)
{
n /= pow10<T>(-min_exp);

exp += -min_exp;
if (is_floating_point<T>() && exp < min_exp)
return false;

n /= pow10<T>(-exp);
}
else
{
n /= pow10<T>(-exp);
}
}
return true;
}

inline bool
scale(int , unused_type )
{
return true;
}

template <typename T>
inline bool
scale(int exp, int frac, T& n)
{
return scale(exp - frac, n);
}

inline bool
scale(int , int , unused_type )
{
return true;
}

inline float
negate(bool neg, float n)
{
return neg ? x3::changesign(n) : n;
}

inline double
negate(bool neg, double n)
{
return neg ? x3::changesign(n) : n;
}

inline long double
negate(bool neg, long double n)
{
return neg ? x3::changesign(n) : n;
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
}}}}

namespace boost { namespace spirit { namespace x3
{
template <typename T, typename RealPolicies>
struct extract_real
{
template <typename Iterator, typename Attribute>
static bool
parse(Iterator& first, Iterator const& last, Attribute& attr,
RealPolicies const& p)
{
if (first == last)
return false;
Iterator save = first;

bool neg = p.parse_sign(first, last);

T n = 0;
bool got_a_number = p.parse_n(first, last, n);

if (!got_a_number)
{
if (p.parse_nan(first, last, n) ||
p.parse_inf(first, last, n))
{
traits::move_to(extension::negate(neg, n), attr);
return true;    
}

if (!p.allow_leading_dot)
{
first = save;
return false;
}
}

bool e_hit = false;
Iterator e_pos;
int frac_digits = 0;

if (p.parse_dot(first, last))
{
Iterator savef = first;
if (p.parse_frac_n(first, last, n))
{
if (!is_same<T, unused_type>::value)
frac_digits =
static_cast<int>(std::distance(savef, first));
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
if (!extension::scale(exp, frac_digits, n))
return false;
}
else
{
first = e_pos;

if (!extension::scale(-frac_digits, n))
return false;
}
}
else if (frac_digits)
{
if (!extension::scale(-frac_digits, n))
return false;
}

traits::move_to(extension::negate(neg, n), attr);

return true;
}
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

}}}

#endif
