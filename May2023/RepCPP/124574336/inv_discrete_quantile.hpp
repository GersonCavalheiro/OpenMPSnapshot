
#ifndef BOOST_MATH_DISTRIBUTIONS_DETAIL_INV_DISCRETE_QUANTILE
#define BOOST_MATH_DISTRIBUTIONS_DETAIL_INV_DISCRETE_QUANTILE

#include <algorithm>

namespace boost{ namespace math{ namespace detail{

template <class Dist>
struct distribution_quantile_finder
{
typedef typename Dist::value_type value_type;
typedef typename Dist::policy_type policy_type;

distribution_quantile_finder(const Dist d, value_type p, bool c)
: dist(d), target(p), comp(c) {}

value_type operator()(value_type const& x)
{
return comp ? value_type(target - cdf(complement(dist, x))) : value_type(cdf(dist, x) - target);
}

private:
Dist dist;
value_type target;
bool comp;
};
template <class Real, class Tol>
void adjust_bounds(Real& , Real& , Tol const& ){}

template <class Real>
void adjust_bounds(Real& , Real& b, tools::equal_floor const& )
{
BOOST_MATH_STD_USING
b -= tools::epsilon<Real>() * b;
}

template <class Real>
void adjust_bounds(Real& a, Real& , tools::equal_ceil const& )
{
BOOST_MATH_STD_USING
a += tools::epsilon<Real>() * a;
}

template <class Real>
void adjust_bounds(Real& a, Real& b, tools::equal_nearest_integer const& )
{
BOOST_MATH_STD_USING
a += tools::epsilon<Real>() * a;
b -= tools::epsilon<Real>() * b;
}
template <class Dist, class Tolerance>
typename Dist::value_type 
do_inverse_discrete_quantile(
const Dist& dist,
const typename Dist::value_type& p,
bool comp,
typename Dist::value_type guess,
const typename Dist::value_type& multiplier,
typename Dist::value_type adder,
const Tolerance& tol,
boost::uintmax_t& max_iter)
{
typedef typename Dist::value_type value_type;
typedef typename Dist::policy_type policy_type;

static const char* function = "boost::math::do_inverse_discrete_quantile<%1%>";

BOOST_MATH_STD_USING

distribution_quantile_finder<Dist> f(dist, p, comp);
value_type min_bound, max_bound;
boost::math::tie(min_bound, max_bound) = support(dist);

if(guess > max_bound)
guess = max_bound;
if(guess < min_bound)
guess = min_bound;

value_type fa = f(guess);
boost::uintmax_t count = max_iter - 1;
value_type fb(fa), a(guess), b =0; 

if(fa == 0)
return guess;

if(guess < 10)
{
b = a;
while((a < 10) && (fa * fb >= 0))
{
if(fb <= 0)
{
a = b;
b = a + 1;
if(b > max_bound)
b = max_bound;
fb = f(b);
--count;
if(fb == 0)
return b;
if(a == b)
return b; 
}
else
{
b = a;
a = (std::max)(value_type(b - 1), value_type(0));
if(a < min_bound)
a = min_bound;
fa = f(a);
--count;
if(fa == 0)
return a;
if(a == b)
return a;  
}
}
}
else if(adder != 0)
{
if(fa < 0)
{
b = a + adder;
if(b > max_bound)
b = max_bound;
}
else
{
b = (std::max)(value_type(a - adder), value_type(0));
if(b < min_bound)
b = min_bound;
}
fb = f(b);
--count;
if(fb == 0)
return b;
if(count && (fa * fb >= 0))
{
a = b;
fa = fb;
if(fa < 0)
{
b = a + adder;
if(b > max_bound)
b = max_bound;
}
else
{
b = (std::max)(value_type(a - adder), value_type(0));
if(b < min_bound)
b = min_bound;
}
fb = f(b);
--count;
}
if(a > b)
{
using std::swap;
swap(a, b);
swap(fa, fb);
}
}
if((boost::math::sign)(fb) == (boost::math::sign)(fa))
{
if(fa < 0)
{
while(((boost::math::sign)(fb) == (boost::math::sign)(fa)) && (a != b))
{
if(count == 0)
return policies::raise_evaluation_error(function, "Unable to bracket root, last nearest value was %1%", b, policy_type());
a = b;
fa = fb;
b *= multiplier;
if(b > max_bound)
b = max_bound;
fb = f(b);
--count;
BOOST_MATH_INSTRUMENT_CODE("a = " << a << " b = " << b << " fa = " << fa << " fb = " << fb << " count = " << count);
}
}
else
{
while(((boost::math::sign)(fb) == (boost::math::sign)(fa)) && (a != b))
{
if(fabs(a) < tools::min_value<value_type>())
{
max_iter -= count;
max_iter += 1;
return 0;
}
if(count == 0)
return policies::raise_evaluation_error(function, "Unable to bracket root, last nearest value was %1%", a, policy_type());
b = a;
fb = fa;
a /= multiplier;
if(a < min_bound)
a = min_bound;
fa = f(a);
--count;
BOOST_MATH_INSTRUMENT_CODE("a = " << a << " b = " << b << " fa = " << fa << " fb = " << fb << " count = " << count);
}
}
}
max_iter -= count;
if(fa == 0)
return a;
if(fb == 0)
return b;
if(a == b)
return b;  
adjust_bounds(a, b, tol);
if(a < tools::min_value<value_type>())
a = tools::min_value<value_type>();
std::pair<value_type, value_type> r = toms748_solve(f, a, b, fa, fb, tol, count, policy_type());
max_iter += count;
BOOST_MATH_INSTRUMENT_CODE("max_iter = " << max_iter << " count = " << count);
return (r.first + r.second) / 2;
}
template <class Dist>
inline typename Dist::value_type round_to_floor(const Dist& d, typename Dist::value_type result, typename Dist::value_type p, bool c)
{
BOOST_MATH_STD_USING
typename Dist::value_type cc = ceil(result);
typename Dist::value_type pp = cc <= support(d).second ? c ? cdf(complement(d, cc)) : cdf(d, cc) : 1;
if(pp == p)
result = cc;
else
result = floor(result);
while(result != 0)
{
cc = result - 1;
if(cc < support(d).first)
break;
pp = c ? cdf(complement(d, cc)) : cdf(d, cc);
if(pp == p)
result = cc;
else if(c ? pp > p : pp < p)
break;
result -= 1;
}

return result;
}

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)
#endif

template <class Dist>
inline typename Dist::value_type round_to_ceil(const Dist& d, typename Dist::value_type result, typename Dist::value_type p, bool c)
{
BOOST_MATH_STD_USING
typename Dist::value_type cc = floor(result);
typename Dist::value_type pp = cc >= support(d).first ? c ? cdf(complement(d, cc)) : cdf(d, cc) : 0;
if(pp == p)
result = cc;
else
result = ceil(result);
while(true)
{
cc = result + 1;
if(cc > support(d).second)
break;
pp = c ? cdf(complement(d, cc)) : cdf(d, cc);
if(pp == p)
result = cc;
else if(c ? pp < p : pp > p)
break;
result += 1;
}

return result;
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
template <class Dist>
inline typename Dist::value_type 
inverse_discrete_quantile(
const Dist& dist,
typename Dist::value_type p,
bool c,
const typename Dist::value_type& guess,
const typename Dist::value_type& multiplier,
const typename Dist::value_type& adder,
const policies::discrete_quantile<policies::real>&,
boost::uintmax_t& max_iter)
{
if(p > 0.5)
{
p = 1 - p;
c = !c;
}
typename Dist::value_type pp = c ? 1 - p : p;
if(pp <= pdf(dist, 0))
return 0;
return do_inverse_discrete_quantile(
dist, 
p, 
c,
guess, 
multiplier, 
adder, 
tools::eps_tolerance<typename Dist::value_type>(policies::digits<typename Dist::value_type, typename Dist::policy_type>()),
max_iter);
}

template <class Dist>
inline typename Dist::value_type 
inverse_discrete_quantile(
const Dist& dist,
const typename Dist::value_type& p,
bool c,
const typename Dist::value_type& guess,
const typename Dist::value_type& multiplier,
const typename Dist::value_type& adder,
const policies::discrete_quantile<policies::integer_round_outwards>&,
boost::uintmax_t& max_iter)
{
typedef typename Dist::value_type value_type;
BOOST_MATH_STD_USING
typename Dist::value_type pp = c ? 1 - p : p;
if(pp <= pdf(dist, 0))
return 0;
if(pp < 0.5f)
return round_to_floor(dist, do_inverse_discrete_quantile(
dist, 
p, 
c,
(guess < 1 ? value_type(1) : (value_type)floor(guess)), 
multiplier, 
adder, 
tools::equal_floor(),
max_iter), p, c);
return round_to_ceil(dist, do_inverse_discrete_quantile(
dist, 
p, 
c,
(value_type)ceil(guess), 
multiplier, 
adder, 
tools::equal_ceil(),
max_iter), p, c);
}

template <class Dist>
inline typename Dist::value_type 
inverse_discrete_quantile(
const Dist& dist,
const typename Dist::value_type& p,
bool c,
const typename Dist::value_type& guess,
const typename Dist::value_type& multiplier,
const typename Dist::value_type& adder,
const policies::discrete_quantile<policies::integer_round_inwards>&,
boost::uintmax_t& max_iter)
{
typedef typename Dist::value_type value_type;
BOOST_MATH_STD_USING
typename Dist::value_type pp = c ? 1 - p : p;
if(pp <= pdf(dist, 0))
return 0;
if(pp < 0.5f)
return round_to_ceil(dist, do_inverse_discrete_quantile(
dist, 
p, 
c,
ceil(guess), 
multiplier, 
adder, 
tools::equal_ceil(),
max_iter), p, c);
return round_to_floor(dist, do_inverse_discrete_quantile(
dist, 
p, 
c,
(guess < 1 ? value_type(1) : floor(guess)), 
multiplier, 
adder, 
tools::equal_floor(),
max_iter), p, c);
}

template <class Dist>
inline typename Dist::value_type 
inverse_discrete_quantile(
const Dist& dist,
const typename Dist::value_type& p,
bool c,
const typename Dist::value_type& guess,
const typename Dist::value_type& multiplier,
const typename Dist::value_type& adder,
const policies::discrete_quantile<policies::integer_round_down>&,
boost::uintmax_t& max_iter)
{
typedef typename Dist::value_type value_type;
BOOST_MATH_STD_USING
typename Dist::value_type pp = c ? 1 - p : p;
if(pp <= pdf(dist, 0))
return 0;
return round_to_floor(dist, do_inverse_discrete_quantile(
dist, 
p, 
c,
(guess < 1 ? value_type(1) : floor(guess)), 
multiplier, 
adder, 
tools::equal_floor(),
max_iter), p, c);
}

template <class Dist>
inline typename Dist::value_type 
inverse_discrete_quantile(
const Dist& dist,
const typename Dist::value_type& p,
bool c,
const typename Dist::value_type& guess,
const typename Dist::value_type& multiplier,
const typename Dist::value_type& adder,
const policies::discrete_quantile<policies::integer_round_up>&,
boost::uintmax_t& max_iter)
{
BOOST_MATH_STD_USING
typename Dist::value_type pp = c ? 1 - p : p;
if(pp <= pdf(dist, 0))
return 0;
return round_to_ceil(dist, do_inverse_discrete_quantile(
dist, 
p, 
c,
ceil(guess), 
multiplier, 
adder, 
tools::equal_ceil(),
max_iter), p, c);
}

template <class Dist>
inline typename Dist::value_type 
inverse_discrete_quantile(
const Dist& dist,
const typename Dist::value_type& p,
bool c,
const typename Dist::value_type& guess,
const typename Dist::value_type& multiplier,
const typename Dist::value_type& adder,
const policies::discrete_quantile<policies::integer_round_nearest>&,
boost::uintmax_t& max_iter)
{
typedef typename Dist::value_type value_type;
BOOST_MATH_STD_USING
typename Dist::value_type pp = c ? 1 - p : p;
if(pp <= pdf(dist, 0))
return 0;
return round_to_floor(dist, do_inverse_discrete_quantile(
dist, 
p, 
c,
(guess < 0.5f ? value_type(1.5f) : floor(guess + 0.5f) + 0.5f), 
multiplier, 
adder, 
tools::equal_nearest_integer(),
max_iter) + 0.5f, p, c);
}

}}} 

#endif 

