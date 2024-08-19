

#ifndef BOOST_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP
#define BOOST_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP

#include <iosfwd>
#include <ios>
#include <istream>
#include <boost/config.hpp>
#include <boost/limits.hpp>
#include <boost/assert.hpp>
#include <boost/random/detail/config.hpp>
#include <boost/random/detail/operators.hpp>
#include <boost/random/detail/uniform_int_float.hpp>
#include <boost/random/detail/signed_unsigned_tools.hpp>
#include <boost/random/traits.hpp>
#include <boost/type_traits/integral_constant.hpp>
#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
#include <boost/type_traits/conditional.hpp>
#endif

namespace boost {
namespace random {
namespace detail {


#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4723)
#endif

template<class Engine, class T>
T generate_uniform_int(
Engine& eng, T min_value, T max_value,
boost::true_type )
{
typedef T result_type;
typedef typename boost::random::traits::make_unsigned_or_unbounded<T>::type range_type;
typedef typename Engine::result_type base_result;
typedef typename boost::random::traits::make_unsigned_or_unbounded<base_result>::type base_unsigned;
const range_type range = random::detail::subtract<result_type>()(max_value, min_value);
const base_result bmin = (eng.min)();
const base_unsigned brange =
random::detail::subtract<base_result>()((eng.max)(), (eng.min)());

if(range == 0) {
return min_value;    
} else if(brange == range) {
base_unsigned v = random::detail::subtract<base_result>()(eng(), bmin);
return random::detail::add<base_unsigned, result_type>()(v, min_value);
} else if(brange < range) {
for(;;) {

range_type limit;
if(range == (std::numeric_limits<range_type>::max)()) {
limit = range/(range_type(brange)+1);
if(range % (range_type(brange)+1) == range_type(brange))
++limit;
} else {
limit = (range+1)/(range_type(brange)+1);
}

range_type result = range_type(0);
range_type mult = range_type(1);

while(mult <= limit) {
result += static_cast<range_type>(static_cast<range_type>(random::detail::subtract<base_result>()(eng(), bmin)) * mult);

if(mult * range_type(brange) == range - mult + 1) {
return(result);
}

mult *= range_type(brange)+range_type(1);
}

range_type result_increment =
generate_uniform_int(
eng,
static_cast<range_type>(0),
static_cast<range_type>(range/mult),
boost::true_type());
if(std::numeric_limits<range_type>::is_bounded && ((std::numeric_limits<range_type>::max)() / mult < result_increment)) {
continue;
}
result_increment *= mult;
result += result_increment;
if(result < result_increment) {
continue;
}
if(result > range) {
continue;
}
return random::detail::add<range_type, result_type>()(result, min_value);
}
} else {                   
#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
typedef typename conditional<
std::numeric_limits<range_type>::is_specialized && std::numeric_limits<base_unsigned>::is_specialized
&& (std::numeric_limits<range_type>::digits >= std::numeric_limits<base_unsigned>::digits),
range_type, base_unsigned>::type mixed_range_type;
#else
typedef base_unsigned mixed_range_type;
#endif

mixed_range_type bucket_size;

if(brange == (std::numeric_limits<base_unsigned>::max)()) {
bucket_size = static_cast<mixed_range_type>(brange) / (static_cast<mixed_range_type>(range)+1);
if(static_cast<mixed_range_type>(brange) % (static_cast<mixed_range_type>(range)+1) == static_cast<mixed_range_type>(range)) {
++bucket_size;
}
} else {
bucket_size = static_cast<mixed_range_type>(brange + 1) / (static_cast<mixed_range_type>(range)+1);
}
for(;;) {
mixed_range_type result =
random::detail::subtract<base_result>()(eng(), bmin);
result /= bucket_size;
if(result <= static_cast<mixed_range_type>(range))
return random::detail::add<mixed_range_type, result_type>()(result, min_value);
}
}
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template<class Engine, class T>
inline T generate_uniform_int(
Engine& eng, T min_value, T max_value,
boost::false_type )
{
uniform_int_float<Engine> wrapper(eng);
return generate_uniform_int(wrapper, min_value, max_value, boost::true_type());
}

template<class Engine, class T>
inline T generate_uniform_int(Engine& eng, T min_value, T max_value)
{
typedef typename Engine::result_type base_result;
return generate_uniform_int(eng, min_value, max_value,
boost::random::traits::is_integral<base_result>());
}

}


template<class IntType = int>
class uniform_int_distribution
{
public:
typedef IntType input_type;
typedef IntType result_type;

class param_type
{
public:

typedef uniform_int_distribution distribution_type;


explicit param_type(
IntType min_arg = 0,
IntType max_arg = (std::numeric_limits<IntType>::max)())
: _min(min_arg), _max(max_arg)
{
BOOST_ASSERT(_min <= _max);
}


IntType a() const { return _min; }

IntType b() const { return _max; }


BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, param_type, parm)
{
os << parm._min << " " << parm._max;
return os;
}


BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, param_type, parm)
{
IntType min_in, max_in;
if(is >> min_in >> std::ws >> max_in) {
if(min_in <= max_in) {
parm._min = min_in;
parm._max = max_in;
} else {
is.setstate(std::ios_base::failbit);
}
}
return is;
}


BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(param_type, lhs, rhs)
{ return lhs._min == rhs._min && lhs._max == rhs._max; }


BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(param_type)

private:

IntType _min;
IntType _max;
};


explicit uniform_int_distribution(
IntType min_arg = 0,
IntType max_arg = (std::numeric_limits<IntType>::max)())
: _min(min_arg), _max(max_arg)
{
BOOST_ASSERT(min_arg <= max_arg);
}

explicit uniform_int_distribution(const param_type& parm)
: _min(parm.a()), _max(parm.b()) {}


IntType min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return _min; }

IntType max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return _max; }


IntType a() const { return _min; }

IntType b() const { return _max; }


param_type param() const { return param_type(_min, _max); }

void param(const param_type& parm)
{
_min = parm.a();
_max = parm.b();
}


void reset() { }


template<class Engine>
result_type operator()(Engine& eng) const
{ return detail::generate_uniform_int(eng, _min, _max); }


template<class Engine>
result_type operator()(Engine& eng, const param_type& parm) const
{ return detail::generate_uniform_int(eng, parm.a(), parm.b()); }


BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, uniform_int_distribution, ud)
{
os << ud.param();
return os;
}


BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, uniform_int_distribution, ud)
{
param_type parm;
if(is >> parm) {
ud.param(parm);
}
return is;
}


BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(uniform_int_distribution, lhs, rhs)
{ return lhs._min == rhs._min && lhs._max == rhs._max; }


BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(uniform_int_distribution)

private:
IntType _min;
IntType _max;
};

} 
} 

#endif 
