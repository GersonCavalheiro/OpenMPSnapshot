
#ifndef BOOST_LEXICAL_CAST_DETAIL_CONVERTER_NUMERIC_HPP
#define BOOST_LEXICAL_CAST_DETAIL_CONVERTER_NUMERIC_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <boost/limits.hpp>
#include <boost/type_traits/type_identity.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_float.hpp>

#include <boost/numeric/conversion/cast.hpp>

namespace boost { namespace detail {

template <class Source >
struct detect_precision_loss
{
typedef Source source_type;
typedef boost::numeric::Trunc<Source> Rounder;
typedef BOOST_DEDUCED_TYPENAME conditional<
boost::is_arithmetic<Source>::value, Source, Source const&
>::type argument_type ;

static inline source_type nearbyint(argument_type s, bool& is_ok) BOOST_NOEXCEPT {
const source_type near_int = Rounder::nearbyint(s);
if (near_int && is_ok) {
const source_type orig_div_round = s / near_int;
const source_type eps = std::numeric_limits<source_type>::epsilon();

is_ok = !((orig_div_round > 1 ? orig_div_round - 1 : 1 - orig_div_round) > eps);
}

return s;
}

typedef typename Rounder::round_style round_style;
};

template <typename Base, class Source>
struct fake_precision_loss: public Base
{
typedef Source source_type ;
typedef BOOST_DEDUCED_TYPENAME conditional<
boost::is_arithmetic<Source>::value, Source, Source const&
>::type argument_type ;

static inline source_type nearbyint(argument_type s, bool& ) BOOST_NOEXCEPT {
return s;
}
};

struct nothrow_overflow_handler
{
inline bool operator() ( boost::numeric::range_check_result r ) const BOOST_NOEXCEPT {
return (r == boost::numeric::cInRange);
}
};

template <typename Target, typename Source>
inline bool noexcept_numeric_convert(const Source& arg, Target& result) BOOST_NOEXCEPT {
typedef boost::numeric::converter<
Target,
Source,
boost::numeric::conversion_traits<Target, Source >,
nothrow_overflow_handler,
detect_precision_loss<Source >
> converter_orig_t;

typedef BOOST_DEDUCED_TYPENAME boost::conditional<
boost::is_base_of< detect_precision_loss<Source >, converter_orig_t >::value,
converter_orig_t,
fake_precision_loss<converter_orig_t, Source>
>::type converter_t;

bool res = nothrow_overflow_handler()(converter_t::out_of_range(arg));
result = converter_t::low_level_convert(converter_t::nearbyint(arg, res));
return res;
}

template <typename Target, typename Source>
struct lexical_cast_dynamic_num_not_ignoring_minus
{
static inline bool try_convert(const Source &arg, Target& result) BOOST_NOEXCEPT {
return noexcept_numeric_convert<Target, Source >(arg, result);
}
};

template <typename Target, typename Source>
struct lexical_cast_dynamic_num_ignoring_minus
{
static inline bool try_convert(const Source &arg, Target& result) BOOST_NOEXCEPT {
typedef BOOST_DEDUCED_TYPENAME boost::conditional<
boost::is_float<Source>::value,
boost::type_identity<Source>,
boost::make_unsigned<Source>
>::type usource_lazy_t;
typedef BOOST_DEDUCED_TYPENAME usource_lazy_t::type usource_t;

if (arg < 0) {
const bool res = noexcept_numeric_convert<Target, usource_t>(0u - arg, result);
result = static_cast<Target>(0u - result);
return res;
} else {
return noexcept_numeric_convert<Target, usource_t>(arg, result);
}
}
};


template <typename Target, typename Source>
struct dynamic_num_converter_impl
{
static inline bool try_convert(const Source &arg, Target& result) BOOST_NOEXCEPT {
typedef BOOST_DEDUCED_TYPENAME boost::conditional<
boost::is_unsigned<Target>::value &&
(boost::is_signed<Source>::value || boost::is_float<Source>::value) &&
!(boost::is_same<Source, bool>::value) &&
!(boost::is_same<Target, bool>::value),
lexical_cast_dynamic_num_ignoring_minus<Target, Source>,
lexical_cast_dynamic_num_not_ignoring_minus<Target, Source>
>::type caster_type;

return caster_type::try_convert(arg, result);
}
};

}} 

#endif 

