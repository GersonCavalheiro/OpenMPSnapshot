





#ifndef BOOST_GEOMETRY_UTIL_TRANSFORM_VARIANT_HPP
#define BOOST_GEOMETRY_UTIL_TRANSFORM_VARIANT_HPP


#include <boost/config/pragma_message.hpp>
#if !defined(BOOST_ALLOW_DEPRECATED_HEADERS)
BOOST_PRAGMA_MESSAGE("This header is deprecated.")
#endif


#include <boost/mpl/transform.hpp>
#include <boost/variant/variant_fwd.hpp>


namespace boost { namespace geometry
{



template <typename Sequence, typename Op, typename In = boost::mpl::na>
struct transform_variant:
make_variant_over<
typename boost::mpl::transform<
Sequence,
Op,
In
>::type
>
{};



template <BOOST_VARIANT_ENUM_PARAMS(typename T), typename Op>
struct transform_variant<variant<BOOST_VARIANT_ENUM_PARAMS(T)>, Op, boost::mpl::na> :
make_variant_over<
typename boost::mpl::transform<
typename variant<BOOST_VARIANT_ENUM_PARAMS(T)>::types,
Op
>::type
>
{};


}} 


#endif 
