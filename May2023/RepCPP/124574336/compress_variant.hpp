





#ifndef BOOST_GEOMETRY_UTIL_COMPRESS_VARIANT_HPP
#define BOOST_GEOMETRY_UTIL_COMPRESS_VARIANT_HPP

#include <boost/config/pragma_message.hpp>
#if !defined(BOOST_ALLOW_DEPRECATED_HEADERS)
BOOST_PRAGMA_MESSAGE("This header is deprecated.")
#endif

#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/variant/variant_fwd.hpp>


namespace boost { namespace geometry
{


namespace detail
{

template <typename Variant>
struct unique_types:
boost::mpl::fold<
typename boost::mpl::reverse_fold<
typename Variant::types,
boost::mpl::set<>,
boost::mpl::insert<
boost::mpl::placeholders::_1,
boost::mpl::placeholders::_2
>
>::type,
boost::mpl::vector<>,
boost::mpl::push_back
<
boost::mpl::placeholders::_1, boost::mpl::placeholders::_2
>
>
{};

template <typename Types>
struct variant_or_single:
boost::mpl::if_<
boost::mpl::equal_to<
boost::mpl::size<Types>,
boost::mpl::int_<1>
>,
typename boost::mpl::front<Types>::type,
typename make_variant_over<Types>::type
>
{};

} 




template <typename Variant>
struct compress_variant:
detail::variant_or_single<
typename detail::unique_types<Variant>::type
>
{};


}} 


#endif 
