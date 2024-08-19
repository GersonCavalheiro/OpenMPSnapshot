





#ifndef BOOST_GEOMETRY_UTIL_COMBINE_IF_HPP
#define BOOST_GEOMETRY_UTIL_COMBINE_IF_HPP

#include <boost/config/pragma_message.hpp>
#if !defined(BOOST_ALLOW_DEPRECATED_HEADERS)
BOOST_PRAGMA_MESSAGE("This header is deprecated.")
#endif

#include <boost/mpl/bind.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/set.hpp>

namespace boost { namespace geometry
{

namespace util
{



template <typename Sequence1, typename Sequence2, typename Pred>
struct combine_if
{
struct combine
{
template <typename Result, typename T>
struct apply
{
typedef typename boost::mpl::fold<Sequence2, Result,
boost::mpl::if_
<
boost::mpl::bind
<
typename boost::mpl::lambda<Pred>::type,
T,
boost::mpl::_2
>,
boost::mpl::insert
<
boost::mpl::_1, boost::mpl::pair<T, boost::mpl::_2>
>,
boost::mpl::_1
>
>::type type;
};
};

typedef typename boost::mpl::fold
<
Sequence1, boost::mpl::set0<>, combine
>::type type;
};


} 

}} 

#endif 
