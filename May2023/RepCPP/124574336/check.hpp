#ifndef BOOST_ARCHIVE_DETAIL_CHECK_HPP
#define BOOST_ARCHIVE_DETAIL_CHECK_HPP

#if defined(_MSC_VER)
# pragma once
#pragma inline_depth(255)
#pragma inline_recursion(on)
#endif

#if defined(__MWERKS__)
#pragma inline_depth(255)
#endif




#include <boost/config.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_const.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/greater.hpp>
#include <boost/mpl/assert.hpp>

#include <boost/serialization/static_warning.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/wrapper.hpp>

namespace boost {
namespace archive {
namespace detail {


template<class T>
inline void check_object_level(){
typedef
typename mpl::greater_equal<
serialization::implementation_level< T >,
mpl::int_<serialization::primitive_type>
>::type typex;

BOOST_STATIC_ASSERT(typex::value);
}

template<class T>
inline void check_object_versioning(){
typedef
typename mpl::or_<
typename mpl::greater<
serialization::implementation_level< T >,
mpl::int_<serialization::object_serializable>
>,
typename mpl::equal_to<
serialization::version< T >,
mpl::int_<0>
>
> typex;
BOOST_STATIC_ASSERT(typex::value);
}

template<class T>
inline void check_object_tracking(){
BOOST_STATIC_ASSERT(! boost::is_const< T >::value);
typedef typename mpl::equal_to<
serialization::tracking_level< T >,
mpl::int_<serialization::track_never>
>::type typex;


BOOST_STATIC_WARNING(typex::value);
}


template<class T>
inline void check_pointer_level(){
typedef
typename mpl::or_<
typename mpl::greater<
serialization::implementation_level< T >,
mpl::int_<serialization::object_serializable>
>,
typename mpl::not_<
typename mpl::equal_to<
serialization::tracking_level< T >,
mpl::int_<serialization::track_selectively>
>
>
> typex;



BOOST_STATIC_WARNING(typex::value);
}

template<class T>
void inline check_pointer_tracking(){
typedef typename mpl::greater<
serialization::tracking_level< T >,
mpl::int_<serialization::track_never>
>::type typex;
BOOST_STATIC_WARNING(typex::value);
}

template<class T>
inline void check_const_loading(){
typedef
typename mpl::or_<
typename boost::serialization::is_wrapper< T >,
typename mpl::not_<
typename boost::is_const< T >
>
>::type typex;
BOOST_STATIC_ASSERT(typex::value);
}

} 
} 
} 

#endif 
