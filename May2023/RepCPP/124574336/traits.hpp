#ifndef BOOST_SERIALIZATION_TRAITS_HPP
#define BOOST_SERIALIZATION_TRAITS_HPP

#if defined(_MSC_VER)
# pragma once
#endif






#include <boost/config.hpp>
#include <boost/static_assert.hpp>

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool_fwd.hpp>
#include <boost/serialization/level_enum.hpp>
#include <boost/serialization/tracking_enum.hpp>

namespace boost {
namespace serialization {

struct basic_traits {};

template <class T>
struct extended_type_info_impl;

template<
class T,
int Level,
int Tracking,
unsigned int Version = 0,
class ETII = extended_type_info_impl< T >,
class Wrapper = mpl::false_
>
struct traits : public basic_traits {
BOOST_STATIC_ASSERT(Version == 0 || Level >= object_class_info);
BOOST_STATIC_ASSERT(Tracking == track_never || Level >= object_serializable);
typedef typename mpl::int_<Level> level;
typedef typename mpl::int_<Tracking> tracking;
typedef typename mpl::int_<Version> version;
typedef ETII type_info_implementation;
typedef Wrapper is_wrapper;
};

} 
} 

#endif 
