

#ifndef BOOST_BIMAP_PROPERTY_MAP_SET_SUPPORT_HPP
#define BOOST_BIMAP_PROPERTY_MAP_SET_SUPPORT_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/property_map/property_map.hpp>
#include <boost/bimap/set_of.hpp>
#include <boost/bimap/support/data_type_by.hpp>
#include <boost/bimap/support/key_type_by.hpp>

#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {

template< class Tag, class Bimap >
struct property_traits< ::boost::bimaps::views::map_view<Tag,Bimap> >
{
typedef BOOST_DEDUCED_TYPENAME
::boost::bimaps::support::data_type_by<Tag,Bimap>::type value_type;
typedef BOOST_DEDUCED_TYPENAME
::boost::bimaps::support:: key_type_by<Tag,Bimap>::type   key_type;

typedef readable_property_map_tag category;
};


template< class Tag, class Bimap >
const BOOST_DEDUCED_TYPENAME ::boost::bimaps::support::data_type_by<Tag,Bimap>::type &
get(const ::boost::bimaps::views::map_view<Tag,Bimap> & m,
const BOOST_DEDUCED_TYPENAME
::boost::bimaps::support::key_type_by<Tag,Bimap>::type & key)
{
return m.at(key);
}

} 

#endif 

#endif 
