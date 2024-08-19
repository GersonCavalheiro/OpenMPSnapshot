#ifndef  BOOST_SERIALIZATION_MAP_HPP
#define BOOST_SERIALIZATION_MAP_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <map>

#include <boost/config.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/detail/stack_constructor.hpp>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/move/utility_core.hpp>

namespace boost {
namespace serialization {


template<class Archive, class Container>
inline void load_map_collection(Archive & ar, Container &s)
{
s.clear();
const boost::serialization::library_version_type library_version(
ar.get_library_version()
);
item_version_type item_version(0);
collection_size_type count;
ar >> BOOST_SERIALIZATION_NVP(count);
if(boost::serialization::library_version_type(3) < library_version){
ar >> BOOST_SERIALIZATION_NVP(item_version);
}
typename Container::iterator hint;
hint = s.begin();
while(count-- > 0){
typedef typename Container::value_type type;
detail::stack_construct<Archive, type> t(ar, item_version);
ar >> boost::serialization::make_nvp("item", t.reference());
typename Container::iterator result =
s.insert(hint, boost::move(t.reference()));
ar.reset_object_address(& (result->second), & t.reference().second);
hint = result;
++hint;
}
}

template<class Archive, class Type, class Key, class Compare, class Allocator >
inline void save(
Archive & ar,
const std::map<Key, Type, Compare, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::save_collection<
Archive,
std::map<Key, Type, Compare, Allocator>
>(ar, t);
}

template<class Archive, class Type, class Key, class Compare, class Allocator >
inline void load(
Archive & ar,
std::map<Key, Type, Compare, Allocator> &t,
const unsigned int 
){
load_map_collection(ar, t);
}

template<class Archive, class Type, class Key, class Compare, class Allocator >
inline void serialize(
Archive & ar,
std::map<Key, Type, Compare, Allocator> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

template<class Archive, class Type, class Key, class Compare, class Allocator >
inline void save(
Archive & ar,
const std::multimap<Key, Type, Compare, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::save_collection<
Archive,
std::multimap<Key, Type, Compare, Allocator>
>(ar, t);
}

template<class Archive, class Type, class Key, class Compare, class Allocator >
inline void load(
Archive & ar,
std::multimap<Key, Type, Compare, Allocator> &t,
const unsigned int 
){
load_map_collection(ar, t);
}

template<class Archive, class Type, class Key, class Compare, class Allocator >
inline void serialize(
Archive & ar,
std::multimap<Key, Type, Compare, Allocator> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#endif 
