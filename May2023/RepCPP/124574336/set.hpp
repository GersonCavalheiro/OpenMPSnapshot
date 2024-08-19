#ifndef  BOOST_SERIALIZATION_SET_HPP
#define BOOST_SERIALIZATION_SET_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <set>

#include <boost/config.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/detail/stack_constructor.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>

#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/move/utility_core.hpp>

namespace boost {
namespace serialization {

template<class Archive, class Container>
inline void load_set_collection(Archive & ar, Container &s)
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
const type * new_address = & (* result);
ar.reset_object_address(new_address, & t.reference());
hint = result;
}
}

template<class Archive, class Key, class Compare, class Allocator >
inline void save(
Archive & ar,
const std::set<Key, Compare, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::save_collection<
Archive, std::set<Key, Compare, Allocator>
>(ar, t);
}

template<class Archive, class Key, class Compare, class Allocator >
inline void load(
Archive & ar,
std::set<Key, Compare, Allocator> &t,
const unsigned int 
){
load_set_collection(ar, t);
}

template<class Archive, class Key, class Compare, class Allocator >
inline void serialize(
Archive & ar,
std::set<Key, Compare, Allocator> & t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

template<class Archive, class Key, class Compare, class Allocator >
inline void save(
Archive & ar,
const std::multiset<Key, Compare, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::save_collection<
Archive,
std::multiset<Key, Compare, Allocator>
>(ar, t);
}

template<class Archive, class Key, class Compare, class Allocator >
inline void load(
Archive & ar,
std::multiset<Key, Compare, Allocator> &t,
const unsigned int 
){
load_set_collection(ar, t);
}

template<class Archive, class Key, class Compare, class Allocator >
inline void serialize(
Archive & ar,
std::multiset<Key, Compare, Allocator> & t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#include <boost/serialization/collection_traits.hpp>

BOOST_SERIALIZATION_COLLECTION_TRAITS(std::set)
BOOST_SERIALIZATION_COLLECTION_TRAITS(std::multiset)

#endif 
