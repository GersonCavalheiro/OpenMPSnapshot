#ifndef  BOOST_SERIALIZATION_BOOST_SERIALIZATION_UNORDERED_MAP_HPP
#define BOOST_SERIALIZATION_BOOST_SERIALIZATION_UNORDERED_MAP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif




#include <boost/config.hpp>

#include <boost/unordered_map.hpp>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/unordered_collections_save_imp.hpp>
#include <boost/serialization/unordered_collections_load_imp.hpp>
#include <boost/serialization/archive_input_unordered_map.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost {
namespace serialization {

template<
class Archive,
class Key,
class T,
class HashFcn,
class EqualKey,
class Allocator
>
inline void save(
Archive & ar,
const boost::unordered_map<Key, T, HashFcn, EqualKey, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::save_unordered_collection<
Archive,
boost::unordered_map<Key, T, HashFcn, EqualKey, Allocator>
>(ar, t);
}

template<
class Archive,
class Key,
class T,
class HashFcn,
class EqualKey,
class Allocator
>
inline void load(
Archive & ar,
boost::unordered_map<Key, T, HashFcn, EqualKey, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::load_unordered_collection<
Archive,
boost::unordered_map<Key, T, HashFcn, EqualKey, Allocator>,
boost::serialization::stl::archive_input_unordered_map<
Archive,
boost::unordered_map<Key, T, HashFcn, EqualKey, Allocator>
>
>(ar, t);
}

template<
class Archive,
class Key,
class T,
class HashFcn,
class EqualKey,
class Allocator
>
inline void serialize(
Archive & ar,
boost::unordered_map<Key, T, HashFcn, EqualKey, Allocator> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

template<
class Archive,
class Key,
class HashFcn,
class T,
class EqualKey,
class Allocator
>
inline void save(
Archive & ar,
const boost::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::save_unordered_collection<
Archive,
boost::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator>
>(ar, t);
}

template<
class Archive,
class Key,
class T,
class HashFcn,
class EqualKey,
class Allocator
>
inline void load(
Archive & ar,
boost::unordered_multimap<
Key, T, HashFcn, EqualKey, Allocator
> &t,
const unsigned int 
){
boost::serialization::stl::load_unordered_collection<
Archive,
boost::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator>,
boost::serialization::stl::archive_input_unordered_multimap<
Archive,
boost::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator>
>
>(ar, t);
}

template<
class Archive,
class Key,
class T,
class HashFcn,
class EqualKey,
class Allocator
>
inline void serialize(
Archive & ar,
boost::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#endif 
