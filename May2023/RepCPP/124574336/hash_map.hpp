#ifndef  BOOST_SERIALIZATION_HASH_MAP_HPP
#define BOOST_SERIALIZATION_HASH_MAP_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_HAS_HASH
#include BOOST_HASH_MAP_HEADER

#include <boost/serialization/utility.hpp>
#include <boost/serialization/hash_collections_save_imp.hpp>
#include <boost/serialization/hash_collections_load_imp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/detail/stack_constructor.hpp>
#include <boost/move/utility_core.hpp>

namespace boost {
namespace serialization {

namespace stl {

template<class Archive, class Container>
struct archive_input_hash_map
{
inline void operator()(
Archive &ar,
Container &s,
const unsigned int v
){
typedef typename Container::value_type type;
detail::stack_construct<Archive, type> t(ar, v);
ar >> boost::serialization::make_nvp("item", t.reference());
std::pair<typename Container::const_iterator, bool> result =
s.insert(boost::move(t.reference()));
if(result.second){
ar.reset_object_address(
& (result.first->second),
& t.reference().second
);
}
}
};

template<class Archive, class Container>
struct archive_input_hash_multimap
{
inline void operator()(
Archive &ar,
Container &s,
const unsigned int v
){
typedef typename Container::value_type type;
detail::stack_construct<Archive, type> t(ar, v);
ar >> boost::serialization::make_nvp("item", t.reference());
typename Container::const_iterator result
= s.insert(boost::move(t.reference()));
ar.reset_object_address(
& result->second,
& t.reference()
);
}
};

} 

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
const BOOST_STD_EXTENSION_NAMESPACE::hash_map<
Key, T, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::save_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_map<
Key, T, HashFcn, EqualKey, Allocator
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
inline void load(
Archive & ar,
BOOST_STD_EXTENSION_NAMESPACE::hash_map<
Key, T, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::load_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_map<
Key, T, HashFcn, EqualKey, Allocator
>,
boost::serialization::stl::archive_input_hash_map<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_map<
Key, T, HashFcn, EqualKey, Allocator
>
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
BOOST_STD_EXTENSION_NAMESPACE::hash_map<
Key, T, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

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
const BOOST_STD_EXTENSION_NAMESPACE::hash_multimap<
Key, T, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::save_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_multimap<
Key, T, HashFcn, EqualKey, Allocator
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
inline void load(
Archive & ar,
BOOST_STD_EXTENSION_NAMESPACE::hash_multimap<
Key, T, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::load_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_multimap<
Key, T, HashFcn, EqualKey, Allocator
>,
boost::serialization::stl::archive_input_hash_multimap<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_multimap<
Key, T, HashFcn, EqualKey, Allocator
>
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
BOOST_STD_EXTENSION_NAMESPACE::hash_multimap<
Key, T, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#endif 
#endif 
