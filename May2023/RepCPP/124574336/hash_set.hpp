#ifndef  BOOST_SERIALIZATION_HASH_SET_HPP
#define BOOST_SERIALIZATION_HASH_SET_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_HAS_HASH
#include BOOST_HASH_SET_HEADER

#include <boost/serialization/hash_collections_save_imp.hpp>
#include <boost/serialization/hash_collections_load_imp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/detail/stack_constructor.hpp>
#include <boost/move/utility_core.hpp>

namespace boost {
namespace serialization {

namespace stl {

template<class Archive, class Container>
struct archive_input_hash_set
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
if(result.second)
ar.reset_object_address(& (* result.first), & t.reference());
}
};

template<class Archive, class Container>
struct archive_input_hash_multiset
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
ar.reset_object_address(& (* result), & t.reference());
}
};

} 

template<
class Archive,
class Key,
class HashFcn,
class EqualKey,
class Allocator
>
inline void save(
Archive & ar,
const BOOST_STD_EXTENSION_NAMESPACE::hash_set<
Key, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::save_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_set<
Key, HashFcn, EqualKey, Allocator
>
>(ar, t);
}

template<
class Archive,
class Key,
class HashFcn,
class EqualKey,
class Allocator
>
inline void load(
Archive & ar,
BOOST_STD_EXTENSION_NAMESPACE::hash_set<
Key, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::load_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_set<
Key, HashFcn, EqualKey, Allocator
>,
boost::serialization::stl::archive_input_hash_set<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_set<
Key, HashFcn, EqualKey, Allocator
>
>
>(ar, t);
}

template<
class Archive,
class Key,
class HashFcn,
class EqualKey,
class Allocator
>
inline void serialize(
Archive & ar,
BOOST_STD_EXTENSION_NAMESPACE::hash_set<
Key, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

template<
class Archive,
class Key,
class HashFcn,
class EqualKey,
class Allocator
>
inline void save(
Archive & ar,
const BOOST_STD_EXTENSION_NAMESPACE::hash_multiset<
Key, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::save_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_multiset<
Key, HashFcn, EqualKey, Allocator
>
>(ar, t);
}

template<
class Archive,
class Key,
class HashFcn,
class EqualKey,
class Allocator
>
inline void load(
Archive & ar,
BOOST_STD_EXTENSION_NAMESPACE::hash_multiset<
Key, HashFcn, EqualKey, Allocator
> &t,
const unsigned int file_version
){
boost::serialization::stl::load_hash_collection<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_multiset<
Key, HashFcn, EqualKey, Allocator
>,
boost::serialization::stl::archive_input_hash_multiset<
Archive,
BOOST_STD_EXTENSION_NAMESPACE::hash_multiset<
Key, HashFcn, EqualKey, Allocator
>
>
>(ar, t);
}

template<
class Archive,
class Key,
class HashFcn,
class EqualKey,
class Allocator
>
inline void serialize(
Archive & ar,
BOOST_STD_EXTENSION_NAMESPACE::hash_multiset<
Key, HashFcn, EqualKey, Allocator
> & t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#include <boost/serialization/collection_traits.hpp>

BOOST_SERIALIZATION_COLLECTION_TRAITS(BOOST_STD_EXTENSION_NAMESPACE::hash_set)
BOOST_SERIALIZATION_COLLECTION_TRAITS(BOOST_STD_EXTENSION_NAMESPACE::hash_multiset)

#endif 
#endif 
