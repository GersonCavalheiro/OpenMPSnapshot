

#ifndef BOOST_BIMAP_UNORDERED_SET_OF_HPP
#define BOOST_BIMAP_UNORDERED_SET_OF_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/detail/user_interface_config.hpp>

#include <functional>
#include <boost/functional/hash.hpp>
#include <boost/mpl/bool.hpp>

#include <boost/concept_check.hpp>

#include <boost/bimap/detail/concept_tags.hpp>

#include <boost/bimap/tags/support/value_type_of.hpp>

#include <boost/bimap/detail/generate_index_binder.hpp>
#include <boost/bimap/detail/generate_view_binder.hpp>
#include <boost/bimap/detail/generate_relation_binder.hpp>

#include <boost/multi_index/hashed_index.hpp>

#include <boost/bimap/views/unordered_map_view.hpp>
#include <boost/bimap/views/unordered_set_view.hpp>

namespace boost {
namespace bimaps {



template
<
class KeyType,
class HashFunctor   = hash< BOOST_DEDUCED_TYPENAME 
::boost::bimaps::tags::support::value_type_of<KeyType>::type >,
class EqualKey      = std::equal_to< BOOST_DEDUCED_TYPENAME 
::boost::bimaps::tags::support::value_type_of<KeyType>::type >
>
struct unordered_set_of : public ::boost::bimaps::detail::set_type_of_tag
{
typedef KeyType user_type;

typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::tags::support::
value_type_of<user_type>::type value_type;

typedef HashFunctor     hasher;

typedef EqualKey        key_equal;

struct lazy_concept_checked
{
BOOST_CLASS_REQUIRE ( value_type,
boost, AssignableConcept );

BOOST_CLASS_REQUIRE3( hasher, std::size_t, value_type,
boost, UnaryFunctionConcept );

BOOST_CLASS_REQUIRE4( key_equal, bool, value_type, value_type,
boost, BinaryFunctionConcept );

typedef unordered_set_of type; 
};

BOOST_BIMAP_GENERATE_INDEX_BINDER_2CP(

multi_index::hashed_unique,

hasher,
key_equal
)

BOOST_BIMAP_GENERATE_MAP_VIEW_BINDER(

views::unordered_map_view
)

BOOST_BIMAP_GENERATE_SET_VIEW_BINDER(

views::unordered_set_view
)

typedef mpl::bool_<false> mutable_key;
};




template
<
class HashFunctor   = hash< _relation >,
class EqualKey      = std::equal_to< _relation >
>
struct unordered_set_of_relation : public ::boost::bimaps::detail::set_type_of_relation_tag
{
typedef HashFunctor     hasher;

typedef EqualKey        key_equal;


BOOST_BIMAP_GENERATE_RELATION_BINDER_2CP(

unordered_set_of,

hasher,
key_equal
)

typedef mpl::bool_<false>  left_mutable_key;
typedef mpl::bool_<false> right_mutable_key;
};


} 
} 


#endif 

