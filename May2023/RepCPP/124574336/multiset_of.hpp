

#ifndef BOOST_BIMAP_MULTISET_OF_HPP
#define BOOST_BIMAP_MULTISET_OF_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/detail/user_interface_config.hpp>

#include <functional>
#include <boost/mpl/bool.hpp>

#include <boost/concept_check.hpp>

#include <boost/bimap/detail/concept_tags.hpp>

#include <boost/bimap/tags/support/value_type_of.hpp>

#include <boost/bimap/detail/generate_index_binder.hpp>
#include <boost/bimap/detail/generate_view_binder.hpp>
#include <boost/bimap/detail/generate_relation_binder.hpp>

#include <boost/multi_index/ordered_index.hpp>

#include <boost/bimap/views/multimap_view.hpp>
#include <boost/bimap/views/multiset_view.hpp>

namespace boost {
namespace bimaps {



template
<
class KeyType,
class KeyCompare = std::less< BOOST_DEDUCED_TYPENAME
::boost::bimaps::tags::support::value_type_of<KeyType>::type >
>
struct multiset_of : public ::boost::bimaps::detail::set_type_of_tag
{
typedef KeyType user_type;

typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::tags::support::
value_type_of<user_type>::type value_type;

typedef KeyCompare key_compare;

struct lazy_concept_checked
{
BOOST_CLASS_REQUIRE ( value_type,
boost, AssignableConcept );

BOOST_CLASS_REQUIRE4( key_compare, bool, value_type, value_type,
boost, BinaryFunctionConcept );

typedef multiset_of type;
};

BOOST_BIMAP_GENERATE_INDEX_BINDER_1CP(

multi_index::ordered_non_unique,

key_compare
)

BOOST_BIMAP_GENERATE_MAP_VIEW_BINDER(

views::multimap_view
)

BOOST_BIMAP_GENERATE_SET_VIEW_BINDER(

views::multiset_view
)

typedef mpl::bool_<false> mutable_key;
};




template< class KeyCompare = std::less< _relation > >
struct multiset_of_relation : public ::boost::bimaps::detail::set_type_of_relation_tag
{
typedef KeyCompare key_compare;


BOOST_BIMAP_GENERATE_RELATION_BINDER_1CP(

multiset_of,

key_compare
)

typedef mpl::bool_<false>  left_mutable_key;
typedef mpl::bool_<false> right_mutable_key;
};

} 
} 


#endif 
