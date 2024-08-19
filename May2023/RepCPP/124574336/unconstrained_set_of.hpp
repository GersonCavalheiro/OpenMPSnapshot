

#ifndef BOOST_BIMAP_UNCONSTRAINED_SET_OF_HPP
#define BOOST_BIMAP_UNCONSTRAINED_SET_OF_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/detail/user_interface_config.hpp>

#include <boost/mpl/bool.hpp>

#include <boost/concept_check.hpp>

#include <boost/bimap/detail/concept_tags.hpp>

#include <boost/bimap/tags/support/value_type_of.hpp>

#include <boost/bimap/detail/generate_index_binder.hpp>
#include <boost/bimap/detail/generate_view_binder.hpp>
#include <boost/bimap/detail/generate_relation_binder.hpp>

#include <boost/bimap/views/unconstrained_map_view.hpp>
#include <boost/bimap/views/unconstrained_set_view.hpp>

namespace boost {
namespace bimaps {



template
<
class KeyType
>
struct unconstrained_set_of : public ::boost::bimaps::detail::set_type_of_tag
{
typedef KeyType user_type;

typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::tags::support::
value_type_of<user_type>::type value_type;

struct lazy_concept_checked
{
BOOST_CLASS_REQUIRE ( value_type,
boost, AssignableConcept );

typedef unconstrained_set_of type;
};

BOOST_BIMAP_GENERATE_INDEX_BINDER_FAKE

BOOST_BIMAP_GENERATE_MAP_VIEW_BINDER(

views::unconstrained_map_view
)

BOOST_BIMAP_GENERATE_SET_VIEW_BINDER(

views::unconstrained_set_view
)

typedef mpl::bool_<true> mutable_key;
};



struct unconstrained_set_of_relation : public ::boost::bimaps::detail::set_type_of_relation_tag
{

BOOST_BIMAP_GENERATE_RELATION_BINDER_0CP(

unconstrained_set_of
)

typedef mpl::bool_<true>  left_mutable_key;
typedef mpl::bool_<true> right_mutable_key;
};

#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace detail {

template<class T>
struct is_unconstrained_set_of :
::boost::mpl::false_ {};

template<class T>
struct is_unconstrained_set_of< unconstrained_set_of<T> > :
::boost::mpl::true_ {};

} 

#endif 

} 
} 





#endif 

