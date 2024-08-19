

#ifndef BOOST_BIMAP_DETAIL_CONCEPT_TAGS_HPP
#define BOOST_BIMAP_DETAIL_CONCEPT_TAGS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/mpl/identity.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/bool.hpp>

namespace boost {
namespace bimaps {
namespace detail {



struct set_type_of_tag          {};


struct set_type_of_relation_tag {};


struct side_based_tag : set_type_of_relation_tag {};

} 






#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

struct left_based : ::boost::bimaps::detail::side_based_tag
{
template< class Relation > struct bind_to { typedef void type; };

typedef mpl::bool_<true>  left_mutable_key;
typedef mpl::bool_<true> right_mutable_key;
};

struct right_based : ::boost::bimaps::detail::side_based_tag
{
template< class Relation > struct bind_to { typedef void type; };

typedef mpl::bool_<true>  left_mutable_key;
typedef mpl::bool_<true> right_mutable_key;
};

#endif 

typedef mpl::_ _relation;

} 
} 


#endif 

