

#ifndef BOOST_BIMAP_DETAIL_IS_SET_TYPE_OF_HPP
#define BOOST_BIMAP_DETAIL_IS_SET_TYPE_OF_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/type_traits/is_base_of.hpp>
#include <boost/bimap/detail/concept_tags.hpp>





#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace detail {

template< class Type >
struct is_set_type_of :
is_base_of< set_type_of_tag, Type > {};

template< class Type >
struct is_set_type_of_relation :
is_base_of< set_type_of_relation_tag, Type > {};

} 
} 
} 

#endif 

#endif 

