

#ifndef BOOST_BIMAP_DETAIL_MANAGE_BIMAP_KEY_HPP
#define BOOST_BIMAP_DETAIL_MANAGE_BIMAP_KEY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

#include <boost/bimap/detail/is_set_type_of.hpp>

#include <boost/bimap/set_of.hpp>

namespace boost {
namespace bimaps {
namespace detail {



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

template< class Type >
struct manage_bimap_key
{

typedef BOOST_DEDUCED_TYPENAME

mpl::eval_if< BOOST_DEDUCED_TYPENAME is_set_type_of< Type >::type,
mpl::identity< Type >,
mpl::identity< set_of< Type > >

>::type set_type;


typedef BOOST_DEDUCED_TYPENAME mpl::if_c< true, set_type, 
BOOST_DEDUCED_TYPENAME set_type::lazy_concept_checked::type
>::type type;
};



#endif 

} 
} 
} 


#endif 


