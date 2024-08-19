

#ifndef BOOST_BIMAP_RELATION_SYMMETRICAL_BASE_HPP
#define BOOST_BIMAP_RELATION_SYMMETRICAL_BASE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/mpl/if.hpp>
#include <boost/type_traits/remove_const.hpp>

#include <boost/bimap/tags/tagged.hpp>
#include <boost/bimap/tags/support/default_tagged.hpp>

#include <boost/bimap/relation/member_at.hpp>


namespace boost {
namespace bimaps {
namespace relation {



template< class TA, class TB, bool force_mutable = false >
class symmetrical_base
{

public:

typedef BOOST_DEDUCED_TYPENAME tags::support::default_tagged
<
TA,
member_at::left

>::type tagged_left_type;

typedef BOOST_DEDUCED_TYPENAME tags::support::default_tagged
<
TB,
member_at::right

>::type tagged_right_type;

public:


typedef BOOST_DEDUCED_TYPENAME ::boost::mpl::if_c< force_mutable,

BOOST_DEDUCED_TYPENAME ::boost::remove_const<
BOOST_DEDUCED_TYPENAME tagged_left_type::value_type >::type,
BOOST_DEDUCED_TYPENAME tagged_left_type::value_type

>::type left_value_type;

typedef BOOST_DEDUCED_TYPENAME ::boost::mpl::if_c< force_mutable,

BOOST_DEDUCED_TYPENAME ::boost::remove_const<
BOOST_DEDUCED_TYPENAME tagged_right_type::value_type >::type,
BOOST_DEDUCED_TYPENAME tagged_right_type::value_type

>::type right_value_type;

typedef BOOST_DEDUCED_TYPENAME tagged_left_type ::tag  left_tag;
typedef BOOST_DEDUCED_TYPENAME tagged_right_type::tag right_tag;
};



} 
} 
} 


#endif 

