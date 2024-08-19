

#ifndef BOOST_BIMAP_TAGS_SUPPORT_APPLY_TO_VALUE_TYPE_HPP
#define BOOST_BIMAP_TAGS_SUPPORT_APPLY_TO_VALUE_TYPE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/tags/tagged.hpp>
#include <boost/mpl/apply.hpp>



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace tags {
namespace support {

template < class F, class TaggedType >
struct apply_to_value_type;

template < class F, class ValueType, class Tag >
struct apply_to_value_type<F, tagged<ValueType,Tag> >
{
typedef BOOST_DEDUCED_TYPENAME mpl::apply< F, ValueType >::type new_value_type;
typedef tagged< new_value_type, Tag > type;
};

} 
} 
} 
} 

#endif 

#endif 
