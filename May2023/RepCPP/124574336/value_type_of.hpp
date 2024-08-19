

#ifndef BOOST_BIMAP_TAGS_SUPPORT_VALUE_TYPE_OF_HPP
#define BOOST_BIMAP_TAGS_SUPPORT_VALUE_TYPE_OF_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/tags/tagged.hpp>



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES


namespace boost {
namespace bimaps {
namespace tags {
namespace support {



template< class Type >
struct value_type_of
{
typedef Type type;
};

template< class Type, class Tag >
struct value_type_of< tagged< Type, Tag > >
{
typedef Type type;
};


} 
} 
} 
} 

#endif 

#endif 


