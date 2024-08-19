

#ifndef BOOST_BIMAP_TAGS_SUPPORT_TAG_OF_HPP
#define BOOST_BIMAP_TAGS_SUPPORT_TAG_OF_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/tags/tagged.hpp>
#include <boost/bimap/detail/debug/static_error.hpp>



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace tags {
namespace support {



template< class Type >
struct tag_of
{
BOOST_BIMAP_STATIC_ERROR( USING_TAG_OF_WITH_AN_UNTAGGED_TYPE, (Type) );
};

template< class Type, class Tag >
struct tag_of< tagged< Type, Tag > >
{
typedef Tag type;
};


} 
} 
} 
} 

#endif 

#endif 

