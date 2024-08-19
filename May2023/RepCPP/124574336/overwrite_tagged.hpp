

#ifndef BOOST_BIMAP_TAGS_SUPPORT_OVERWRITE_TAGGED_HPP
#define BOOST_BIMAP_TAGS_SUPPORT_OVERWRITE_TAGGED_HPP

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



template< class Type, class NewTag >
struct overwrite_tagged
{
typedef tagged<Type,NewTag> type;
};

template< class Type, class OldTag, class NewTag >
struct overwrite_tagged< tagged< Type, OldTag >, NewTag >
{
typedef tagged<Type,NewTag> type;
};


} 
} 
} 
} 

#endif 

#endif 


