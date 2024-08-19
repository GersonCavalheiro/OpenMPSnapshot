

#ifndef BOOST_BIMAP_SUPPORT_MAP_BY_HPP
#define BOOST_BIMAP_SUPPORT_MAP_BY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/support/map_type_by.hpp>
#include <boost/bimap/relation/detail/access_builder.hpp>


#ifdef BOOST_BIMAP_ONLY_DOXYGEN_WILL_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace support {



template< class Tag, class Bimap >
BOOST_DEDUCED_TYPENAME result_of::map_by<Tag,Bimap>::type
map_by( Bimap & b );

} 
} 
} 

#endif 



#ifndef BOOST_BIMAP_DOXIGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace support {




BOOST_BIMAP_SYMMETRIC_ACCESS_RESULT_OF_BUILDER
(
map_by,
map_type_by
)


BOOST_BIMAP_SYMMETRIC_ACCESS_IMPLEMENTATION_BUILDER
(
map_by,
Bimap,
b,
return b.left,
return b.right
)


BOOST_BIMAP_SYMMETRIC_ACCESS_INTERFACE_BUILDER
(
map_by
)

} 
} 
} 

#endif 

#endif 

