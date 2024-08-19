

#ifndef BOOST_BIMAP_RELATION_SUPPORT_PAIR_BY_HPP
#define BOOST_BIMAP_RELATION_SUPPORT_PAIR_BY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/relation/support/pair_type_by.hpp>
#include <boost/bimap/relation/detail/access_builder.hpp>

#ifdef BOOST_BIMAP_ONLY_DOXYGEN_WILL_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace relation {
namespace support {



template< class Tag, class Relation >
BOOST_DEDUCED_TYPENAME result_of::pair_by<Tag,Relation>::type
pair_by( Relation & rel );

} 
} 
} 
} 

#endif 


#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace relation {
namespace support {





BOOST_BIMAP_SYMMETRIC_ACCESS_RESULT_OF_BUILDER
(
pair_by,
pair_type_by
)




BOOST_BIMAP_SYMMETRIC_ACCESS_IMPLEMENTATION_BUILDER
(
pair_by,
Relation,
rel,
return rel.get_left_pair(),
return rel.get_right_pair()
)


BOOST_BIMAP_SYMMETRIC_ACCESS_INTERFACE_BUILDER
(
pair_by
)

} 
} 
} 
} 


#endif 

#endif 
