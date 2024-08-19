

#ifndef BOOST_BIMAP_SUPPORT_ITERATOR_TYPE_BY_HPP
#define BOOST_BIMAP_SUPPORT_ITERATOR_TYPE_BY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/relation/detail/metadata_access_builder.hpp>
#include <boost/bimap/relation/detail/static_access_builder.hpp>



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace support {


BOOST_BIMAP_SYMMETRIC_METADATA_ACCESS_BUILDER
(
iterator_type_by,
left_iterator,
right_iterator
)


BOOST_BIMAP_SYMMETRIC_METADATA_ACCESS_BUILDER
(
const_iterator_type_by,
left_const_iterator,
right_const_iterator
)



BOOST_BIMAP_SYMMETRIC_METADATA_ACCESS_BUILDER
(
reverse_iterator_type_by,
left_reverse_iterator,
right_reverse_iterator
)


BOOST_BIMAP_SYMMETRIC_METADATA_ACCESS_BUILDER
(
const_reverse_iterator_type_by,
left_const_reverse_iterator,
right_const_reverse_iterator
)



BOOST_BIMAP_SYMMETRIC_METADATA_ACCESS_BUILDER
(
local_iterator_type_by,
left_local_iterator,
right_local_iterator
)


BOOST_BIMAP_SYMMETRIC_METADATA_ACCESS_BUILDER
(
const_local_iterator_type_by,
left_const_local_iterator,
right_const_local_iterator
)

} 
} 
} 

#endif 

#endif 

