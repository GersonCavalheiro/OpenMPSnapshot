

#ifndef BOOST_BIMAP_TAGS_SUPPORT_IS_TAGGED_HPP
#define BOOST_BIMAP_TAGS_SUPPORT_IS_TAGGED_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/bimap/tags/tagged.hpp>



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace tags {
namespace support {



template< class Type >
struct is_tagged :
::boost::mpl::false_ {};

template< class Type, class Tag >
struct is_tagged< tagged< Type, Tag > > :
::boost::mpl::true_ {};

} 
} 
} 
} 

#endif 

#endif 

