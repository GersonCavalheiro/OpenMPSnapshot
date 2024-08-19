

#ifndef BOOST_BIMAP_TAGS_TAGGED_HPP
#define BOOST_BIMAP_TAGS_TAGGED_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

namespace boost {
namespace bimaps {



namespace tags {


template< class Type, class Tag >
struct tagged
{
typedef Type value_type;
typedef Tag tag;
};

} 
} 
} 



#endif 




