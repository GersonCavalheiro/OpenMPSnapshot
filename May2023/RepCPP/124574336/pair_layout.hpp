

#ifndef BOOST_BIMAP_RELATION_PAIR_LAYOUT_HPP
#define BOOST_BIMAP_RELATION_PAIR_LAYOUT_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

namespace boost {
namespace bimaps {
namespace relation {



struct normal_layout {};


struct mirror_layout {};




#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

template< class Layout >
struct inverse_layout
{
typedef normal_layout type;
};

template<>
struct inverse_layout< normal_layout >
{
typedef mirror_layout type;
};

#endif 

} 
} 
} 

#endif 

