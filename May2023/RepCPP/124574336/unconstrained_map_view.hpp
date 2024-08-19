

#ifndef BOOST_BIMAP_VIEWS_UNCONSTRAINED_MAP_VIEW_HPP
#define BOOST_BIMAP_VIEWS_UNCONSTRAINED_MAP_VIEW_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

namespace boost {
namespace bimaps {
namespace views {


template< class Tag, class BimapType>
class unconstrained_map_view
{
public:
template< class T >
unconstrained_map_view(const T &) {}

typedef void iterator;
typedef void const_iterator;
typedef void reference;
typedef void const_reference;
typedef void info_type;
};

} 
} 
} 

#endif 

