

#ifndef BOOST_BIMAP_VIEWS_UNCONSTRAINED_SET_VIEW_HPP
#define BOOST_BIMAP_VIEWS_UNCONSTRAINED_SET_VIEW_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

namespace boost {
namespace bimaps {
namespace views {


template< class CoreIndex >
class unconstrained_set_view
{
public:
template< class T >
unconstrained_set_view(const T &) {}

typedef void iterator;
typedef void const_iterator;
};

} 
} 
} 

#endif 
