

#ifndef BOOST_BIMAP_VIEWS_UNORDERED_SET_VIEW_HPP
#define BOOST_BIMAP_VIEWS_UNORDERED_SET_VIEW_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/unordered_set_adaptor.hpp>
#include <boost/bimap/detail/set_view_base.hpp>

namespace boost {
namespace bimaps {
namespace views {



template< class CoreIndex >
class unordered_set_view
:
public BOOST_BIMAP_SET_VIEW_CONTAINER_ADAPTOR(
unordered_set_adaptor,
CoreIndex,
local_iterator,
const_local_iterator
),

public ::boost::bimaps::detail::
set_view_base< unordered_set_view< CoreIndex >, CoreIndex >
{
BOOST_BIMAP_SET_VIEW_BASE_FRIEND(unordered_set_view,CoreIndex)

typedef BOOST_BIMAP_SET_VIEW_CONTAINER_ADAPTOR(
unordered_set_adaptor,
CoreIndex,
local_iterator,
const_local_iterator

) base_;

public:

unordered_set_view(BOOST_DEDUCED_TYPENAME base_::base_type & c)
: base_(c) {}

unordered_set_view & operator=(const unordered_set_view & v) 
{
this->base() = v.base();
return *this;
}
};


} 
} 
} 

#endif 

