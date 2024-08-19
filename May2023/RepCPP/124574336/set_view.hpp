

#ifndef BOOST_BIMAP_VIEWS_SET_VIEW_HPP
#define BOOST_BIMAP_VIEWS_SET_VIEW_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/set_adaptor.hpp>
#include <boost/bimap/detail/set_view_base.hpp>

namespace boost {
namespace bimaps {
namespace views {



template< class CoreIndex >
class set_view
:
public BOOST_BIMAP_SET_VIEW_CONTAINER_ADAPTOR(
set_adaptor,
CoreIndex,
reverse_iterator, const_reverse_iterator
),

public ::boost::bimaps::detail::
set_view_base< set_view< CoreIndex >, CoreIndex >
{
typedef BOOST_BIMAP_SET_VIEW_CONTAINER_ADAPTOR(
set_adaptor,
CoreIndex,
reverse_iterator, const_reverse_iterator

) base_;

BOOST_BIMAP_SET_VIEW_BASE_FRIEND(set_view,CoreIndex)

public:

set_view(BOOST_DEDUCED_TYPENAME base_::base_type & c) : base_(c) {}



set_view & operator=(const set_view & v) 
{
this->base() = v.base();
return *this;
}
};


} 
} 
} 

#endif 


