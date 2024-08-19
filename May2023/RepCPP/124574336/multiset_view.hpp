

#ifndef BOOST_BIMAP_VIEWS_MULTISET_VIEW_HPP
#define BOOST_BIMAP_VIEWS_MULTISET_VIEW_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/multiset_adaptor.hpp>
#include <boost/bimap/container_adaptor/detail/comparison_adaptor.hpp>
#include <boost/bimap/detail/non_unique_views_helper.hpp>
#include <boost/bimap/detail/set_view_base.hpp>

namespace boost {
namespace bimaps {
namespace views {



template< class CoreIndex >
class multiset_view
:
public BOOST_BIMAP_SET_VIEW_CONTAINER_ADAPTOR(
multiset_adaptor,
CoreIndex,
reverse_iterator,
const_reverse_iterator
),

public ::boost::bimaps::detail::
set_view_base< multiset_view< CoreIndex >, CoreIndex >
{
BOOST_BIMAP_SET_VIEW_BASE_FRIEND(multiset_view, CoreIndex)

typedef BOOST_BIMAP_SET_VIEW_CONTAINER_ADAPTOR(
multiset_adaptor,
CoreIndex,
reverse_iterator,
const_reverse_iterator

) base_;

public:

multiset_view(BOOST_DEDUCED_TYPENAME base_::base_type & c) : base_(c) {}



multiset_view & operator=(const multiset_view & v) 
{
this->base() = v.base(); return *this;
}

BOOST_BIMAP_NON_UNIQUE_VIEW_INSERT_FUNCTIONS
};


} 
} 
} 

#endif 

