

#ifndef BOOST_BIMAP_VIEWS_LIST_SET_VIEW_HPP
#define BOOST_BIMAP_VIEWS_LIST_SET_VIEW_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/list_adaptor.hpp>
#include <boost/bimap/detail/set_view_base.hpp>
#include <boost/bimap/detail/map_view_base.hpp>

namespace boost {
namespace bimaps {
namespace views {



template< class CoreIndex >
class list_set_view
:
public BOOST_BIMAP_SEQUENCED_SET_VIEW_CONTAINER_ADAPTOR(
list_adaptor,
CoreIndex,
reverse_iterator, const_reverse_iterator
),

public ::boost::bimaps::detail::
set_view_base< list_set_view< CoreIndex >, CoreIndex >
{
BOOST_BIMAP_SET_VIEW_BASE_FRIEND(list_set_view,CoreIndex)

typedef BOOST_BIMAP_SEQUENCED_SET_VIEW_CONTAINER_ADAPTOR(
list_adaptor,
CoreIndex,
reverse_iterator, const_reverse_iterator

) base_;

public:

list_set_view(BOOST_DEDUCED_TYPENAME base_::base_type & c) :
base_(c) {}

list_set_view & operator=(const list_set_view & v) 
{
this->base() = v.base();
return *this;
}

BOOST_BIMAP_VIEW_ASSIGN_IMPLEMENTATION(base_)

BOOST_BIMAP_VIEW_FRONT_BACK_IMPLEMENTATION(base_)


void relocate(BOOST_DEDUCED_TYPENAME base_::iterator position, 
BOOST_DEDUCED_TYPENAME base_::iterator i)
{
this->base().relocate(
this->template functor<
BOOST_DEDUCED_TYPENAME base_::iterator_to_base>()(position),
this->template functor<
BOOST_DEDUCED_TYPENAME base_::iterator_to_base>()(i)
);
}

void relocate(BOOST_DEDUCED_TYPENAME base_::iterator position,
BOOST_DEDUCED_TYPENAME base_::iterator first,
BOOST_DEDUCED_TYPENAME base_::iterator last)
{
this->base().relocate(
this->template functor<
BOOST_DEDUCED_TYPENAME base_::iterator_to_base>()(position),
this->template functor<
BOOST_DEDUCED_TYPENAME base_::iterator_to_base>()(first),
this->template functor<
BOOST_DEDUCED_TYPENAME base_::iterator_to_base>()(last)
);
}
};


} 
} 
} 


#endif 

