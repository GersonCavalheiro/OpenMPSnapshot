

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_FUNCTOR_BAG_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_FUNCTOR_BAG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#   pragma warning(push)
#   pragma warning(disable:4181)
#endif

#include <boost/mpl/placeholders.hpp>

#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/inherit.hpp>

namespace boost {
namespace bimaps {
namespace container_adaptor {
namespace detail {



template < class Data, class FunctorList >
struct data_with_functor_bag :

public mpl::inherit_linearly<

FunctorList,
mpl::if_< is_base_of< mpl::_2, mpl::_1 >,
mpl::_1,
mpl::inherit< mpl::_1, mpl::_2 >
>

>::type
{
Data data;

data_with_functor_bag() {}

data_with_functor_bag(BOOST_DEDUCED_TYPENAME add_reference<Data>::type d)
: data(d) {}

template< class Functor >
Functor& functor()
{
return *(static_cast<Functor*>(this));
}

template< class Functor >
const Functor& functor() const
{
return *(static_cast<Functor const *>(this));
}
};

} 
} 
} 
} 

#if defined(BOOST_MSVC)
#   pragma warning(pop)
#endif

#endif 


