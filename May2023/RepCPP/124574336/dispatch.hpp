

#ifndef BOOST_IOSTREAMS_DETAIL_DISPATCH_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_DISPATCH_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/config.hpp>                   
#include <boost/iostreams/detail/select.hpp>
#include <boost/iostreams/traits.hpp>         
#include <boost/mpl/void.hpp>          
#include <boost/type_traits/is_convertible.hpp>         

namespace boost { namespace iostreams {namespace detail {

template< typename T, typename Tag1, typename Tag2,
typename Tag3 = mpl::void_, typename Tag4 = mpl::void_,
typename Tag5 = mpl::void_, typename Tag6 = mpl::void_,
typename Category = 
BOOST_DEDUCED_TYPENAME category_of<T>::type >
struct dispatch 
: iostreams::select<  
is_convertible<Category, Tag1>, Tag1,
is_convertible<Category, Tag2>, Tag2,
is_convertible<Category, Tag3>, Tag3,
is_convertible<Category, Tag4>, Tag4,
is_convertible<Category, Tag5>, Tag5,
is_convertible<Category, Tag6>, Tag6
>
{ };

} } } 

#endif 
