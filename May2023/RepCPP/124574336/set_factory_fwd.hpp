

#ifndef BOOST_FLYWEIGHT_SET_FACTORY_FWD_HPP
#define BOOST_FLYWEIGHT_SET_FACTORY_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/detail/not_placeholder_expr.hpp>
#include <boost/mpl/aux_/na.hpp>

namespace boost{

namespace flyweights{

template<
typename Entry,typename Key,
typename Compare=mpl::na,typename Allocator=mpl::na
>
class set_factory_class;

template<
typename Compare=mpl::na,typename Allocator=mpl::na
BOOST_FLYWEIGHT_NOT_A_PLACEHOLDER_EXPRESSION
>
struct set_factory;

} 

} 

#endif
