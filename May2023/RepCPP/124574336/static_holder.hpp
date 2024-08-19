

#ifndef BOOST_FLYWEIGHT_STATIC_HOLDER_HPP
#define BOOST_FLYWEIGHT_STATIC_HOLDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/static_holder_fwd.hpp>
#include <boost/flyweight/holder_tag.hpp>
#include <boost/mpl/aux_/lambda_support.hpp>



namespace boost{

namespace flyweights{

template<typename C>
struct static_holder_class:holder_marker
{
static C& get()
{
static C c;
return c;
}

typedef static_holder_class type;
BOOST_MPL_AUX_LAMBDA_SUPPORT(1,static_holder_class,(C))
};



struct static_holder:holder_marker
{
template<typename C>
struct apply
{
typedef static_holder_class<C> type;
};
};

} 

} 

#endif
