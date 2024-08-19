

#ifndef BOOST_FLYWEIGHT_INTERMODULE_HOLDER_HPP
#define BOOST_FLYWEIGHT_INTERMODULE_HOLDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/holder_tag.hpp>
#include <boost/flyweight/intermodule_holder_fwd.hpp>
#include <boost/interprocess/detail/intermodule_singleton.hpp>
#include <boost/mpl/aux_/lambda_support.hpp>



namespace boost{

namespace flyweights{

template<typename C>
struct intermodule_holder_class:
interprocess::ipcdetail::intermodule_singleton<C,true>,
holder_marker
{
typedef intermodule_holder_class type;
BOOST_MPL_AUX_LAMBDA_SUPPORT(1,intermodule_holder_class,(C))
};



struct intermodule_holder:holder_marker
{
template<typename C>
struct apply
{
typedef intermodule_holder_class<C> type;
};
};

} 

} 

#endif
