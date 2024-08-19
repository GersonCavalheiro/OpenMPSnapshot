

#ifndef BOOST_FLYWEIGHT_TRACKING_TAG_HPP
#define BOOST_FLYWEIGHT_TRACKING_TAG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/parameter/parameters.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

namespace boost{

namespace flyweights{



struct tracking_marker{};

template<typename T>
struct is_tracking:is_base_and_derived<tracking_marker,T>
{};

template<typename T=parameter::void_>
struct tracking:parameter::template_keyword<tracking<>,T>
{};

} 

} 

#endif
