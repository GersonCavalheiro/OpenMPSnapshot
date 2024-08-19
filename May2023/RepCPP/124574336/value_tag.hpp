

#ifndef BOOST_FLYWEIGHT_DETAIL_VALUE_TAG_HPP
#define BOOST_FLYWEIGHT_DETAIL_VALUE_TAG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/parameter/parameters.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

namespace boost{

namespace flyweights{

namespace detail{



struct value_marker{};

template<typename T>
struct is_value:is_base_and_derived<value_marker,T>
{};

template<typename T=parameter::void_>
struct value:parameter::template_keyword<value<>,T>
{};

} 

} 

} 

#endif
