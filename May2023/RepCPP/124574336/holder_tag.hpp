

#ifndef BOOST_FLYWEIGHT_HOLDER_TAG_HPP
#define BOOST_FLYWEIGHT_HOLDER_TAG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/parameter/parameters.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

namespace boost{

namespace flyweights{



struct holder_marker{};

template<typename T>
struct is_holder:is_base_and_derived<holder_marker,T>
{};

template<typename T=parameter::void_>
struct holder:parameter::template_keyword<holder<>,T>
{};

} 

} 

#endif
