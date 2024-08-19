#ifndef  BOOST_SERIALIZATION_VOID_CAST_FWD_HPP
#define BOOST_SERIALIZATION_VOID_CAST_FWD_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstddef> 
#include <boost/serialization/force_include.hpp>

namespace boost {
namespace serialization {
namespace void_cast_detail{
class void_caster;
} 
template<class Derived, class Base>
BOOST_DLLEXPORT
inline const void_cast_detail::void_caster & void_cast_register(
const Derived * dnull = NULL,
const Base * bnull = NULL
) BOOST_USED;
} 
} 

#endif 
