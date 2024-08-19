

#ifndef BOOST_FLYWEIGHT_KEY_VALUE_FWD_HPP
#define BOOST_FLYWEIGHT_KEY_VALUE_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

namespace boost{

namespace flyweights{

struct no_key_from_value;

template<typename Key,typename Value,typename KeyFromValue=no_key_from_value>
struct key_value;

} 

} 

#endif
