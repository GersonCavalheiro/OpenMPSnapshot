


#ifndef BOOST_CORE_NULL_DELETER_HPP
#define BOOST_CORE_NULL_DELETER_HPP

#include <boost/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {

struct null_deleter
{
typedef void result_type;

template< typename T >
void operator() (T*) const BOOST_NOEXCEPT {}
};

} 

#endif 
