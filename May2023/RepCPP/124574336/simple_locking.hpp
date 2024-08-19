

#ifndef BOOST_FLYWEIGHT_SIMPLE_LOCKING_HPP
#define BOOST_FLYWEIGHT_SIMPLE_LOCKING_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/detail/recursive_lw_mutex.hpp>
#include <boost/flyweight/simple_locking_fwd.hpp>
#include <boost/flyweight/locking_tag.hpp>



namespace boost{

namespace flyweights{

struct simple_locking:locking_marker
{
typedef detail::recursive_lightweight_mutex mutex_type;
typedef mutex_type::scoped_lock             lock_type;
};

} 

} 

#endif
