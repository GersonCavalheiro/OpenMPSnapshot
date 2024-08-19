
#ifndef BOOST_INTERPROCESS_LOCK_OPTIONS_HPP
#define BOOST_INTERPROCESS_LOCK_OPTIONS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>


namespace boost {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace posix_time
{  class ptime;   }

#endif   

namespace interprocess {

struct defer_lock_type{};
struct try_to_lock_type {};
struct accept_ownership_type{};

static const defer_lock_type      defer_lock      = defer_lock_type();

static const try_to_lock_type     try_to_lock    = try_to_lock_type();

static const accept_ownership_type  accept_ownership = accept_ownership_type();

} 
} 

#include <boost/interprocess/detail/config_end.hpp>

#endif 
