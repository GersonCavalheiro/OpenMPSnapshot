
#ifndef BOOST_INTERPROCESS_NULL_MUTEX_HPP
#define BOOST_INTERPROCESS_NULL_MUTEX_HPP

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

class null_mutex
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
null_mutex(const null_mutex&);
null_mutex &operator= (const null_mutex&);
#endif   
public:

null_mutex(){}

~null_mutex(){}

void lock(){}

bool try_lock()
{  return true;   }

bool timed_lock(const boost::posix_time::ptime &)
{  return true;   }

void unlock(){}

void lock_sharable(){}

bool try_lock_sharable()
{  return true;   }

bool timed_lock_sharable(const boost::posix_time::ptime &)
{  return true;   }

void unlock_sharable(){}

void lock_upgradable(){}

bool try_lock_upgradable()
{  return true;   }

bool timed_lock_upgradable(const boost::posix_time::ptime &)
{  return true;   }

void unlock_upgradable(){}

void unlock_and_lock_upgradable(){}

void unlock_and_lock_sharable(){}

void unlock_upgradable_and_lock_sharable(){}


void unlock_upgradable_and_lock(){}

bool try_unlock_upgradable_and_lock()
{  return true;   }

bool timed_unlock_upgradable_and_lock(const boost::posix_time::ptime &)
{  return true;   }

bool try_unlock_sharable_and_lock()
{  return true;   }

bool try_unlock_sharable_and_lock_upgradable()
{  return true;   }
};

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
