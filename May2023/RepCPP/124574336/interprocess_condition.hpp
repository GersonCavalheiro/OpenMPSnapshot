
#ifndef BOOST_INTERPROCESS_CONDITION_HPP
#define BOOST_INTERPROCESS_CONDITION_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/detail/locks.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/limits.hpp>
#include <boost/assert.hpp>

#if !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && defined(BOOST_INTERPROCESS_POSIX_PROCESS_SHARED)
#include <boost/interprocess/sync/posix/condition.hpp>
#define BOOST_INTERPROCESS_USE_POSIX
#elif !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && defined (BOOST_INTERPROCESS_WINDOWS)
#include <boost/interprocess/sync/windows/condition.hpp>
#define BOOST_INTERPROCESS_USE_WINDOWS
#elif !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
#include <boost/interprocess/sync/spin/condition.hpp>
#define BOOST_INTERPROCESS_USE_GENERIC_EMULATION
#endif

#endif   


namespace boost {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace posix_time
{  class ptime;   }

#endif   

namespace interprocess {

class named_condition;

class interprocess_condition
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
interprocess_condition(const interprocess_condition &);
interprocess_condition &operator=(const interprocess_condition &);
friend class named_condition;
#endif   

public:
interprocess_condition()
{}

~interprocess_condition()
{}

void notify_one()
{  m_condition.notify_one();  }

void notify_all()
{  m_condition.notify_all();  }

template <typename L>
void wait(L& lock)
{
ipcdetail::internal_mutex_lock<L> internal_lock(lock);
m_condition.wait(internal_lock);
}

template <typename L, typename Pr>
void wait(L& lock, Pr pred)
{
ipcdetail::internal_mutex_lock<L> internal_lock(lock);
m_condition.wait(internal_lock, pred);
}

template <typename L>
bool timed_wait(L& lock, const boost::posix_time::ptime &abs_time)
{
ipcdetail::internal_mutex_lock<L> internal_lock(lock);
return m_condition.timed_wait(internal_lock, abs_time);
}

template <typename L, typename Pr>
bool timed_wait(L& lock, const boost::posix_time::ptime &abs_time, Pr pred)
{
ipcdetail::internal_mutex_lock<L> internal_lock(lock);
return m_condition.timed_wait(internal_lock, abs_time, pred);
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

private:
#if defined (BOOST_INTERPROCESS_USE_GENERIC_EMULATION)
#undef BOOST_INTERPROCESS_USE_GENERIC_EMULATION
ipcdetail::spin_condition m_condition;
#elif defined(BOOST_INTERPROCESS_USE_POSIX)
#undef BOOST_INTERPROCESS_USE_POSIX
ipcdetail::posix_condition m_condition;
#elif defined(BOOST_INTERPROCESS_USE_WINDOWS)
#undef BOOST_INTERPROCESS_USE_WINDOWS
ipcdetail::windows_condition m_condition;
#else
#error "Unknown platform for interprocess_mutex"
#endif
#endif   
};

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif 
