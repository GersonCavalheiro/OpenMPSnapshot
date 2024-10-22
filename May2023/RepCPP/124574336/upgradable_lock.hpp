
#ifndef BOOST_INTERPROCESS_UPGRADABLE_LOCK_HPP
#define BOOST_INTERPROCESS_UPGRADABLE_LOCK_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/sync/lock_options.hpp>
#include <boost/interprocess/detail/mpl.hpp>
#include <boost/interprocess/detail/type_traits.hpp>

#include <boost/interprocess/exceptions.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>


namespace boost {
namespace interprocess {

template <class UpgradableMutex>
class upgradable_lock
{
public:
typedef UpgradableMutex mutex_type;
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef upgradable_lock<UpgradableMutex> this_type;
explicit upgradable_lock(scoped_lock<mutex_type>&);
typedef bool this_type::*unspecified_bool_type;
BOOST_MOVABLE_BUT_NOT_COPYABLE(upgradable_lock)
#endif   
public:

upgradable_lock()
: mp_mutex(0), m_locked(false)
{}

explicit upgradable_lock(mutex_type& m)
: mp_mutex(&m), m_locked(false)
{  mp_mutex->lock_upgradable();   m_locked = true;  }

upgradable_lock(mutex_type& m, defer_lock_type)
: mp_mutex(&m), m_locked(false)
{}

upgradable_lock(mutex_type& m, accept_ownership_type)
: mp_mutex(&m), m_locked(true)
{}

upgradable_lock(mutex_type& m, try_to_lock_type)
: mp_mutex(&m), m_locked(false)
{  m_locked = mp_mutex->try_lock_upgradable();   }

upgradable_lock(mutex_type& m, const boost::posix_time::ptime& abs_time)
: mp_mutex(&m), m_locked(false)
{  m_locked = mp_mutex->timed_lock_upgradable(abs_time);  }

upgradable_lock(BOOST_RV_REF(upgradable_lock<mutex_type>) upgr)
: mp_mutex(0), m_locked(upgr.owns())
{  mp_mutex = upgr.release(); }

template<class T>
upgradable_lock(BOOST_RV_REF(scoped_lock<T>) scop
, typename ipcdetail::enable_if< ipcdetail::is_same<T, UpgradableMutex> >::type * = 0)
: mp_mutex(0), m_locked(false)
{
scoped_lock<mutex_type> &u_lock = scop;
if(u_lock.owns()){
u_lock.mutex()->unlock_and_lock_upgradable();
m_locked = true;
}
mp_mutex = u_lock.release();
}

template<class T>
upgradable_lock( BOOST_RV_REF(sharable_lock<T>) shar, try_to_lock_type
, typename ipcdetail::enable_if< ipcdetail::is_same<T, UpgradableMutex> >::type * = 0)
: mp_mutex(0), m_locked(false)
{
sharable_lock<mutex_type> &s_lock = shar;
if(s_lock.owns()){
if((m_locked = s_lock.mutex()->try_unlock_sharable_and_lock_upgradable()) == true){
mp_mutex = s_lock.release();
}
}
else{
s_lock.release();
}
}

~upgradable_lock()
{
try{
if(m_locked && mp_mutex)   mp_mutex->unlock_upgradable();
}
catch(...){}
}

upgradable_lock &operator=(BOOST_RV_REF(upgradable_lock) upgr)
{
if(this->owns())
this->unlock();
m_locked = upgr.owns();
mp_mutex = upgr.release();
return *this;
}

void lock()
{
if(!mp_mutex || m_locked)
throw lock_exception();
mp_mutex->lock_upgradable();
m_locked = true;
}

bool try_lock()
{
if(!mp_mutex || m_locked)
throw lock_exception();
m_locked = mp_mutex->try_lock_upgradable();
return m_locked;
}

bool timed_lock(const boost::posix_time::ptime& abs_time)
{
if(!mp_mutex || m_locked)
throw lock_exception();
m_locked = mp_mutex->timed_lock_upgradable(abs_time);
return m_locked;
}

void unlock()
{
if(!mp_mutex || !m_locked)
throw lock_exception();
mp_mutex->unlock_upgradable();
m_locked = false;
}

bool owns() const
{  return m_locked && mp_mutex;  }

operator unspecified_bool_type() const
{  return m_locked? &this_type::m_locked : 0;   }

mutex_type* mutex() const
{  return  mp_mutex;  }

mutex_type* release()
{
mutex_type *mut = mp_mutex;
mp_mutex = 0;
m_locked = false;
return mut;
}

void swap(upgradable_lock<mutex_type> &other)
{
(simple_swap)(mp_mutex, other.mp_mutex);
(simple_swap)(m_locked, other.m_locked);
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
mutex_type *mp_mutex;
bool        m_locked;
#endif   
};

} 
} 

#include <boost/interprocess/detail/config_end.hpp>

#endif 
