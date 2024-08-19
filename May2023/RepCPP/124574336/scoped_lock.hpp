
#ifndef BOOST_INTERPROCESS_SCOPED_LOCK_HPP
#define BOOST_INTERPROCESS_SCOPED_LOCK_HPP

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
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/mpl.hpp>
#include <boost/interprocess/detail/type_traits.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/detail/simple_swap.hpp>


namespace boost {
namespace interprocess {


template <class Mutex>
class scoped_lock
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef scoped_lock<Mutex> this_type;
BOOST_MOVABLE_BUT_NOT_COPYABLE(scoped_lock)
typedef bool this_type::*unspecified_bool_type;
#endif   
public:

typedef Mutex mutex_type;

scoped_lock()
: mp_mutex(0), m_locked(false)
{}

explicit scoped_lock(mutex_type& m)
: mp_mutex(&m), m_locked(false)
{  mp_mutex->lock();   m_locked = true;  }

scoped_lock(mutex_type& m, defer_lock_type)
: mp_mutex(&m), m_locked(false)
{}

scoped_lock(mutex_type& m, accept_ownership_type)
: mp_mutex(&m), m_locked(true)
{}

scoped_lock(mutex_type& m, try_to_lock_type)
: mp_mutex(&m), m_locked(mp_mutex->try_lock())
{}

scoped_lock(mutex_type& m, const boost::posix_time::ptime& abs_time)
: mp_mutex(&m), m_locked(mp_mutex->timed_lock(abs_time))
{}

scoped_lock(BOOST_RV_REF(scoped_lock) scop)
: mp_mutex(0), m_locked(scop.owns())
{  mp_mutex = scop.release(); }

template<class T>
explicit scoped_lock(BOOST_RV_REF(upgradable_lock<T>) upgr
, typename ipcdetail::enable_if< ipcdetail::is_same<T, Mutex> >::type * = 0)
: mp_mutex(0), m_locked(false)
{
upgradable_lock<mutex_type> &u_lock = upgr;
if(u_lock.owns()){
u_lock.mutex()->unlock_upgradable_and_lock();
m_locked = true;
}
mp_mutex = u_lock.release();
}

template<class T>
scoped_lock(BOOST_RV_REF(upgradable_lock<T>) upgr, try_to_lock_type
, typename ipcdetail::enable_if< ipcdetail::is_same<T, Mutex> >::type * = 0)
: mp_mutex(0), m_locked(false)
{
upgradable_lock<mutex_type> &u_lock = upgr;
if(u_lock.owns()){
if((m_locked = u_lock.mutex()->try_unlock_upgradable_and_lock()) == true){
mp_mutex = u_lock.release();
}
}
else{
u_lock.release();
}
}

template<class T>
scoped_lock(BOOST_RV_REF(upgradable_lock<T>) upgr, boost::posix_time::ptime &abs_time
, typename ipcdetail::enable_if< ipcdetail::is_same<T, Mutex> >::type * = 0)
: mp_mutex(0), m_locked(false)
{
upgradable_lock<mutex_type> &u_lock = upgr;
if(u_lock.owns()){
if((m_locked = u_lock.mutex()->timed_unlock_upgradable_and_lock(abs_time)) == true){
mp_mutex = u_lock.release();
}
}
else{
u_lock.release();
}
}

template<class T>
scoped_lock(BOOST_RV_REF(sharable_lock<T>) shar, try_to_lock_type
, typename ipcdetail::enable_if< ipcdetail::is_same<T, Mutex> >::type * = 0)
: mp_mutex(0), m_locked(false)
{
sharable_lock<mutex_type> &s_lock = shar;
if(s_lock.owns()){
if((m_locked = s_lock.mutex()->try_unlock_sharable_and_lock()) == true){
mp_mutex = s_lock.release();
}
}
else{
s_lock.release();
}
}

~scoped_lock()
{
try{  if(m_locked && mp_mutex)   mp_mutex->unlock();  }
catch(...){}
}

scoped_lock &operator=(BOOST_RV_REF(scoped_lock) scop)
{
if(this->owns())
this->unlock();
m_locked = scop.owns();
mp_mutex = scop.release();
return *this;
}

void lock()
{
if(!mp_mutex || m_locked)
throw lock_exception();
mp_mutex->lock();
m_locked = true;
}

bool try_lock()
{
if(!mp_mutex || m_locked)
throw lock_exception();
m_locked = mp_mutex->try_lock();
return m_locked;
}

bool timed_lock(const boost::posix_time::ptime& abs_time)
{
if(!mp_mutex || m_locked)
throw lock_exception();
m_locked = mp_mutex->timed_lock(abs_time);
return m_locked;
}

void unlock()
{
if(!mp_mutex || !m_locked)
throw lock_exception();
mp_mutex->unlock();
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

void swap( scoped_lock<mutex_type> &other)
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
