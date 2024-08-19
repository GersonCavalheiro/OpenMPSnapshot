
#ifndef BOOST_INTERPROCESS_NAMED_UPGRADABLE_MUTEX_HPP
#define BOOST_INTERPROCESS_NAMED_UPGRADABLE_MUTEX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/detail/managed_open_or_create_impl.hpp>
#include <boost/interprocess/sync/interprocess_upgradable_mutex.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/shm/named_creation_functor.hpp>
#include <boost/interprocess/permissions.hpp>


namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
namespace ipcdetail{ class interprocess_tester; }
#endif   

class named_condition;

class named_upgradable_mutex
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
named_upgradable_mutex();
named_upgradable_mutex(const named_upgradable_mutex &);
named_upgradable_mutex &operator=(const named_upgradable_mutex &);
friend class named_condition;
#endif   
public:

named_upgradable_mutex(create_only_t create_only, const char *name, const permissions &perm = permissions());

named_upgradable_mutex(open_or_create_t open_or_create, const char *name, const permissions &perm = permissions());

named_upgradable_mutex(open_only_t open_only, const char *name);

~named_upgradable_mutex();


void lock();

bool try_lock();

bool timed_lock(const boost::posix_time::ptime &abs_time);

void unlock();


void lock_sharable();

bool try_lock_sharable();

bool timed_lock_sharable(const boost::posix_time::ptime &abs_time);

void unlock_sharable();


void lock_upgradable();

bool try_lock_upgradable();

bool timed_lock_upgradable(const boost::posix_time::ptime &abs_time);

void unlock_upgradable();


void unlock_and_lock_upgradable();

void unlock_and_lock_sharable();

void unlock_upgradable_and_lock_sharable();


void unlock_upgradable_and_lock();

bool try_unlock_upgradable_and_lock();

bool timed_unlock_upgradable_and_lock(const boost::posix_time::ptime &abs_time);

bool try_unlock_sharable_and_lock();

bool try_unlock_sharable_and_lock_upgradable();

static bool remove(const char *name);

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
friend class ipcdetail::interprocess_tester;
void dont_close_on_destruction();

interprocess_upgradable_mutex *mutex() const
{  return static_cast<interprocess_upgradable_mutex*>(m_shmem.get_user_address()); }

typedef ipcdetail::managed_open_or_create_impl<shared_memory_object, 0, true, false> open_create_impl_t;
open_create_impl_t m_shmem;
typedef ipcdetail::named_creation_functor<interprocess_upgradable_mutex> construct_func_t;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

inline named_upgradable_mutex::~named_upgradable_mutex()
{}

inline named_upgradable_mutex::named_upgradable_mutex
(create_only_t, const char *name, const permissions &perm)
:  m_shmem  (create_only
,name
,sizeof(interprocess_upgradable_mutex) +
open_create_impl_t::ManagedOpenOrCreateUserOffset
,read_write
,0
,construct_func_t(ipcdetail::DoCreate)
,perm)
{}

inline named_upgradable_mutex::named_upgradable_mutex
(open_or_create_t, const char *name, const permissions &perm)
:  m_shmem  (open_or_create
,name
,sizeof(interprocess_upgradable_mutex) +
open_create_impl_t::ManagedOpenOrCreateUserOffset
,read_write
,0
,construct_func_t(ipcdetail::DoOpenOrCreate)
,perm)
{}

inline named_upgradable_mutex::named_upgradable_mutex
(open_only_t, const char *name)
:  m_shmem  (open_only
,name
,read_write
,0
,construct_func_t(ipcdetail::DoOpen))
{}

inline void named_upgradable_mutex::dont_close_on_destruction()
{  ipcdetail::interprocess_tester::dont_close_on_destruction(m_shmem);  }

inline void named_upgradable_mutex::lock()
{  this->mutex()->lock();  }

inline void named_upgradable_mutex::unlock()
{  this->mutex()->unlock();  }

inline bool named_upgradable_mutex::try_lock()
{  return this->mutex()->try_lock();  }

inline bool named_upgradable_mutex::timed_lock
(const boost::posix_time::ptime &abs_time)
{  return this->mutex()->timed_lock(abs_time);  }

inline void named_upgradable_mutex::lock_upgradable()
{  this->mutex()->lock_upgradable();  }

inline void named_upgradable_mutex::unlock_upgradable()
{  this->mutex()->unlock_upgradable();  }

inline bool named_upgradable_mutex::try_lock_upgradable()
{  return this->mutex()->try_lock_upgradable();  }

inline bool named_upgradable_mutex::timed_lock_upgradable
(const boost::posix_time::ptime &abs_time)
{  return this->mutex()->timed_lock_upgradable(abs_time);   }

inline void named_upgradable_mutex::lock_sharable()
{  this->mutex()->lock_sharable();  }

inline void named_upgradable_mutex::unlock_sharable()
{  this->mutex()->unlock_sharable();  }

inline bool named_upgradable_mutex::try_lock_sharable()
{  return this->mutex()->try_lock_sharable();  }

inline bool named_upgradable_mutex::timed_lock_sharable
(const boost::posix_time::ptime &abs_time)
{  return this->mutex()->timed_lock_sharable(abs_time);  }

inline void named_upgradable_mutex::unlock_and_lock_upgradable()
{  this->mutex()->unlock_and_lock_upgradable();  }

inline void named_upgradable_mutex::unlock_and_lock_sharable()
{  this->mutex()->unlock_and_lock_sharable();  }

inline void named_upgradable_mutex::unlock_upgradable_and_lock_sharable()
{  this->mutex()->unlock_upgradable_and_lock_sharable();  }

inline void named_upgradable_mutex::unlock_upgradable_and_lock()
{  this->mutex()->unlock_upgradable_and_lock();  }

inline bool named_upgradable_mutex::try_unlock_upgradable_and_lock()
{  return this->mutex()->try_unlock_upgradable_and_lock();  }

inline bool named_upgradable_mutex::timed_unlock_upgradable_and_lock
(const boost::posix_time::ptime &abs_time)
{  return this->mutex()->timed_unlock_upgradable_and_lock(abs_time);  }

inline bool named_upgradable_mutex::try_unlock_sharable_and_lock()
{  return this->mutex()->try_unlock_sharable_and_lock();  }

inline bool named_upgradable_mutex::try_unlock_sharable_and_lock_upgradable()
{  return this->mutex()->try_unlock_sharable_and_lock_upgradable();  }

inline bool named_upgradable_mutex::remove(const char *name)
{  return shared_memory_object::remove(name); }

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
