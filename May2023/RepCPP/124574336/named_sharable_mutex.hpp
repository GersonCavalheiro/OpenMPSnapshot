
#ifndef BOOST_INTERPROCESS_NAMED_SHARABLE_MUTEX_HPP
#define BOOST_INTERPROCESS_NAMED_SHARABLE_MUTEX_HPP

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
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/shm/named_creation_functor.hpp>
#include <boost/interprocess/permissions.hpp>


namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
namespace ipcdetail{ class interprocess_tester; }
#endif   

class named_condition;

class named_sharable_mutex
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
named_sharable_mutex();
named_sharable_mutex(const named_sharable_mutex &);
named_sharable_mutex &operator=(const named_sharable_mutex &);
#endif   
public:

named_sharable_mutex(create_only_t create_only, const char *name, const permissions &perm = permissions());

named_sharable_mutex(open_or_create_t open_or_create, const char *name, const permissions &perm = permissions());

named_sharable_mutex(open_only_t open_only, const char *name);

~named_sharable_mutex();


void lock();

bool try_lock();

bool timed_lock(const boost::posix_time::ptime &abs_time);

void unlock();


void lock_sharable();

bool try_lock_sharable();

bool timed_lock_sharable(const boost::posix_time::ptime &abs_time);

void unlock_sharable();

static bool remove(const char *name);

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
friend class ipcdetail::interprocess_tester;
void dont_close_on_destruction();

interprocess_sharable_mutex *mutex() const
{  return static_cast<interprocess_sharable_mutex*>(m_shmem.get_user_address()); }

typedef ipcdetail::managed_open_or_create_impl<shared_memory_object, 0, true, false> open_create_impl_t;
open_create_impl_t m_shmem;
typedef ipcdetail::named_creation_functor<interprocess_sharable_mutex> construct_func_t;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

inline named_sharable_mutex::~named_sharable_mutex()
{}

inline named_sharable_mutex::named_sharable_mutex
(create_only_t, const char *name, const permissions &perm)
:  m_shmem  (create_only
,name
,sizeof(interprocess_sharable_mutex) +
open_create_impl_t::ManagedOpenOrCreateUserOffset
,read_write
,0
,construct_func_t(ipcdetail::DoCreate)
,perm)
{}

inline named_sharable_mutex::named_sharable_mutex
(open_or_create_t, const char *name, const permissions &perm)
:  m_shmem  (open_or_create
,name
,sizeof(interprocess_sharable_mutex) +
open_create_impl_t::ManagedOpenOrCreateUserOffset
,read_write
,0
,construct_func_t(ipcdetail::DoOpenOrCreate)
,perm)
{}

inline named_sharable_mutex::named_sharable_mutex
(open_only_t, const char *name)
:  m_shmem  (open_only
,name
,read_write
,0
,construct_func_t(ipcdetail::DoOpen))
{}

inline void named_sharable_mutex::dont_close_on_destruction()
{  ipcdetail::interprocess_tester::dont_close_on_destruction(m_shmem);  }

inline void named_sharable_mutex::lock()
{  this->mutex()->lock();  }

inline void named_sharable_mutex::unlock()
{  this->mutex()->unlock();  }

inline bool named_sharable_mutex::try_lock()
{  return this->mutex()->try_lock();  }

inline bool named_sharable_mutex::timed_lock
(const boost::posix_time::ptime &abs_time)
{  return this->mutex()->timed_lock(abs_time);  }

inline void named_sharable_mutex::lock_sharable()
{  this->mutex()->lock_sharable();  }

inline void named_sharable_mutex::unlock_sharable()
{  this->mutex()->unlock_sharable();  }

inline bool named_sharable_mutex::try_lock_sharable()
{  return this->mutex()->try_lock_sharable();  }

inline bool named_sharable_mutex::timed_lock_sharable
(const boost::posix_time::ptime &abs_time)
{  return this->mutex()->timed_lock_sharable(abs_time);  }

inline bool named_sharable_mutex::remove(const char *name)
{  return shared_memory_object::remove(name); }

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
