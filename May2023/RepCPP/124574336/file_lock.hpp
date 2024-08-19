
#ifndef BOOST_INTERPROCESS_FILE_LOCK_HPP
#define BOOST_INTERPROCESS_FILE_LOCK_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/detail/os_thread_functions.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/detail/common_algorithms.hpp>
#include <boost/interprocess/sync/detail/locks.hpp>
#include <boost/move/utility_core.hpp>


namespace boost {
namespace interprocess {


class file_lock
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
BOOST_MOVABLE_BUT_NOT_COPYABLE(file_lock)
#endif   

public:
file_lock()
:  m_file_hnd(file_handle_t(ipcdetail::invalid_file()))
{}

file_lock(const char *name);

file_lock(BOOST_RV_REF(file_lock) moved)
:  m_file_hnd(file_handle_t(ipcdetail::invalid_file()))
{  this->swap(moved);   }

file_lock &operator=(BOOST_RV_REF(file_lock) moved)
{
file_lock tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

~file_lock();

void swap(file_lock &other)
{
file_handle_t tmp = m_file_hnd;
m_file_hnd = other.m_file_hnd;
other.m_file_hnd = tmp;
}


void lock();

bool try_lock();

bool timed_lock(const boost::posix_time::ptime &abs_time);

void unlock();


void lock_sharable();

bool try_lock_sharable();

bool timed_lock_sharable(const boost::posix_time::ptime &abs_time);

void unlock_sharable();
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
file_handle_t m_file_hnd;

#endif   
};

inline file_lock::file_lock(const char *name)
{
m_file_hnd = ipcdetail::open_existing_file(name, read_write);

if(m_file_hnd == ipcdetail::invalid_file()){
error_info err(system_error_code());
throw interprocess_exception(err);
}
}

inline file_lock::~file_lock()
{
if(m_file_hnd != ipcdetail::invalid_file()){
ipcdetail::close_file(m_file_hnd);
m_file_hnd = ipcdetail::invalid_file();
}
}

inline void file_lock::lock()
{
if(!ipcdetail::acquire_file_lock(m_file_hnd)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
}

inline bool file_lock::try_lock()
{
bool result;
if(!ipcdetail::try_acquire_file_lock(m_file_hnd, result)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
return result;
}

inline bool file_lock::timed_lock(const boost::posix_time::ptime &abs_time)
{  return ipcdetail::try_based_timed_lock(*this, abs_time);   }

inline void file_lock::unlock()
{
if(!ipcdetail::release_file_lock(m_file_hnd)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
}

inline void file_lock::lock_sharable()
{
if(!ipcdetail::acquire_file_lock_sharable(m_file_hnd)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
}

inline bool file_lock::try_lock_sharable()
{
bool result;
if(!ipcdetail::try_acquire_file_lock_sharable(m_file_hnd, result)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
return result;
}

inline bool file_lock::timed_lock_sharable(const boost::posix_time::ptime &abs_time)
{
ipcdetail::lock_to_sharable<file_lock> lsh(*this);
return ipcdetail::try_based_timed_lock(lsh, abs_time);
}

inline void file_lock::unlock_sharable()
{
if(!ipcdetail::release_file_lock_sharable(m_file_hnd)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
}

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
