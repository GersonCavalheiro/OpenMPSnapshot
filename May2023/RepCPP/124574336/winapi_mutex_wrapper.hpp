
#ifndef BOOST_INTERPROCESS_DETAIL_WINAPI_MUTEX_WRAPPER_HPP
#define BOOST_INTERPROCESS_DETAIL_WINAPI_MUTEX_WRAPPER_HPP

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
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/detail/win32_api.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/windows/winapi_wrapper_common.hpp>
#include <boost/interprocess/errors.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <limits>

namespace boost {
namespace interprocess {
namespace ipcdetail {

class winapi_mutex_functions
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

winapi_mutex_functions(const winapi_mutex_functions &);
winapi_mutex_functions &operator=(const winapi_mutex_functions &);
#endif   

public:
winapi_mutex_functions(void *mtx_hnd)
: m_mtx_hnd(mtx_hnd)
{}

void unlock()
{  winapi::release_mutex(m_mtx_hnd);   }

void lock()
{  return winapi_wrapper_wait_for_single_object(m_mtx_hnd);  }

bool try_lock()
{  return winapi_wrapper_try_wait_for_single_object(m_mtx_hnd);  }

bool timed_lock(const boost::posix_time::ptime &abs_time)
{  return winapi_wrapper_timed_wait_for_single_object(m_mtx_hnd, abs_time);  }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
protected:
void *m_mtx_hnd;
#endif   
};

class winapi_mutex_wrapper
: public winapi_mutex_functions
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

winapi_mutex_wrapper(const winapi_mutex_wrapper &);
winapi_mutex_wrapper &operator=(const winapi_mutex_wrapper &);
#endif   


public:
winapi_mutex_wrapper(void *mtx_hnd = 0)
: winapi_mutex_functions(mtx_hnd)
{}

~winapi_mutex_wrapper()
{  this->close(); }

void *release()
{
void *hnd = m_mtx_hnd;
m_mtx_hnd = 0;
return hnd;
}

void *handle() const
{  return m_mtx_hnd; }

bool open_or_create(const char *name, const permissions &perm)
{
if(m_mtx_hnd == 0){
m_mtx_hnd = winapi::open_or_create_mutex
( name
, false
, (winapi::interprocess_security_attributes*)perm.get_permissions()
);
return m_mtx_hnd != 0;
}
else{
return false;
}
}

void close()
{
if(m_mtx_hnd != 0){
winapi::close_handle(m_mtx_hnd);
m_mtx_hnd = 0;
}
}

void swap(winapi_mutex_wrapper &other)
{  void *tmp = m_mtx_hnd; m_mtx_hnd = other.m_mtx_hnd; other.m_mtx_hnd = tmp;   }
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
