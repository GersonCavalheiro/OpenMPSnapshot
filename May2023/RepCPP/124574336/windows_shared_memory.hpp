
#ifndef BOOST_INTERPROCESS_WINDOWS_SHARED_MEMORY_HPP
#define BOOST_INTERPROCESS_WINDOWS_SHARED_MEMORY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/detail/simple_swap.hpp>

#if !defined(BOOST_INTERPROCESS_WINDOWS)
#error "This header can only be used in Windows operating systems"
#endif

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/win32_api.hpp>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <string>


namespace boost {
namespace interprocess {

class windows_shared_memory
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
BOOST_MOVABLE_BUT_NOT_COPYABLE(windows_shared_memory)
#endif   

public:
windows_shared_memory();

windows_shared_memory(create_only_t, const char *name, mode_t mode, std::size_t size, const permissions& perm = permissions())
{  this->priv_open_or_create(ipcdetail::DoCreate, name, mode, size, perm);  }

windows_shared_memory(open_or_create_t, const char *name, mode_t mode, std::size_t size, const permissions& perm = permissions())
{  this->priv_open_or_create(ipcdetail::DoOpenOrCreate, name, mode, size, perm);  }

windows_shared_memory(open_only_t, const char *name, mode_t mode)
{  this->priv_open_or_create(ipcdetail::DoOpen, name, mode, 0, permissions());  }

windows_shared_memory(BOOST_RV_REF(windows_shared_memory) moved)
: m_handle(0)
{  this->swap(moved);   }

windows_shared_memory &operator=(BOOST_RV_REF(windows_shared_memory) moved)
{
windows_shared_memory tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

void swap(windows_shared_memory &other);

~windows_shared_memory();

const char *get_name() const;

mode_t get_mode() const;

mapping_handle_t get_mapping_handle() const;

offset_t get_size() const;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:

void priv_close();

bool priv_open_or_create(ipcdetail::create_enum_t type, const char *filename, mode_t mode, std::size_t size, const permissions& perm = permissions());

void *         m_handle;
mode_t         m_mode;
std::string    m_name;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

inline windows_shared_memory::windows_shared_memory()
:  m_handle(0)
{}

inline windows_shared_memory::~windows_shared_memory()
{  this->priv_close(); }

inline const char *windows_shared_memory::get_name() const
{  return m_name.c_str(); }

inline void windows_shared_memory::swap(windows_shared_memory &other)
{
(simple_swap)(m_handle,  other.m_handle);
(simple_swap)(m_mode,    other.m_mode);
m_name.swap(other.m_name);
}

inline mapping_handle_t windows_shared_memory::get_mapping_handle() const
{  mapping_handle_t mhnd = { m_handle, true};   return mhnd;   }

inline mode_t windows_shared_memory::get_mode() const
{  return m_mode; }

inline offset_t windows_shared_memory::get_size() const
{
offset_t size; 
return (m_handle && winapi::get_file_mapping_size(m_handle, size)) ? size : 0;
}

inline bool windows_shared_memory::priv_open_or_create
(ipcdetail::create_enum_t type, const char *filename, mode_t mode, std::size_t size, const permissions& perm)
{
m_name = filename ? filename : "";

unsigned long protection = 0;
unsigned long map_access = 0;

switch(mode)
{
case read_only:
protection   |= winapi::page_readonly;
map_access   |= winapi::file_map_read | winapi::section_query;
break;
case read_write:
protection   |= winapi::page_readwrite;
map_access   |= winapi::file_map_write | winapi::section_query;
break;
case copy_on_write:
protection   |= winapi::page_writecopy;
map_access   |= winapi::file_map_copy;
break;
default:
{
error_info err(mode_error);
throw interprocess_exception(err);
}
break;
}

switch(type){
case ipcdetail::DoOpen:
m_handle = winapi::open_file_mapping(map_access, filename);
break;
case ipcdetail::DoCreate:
case ipcdetail::DoOpenOrCreate:
{
m_handle = winapi::create_file_mapping
( winapi::invalid_handle_value, protection, size, filename
, (winapi::interprocess_security_attributes*)perm.get_permissions());
}
break;
default:
{
error_info err = other_error;
throw interprocess_exception(err);
}
}

if(!m_handle || (type == ipcdetail::DoCreate && winapi::get_last_error() == winapi::error_already_exists)){
error_info err = system_error_code();
this->priv_close();
throw interprocess_exception(err);
}

m_mode = mode;
return true;
}

inline void windows_shared_memory::priv_close()
{
if(m_handle){
winapi::close_handle(m_handle);
m_handle = 0;
}
}

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
