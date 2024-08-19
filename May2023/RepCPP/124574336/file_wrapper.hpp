
#ifndef BOOST_INTERPROCESS_DETAIL_FILE_WRAPPER_HPP
#define BOOST_INTERPROCESS_DETAIL_FILE_WRAPPER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/simple_swap.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail{

class file_wrapper
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
BOOST_MOVABLE_BUT_NOT_COPYABLE(file_wrapper)
#endif   
public:

file_wrapper();

file_wrapper(create_only_t, const char *name, mode_t mode, const permissions &perm = permissions())
{  this->priv_open_or_create(ipcdetail::DoCreate, name, mode, perm);  }

file_wrapper(open_or_create_t, const char *name, mode_t mode, const permissions &perm  = permissions())
{  this->priv_open_or_create(ipcdetail::DoOpenOrCreate, name, mode, perm);  }

file_wrapper(open_only_t, const char *name, mode_t mode)
{  this->priv_open_or_create(ipcdetail::DoOpen, name, mode, permissions());  }

file_wrapper(BOOST_RV_REF(file_wrapper) moved)
:  m_handle(file_handle_t(ipcdetail::invalid_file()))
{  this->swap(moved);   }

file_wrapper &operator=(BOOST_RV_REF(file_wrapper) moved)
{
file_wrapper tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

void swap(file_wrapper &other);

static bool remove(const char *name);

void truncate(offset_t length);

~file_wrapper();

const char *get_name() const;

bool get_size(offset_t &size) const;

mode_t get_mode() const;

mapping_handle_t get_mapping_handle() const;

private:
void priv_close();
bool priv_open_or_create(ipcdetail::create_enum_t type, const char *filename, mode_t mode, const permissions &perm);

file_handle_t  m_handle;
mode_t         m_mode;
std::string    m_filename;
};

inline file_wrapper::file_wrapper()
: m_handle(file_handle_t(ipcdetail::invalid_file()))
, m_mode(read_only), m_filename()
{}

inline file_wrapper::~file_wrapper()
{  this->priv_close(); }

inline const char *file_wrapper::get_name() const
{  return m_filename.c_str(); }

inline bool file_wrapper::get_size(offset_t &size) const
{  return get_file_size((file_handle_t)m_handle, size);  }

inline void file_wrapper::swap(file_wrapper &other)
{
(simple_swap)(m_handle,  other.m_handle);
(simple_swap)(m_mode,    other.m_mode);
m_filename.swap(other.m_filename);
}

inline mapping_handle_t file_wrapper::get_mapping_handle() const
{  return mapping_handle_from_file_handle(m_handle);  }

inline mode_t file_wrapper::get_mode() const
{  return m_mode; }

inline bool file_wrapper::priv_open_or_create
(ipcdetail::create_enum_t type,
const char *filename,
mode_t mode,
const permissions &perm = permissions())
{
m_filename = filename;

if(mode != read_only && mode != read_write){
error_info err(mode_error);
throw interprocess_exception(err);
}

switch(type){
case ipcdetail::DoOpen:
m_handle = open_existing_file(filename, mode);
break;
case ipcdetail::DoCreate:
m_handle = create_new_file(filename, mode, perm);
break;
case ipcdetail::DoOpenOrCreate:
m_handle = create_or_open_file(filename, mode, perm);
break;
default:
{
error_info err = other_error;
throw interprocess_exception(err);
}
}

if(m_handle == invalid_file()){
error_info err = system_error_code();
throw interprocess_exception(err);
}

m_mode = mode;
return true;
}

inline bool file_wrapper::remove(const char *filename)
{  return delete_file(filename); }

inline void file_wrapper::truncate(offset_t length)
{
if(!truncate_file(m_handle, length)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
}

inline void file_wrapper::priv_close()
{
if(m_handle != invalid_file()){
close_file(m_handle);
m_handle = invalid_file();
}
}

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
