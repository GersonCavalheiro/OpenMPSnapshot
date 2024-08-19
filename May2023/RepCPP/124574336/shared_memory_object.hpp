
#ifndef BOOST_INTERPROCESS_SHARED_MEMORY_OBJECT_HPP
#define BOOST_INTERPROCESS_SHARED_MEMORY_OBJECT_HPP

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
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/detail/shared_dir_helpers.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/move/adl_move_swap.hpp>
#include <cstddef>
#include <string>

#if defined(BOOST_INTERPROCESS_POSIX_SHARED_MEMORY_OBJECTS)
#  include <fcntl.h>        
#  include <sys/mman.h>     
#  include <unistd.h>       
#  include <sys/stat.h>     
#  if defined(BOOST_INTERPROCESS_RUNTIME_FILESYSTEM_BASED_POSIX_SHARED_MEMORY)
#     if defined(__FreeBSD__)
#        include <sys/sysctl.h>
#     endif
#  endif
#else
#endif


namespace boost {
namespace interprocess {

class shared_memory_object
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
BOOST_MOVABLE_BUT_NOT_COPYABLE(shared_memory_object)
#endif   

public:
shared_memory_object();

shared_memory_object(create_only_t, const char *name, mode_t mode, const permissions &perm = permissions())
{  this->priv_open_or_create(ipcdetail::DoCreate, name, mode, perm);  }

shared_memory_object(open_or_create_t, const char *name, mode_t mode, const permissions &perm = permissions())
{  this->priv_open_or_create(ipcdetail::DoOpenOrCreate, name, mode, perm);  }

shared_memory_object(open_only_t, const char *name, mode_t mode)
{  this->priv_open_or_create(ipcdetail::DoOpen, name, mode, permissions());  }

shared_memory_object(BOOST_RV_REF(shared_memory_object) moved)
:  m_handle(file_handle_t(ipcdetail::invalid_file()))
,  m_mode(read_only)
{  this->swap(moved);   }

shared_memory_object &operator=(BOOST_RV_REF(shared_memory_object) moved)
{
shared_memory_object tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

void swap(shared_memory_object &moved);

static bool remove(const char *name);

void truncate(offset_t length);

~shared_memory_object();

const char *get_name() const;

bool get_size(offset_t &size) const;

mode_t get_mode() const;

mapping_handle_t get_mapping_handle() const;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:

void priv_close();

bool priv_open_or_create(ipcdetail::create_enum_t type, const char *filename, mode_t mode, const permissions &perm);

file_handle_t  m_handle;
mode_t         m_mode;
std::string    m_filename;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

inline shared_memory_object::shared_memory_object()
:  m_handle(file_handle_t(ipcdetail::invalid_file()))
,  m_mode(read_only)
{}

inline shared_memory_object::~shared_memory_object()
{  this->priv_close(); }


inline const char *shared_memory_object::get_name() const
{  return m_filename.c_str(); }

inline bool shared_memory_object::get_size(offset_t &size) const
{  return ipcdetail::get_file_size((file_handle_t)m_handle, size);  }

inline void shared_memory_object::swap(shared_memory_object &other)
{
boost::adl_move_swap(m_handle, other.m_handle);
boost::adl_move_swap(m_mode,   other.m_mode);
m_filename.swap(other.m_filename);
}

inline mapping_handle_t shared_memory_object::get_mapping_handle() const
{
return ipcdetail::mapping_handle_from_file_handle(m_handle);
}

inline mode_t shared_memory_object::get_mode() const
{  return m_mode; }

#if !defined(BOOST_INTERPROCESS_POSIX_SHARED_MEMORY_OBJECTS)

inline bool shared_memory_object::priv_open_or_create
(ipcdetail::create_enum_t type, const char *filename, mode_t mode, const permissions &perm)
{
m_filename = filename;
std::string shmfile;
ipcdetail::create_shared_dir_cleaning_old_and_get_filepath(filename, shmfile);

if (mode != read_write && mode != read_only){
error_info err = other_error;
throw interprocess_exception(err);
}

switch(type){
case ipcdetail::DoOpen:
m_handle = ipcdetail::open_existing_file(shmfile.c_str(), mode, true);
break;
case ipcdetail::DoCreate:
m_handle = ipcdetail::create_new_file(shmfile.c_str(), mode, perm, true);
break;
case ipcdetail::DoOpenOrCreate:
m_handle = ipcdetail::create_or_open_file(shmfile.c_str(), mode, perm, true);
break;
default:
{
error_info err = other_error;
throw interprocess_exception(err);
}
}

if(m_handle == ipcdetail::invalid_file()){
error_info err = system_error_code();
this->priv_close();
throw interprocess_exception(err);
}

m_mode = mode;
return true;
}

inline bool shared_memory_object::remove(const char *filename)
{
try{
std::string shmfile;
ipcdetail::shared_filepath(filename, shmfile);
return ipcdetail::delete_file(shmfile.c_str());
}
catch(...){
return false;
}
}

inline void shared_memory_object::truncate(offset_t length)
{
if(!ipcdetail::truncate_file(m_handle, length)){
error_info err = system_error_code();
throw interprocess_exception(err);
}
}

inline void shared_memory_object::priv_close()
{
if(m_handle != ipcdetail::invalid_file()){
ipcdetail::close_file(m_handle);
m_handle = ipcdetail::invalid_file();
}
}

#else 

namespace shared_memory_object_detail {

#ifdef BOOST_INTERPROCESS_RUNTIME_FILESYSTEM_BASED_POSIX_SHARED_MEMORY

#if defined(__FreeBSD__)

inline bool use_filesystem_based_posix()
{
int jailed = 0;
std::size_t len = sizeof(jailed);
::sysctlbyname("security.jail.jailed", &jailed, &len, NULL, 0);
return jailed != 0;
}

#else
#error "Not supported platform for BOOST_INTERPROCESS_RUNTIME_FILESYSTEM_BASED_POSIX_SHARED_MEMORY"
#endif

#endif

}  

inline bool shared_memory_object::priv_open_or_create
(ipcdetail::create_enum_t type,
const char *filename,
mode_t mode, const permissions &perm)
{
#if defined(BOOST_INTERPROCESS_FILESYSTEM_BASED_POSIX_SHARED_MEMORY)
const bool add_leading_slash = false;
#elif defined(BOOST_INTERPROCESS_RUNTIME_FILESYSTEM_BASED_POSIX_SHARED_MEMORY)
const bool add_leading_slash = !shared_memory_object_detail::use_filesystem_based_posix();
#else
const bool add_leading_slash = true;
#endif
if(add_leading_slash){
ipcdetail::add_leading_slash(filename, m_filename);
}
else{
ipcdetail::create_shared_dir_cleaning_old_and_get_filepath(filename, m_filename);
}

int oflag = 0;
if(mode == read_only){
oflag |= O_RDONLY;
}
else if(mode == read_write){
oflag |= O_RDWR;
}
else{
error_info err(mode_error);
throw interprocess_exception(err);
}
int unix_perm = perm.get_permissions();

switch(type){
case ipcdetail::DoOpen:
{
m_handle = shm_open(m_filename.c_str(), oflag, unix_perm);
}
break;
case ipcdetail::DoCreate:
{
oflag |= (O_CREAT | O_EXCL);
m_handle = shm_open(m_filename.c_str(), oflag, unix_perm);
if(m_handle >= 0){
::fchmod(m_handle, unix_perm);
}
}
break;
case ipcdetail::DoOpenOrCreate:
{
while(true){
m_handle = shm_open(m_filename.c_str(), oflag | (O_CREAT | O_EXCL), unix_perm);
if(m_handle >= 0){
::fchmod(m_handle, unix_perm);
}
else if(errno == EEXIST){
m_handle = shm_open(m_filename.c_str(), oflag, unix_perm);
if(m_handle < 0 && errno == ENOENT){
continue;
}
}
break;
}
}
break;
default:
{
error_info err = other_error;
throw interprocess_exception(err);
}
}

if(m_handle < 0){
error_info err = errno;
this->priv_close();
throw interprocess_exception(err);
}

m_filename = filename;
m_mode = mode;
return true;
}

inline bool shared_memory_object::remove(const char *filename)
{
try{
std::string filepath;
#if defined(BOOST_INTERPROCESS_FILESYSTEM_BASED_POSIX_SHARED_MEMORY)
const bool add_leading_slash = false;
#elif defined(BOOST_INTERPROCESS_RUNTIME_FILESYSTEM_BASED_POSIX_SHARED_MEMORY)
const bool add_leading_slash = !shared_memory_object_detail::use_filesystem_based_posix();
#else
const bool add_leading_slash = true;
#endif
if(add_leading_slash){
ipcdetail::add_leading_slash(filename, filepath);
}
else{
ipcdetail::shared_filepath(filename, filepath);
}
return 0 == shm_unlink(filepath.c_str());
}
catch(...){
return false;
}
}

inline void shared_memory_object::truncate(offset_t length)
{
if(0 != ftruncate(m_handle, length)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
}

inline void shared_memory_object::priv_close()
{
if(m_handle != -1){
::close(m_handle);
m_handle = -1;
}
}

#endif

class remove_shared_memory_on_destroy
{
const char * m_name;
public:
remove_shared_memory_on_destroy(const char *name)
:  m_name(name)
{}

~remove_shared_memory_on_destroy()
{  shared_memory_object::remove(m_name);  }
};

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
