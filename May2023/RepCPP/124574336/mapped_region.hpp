
#ifndef BOOST_INTERPROCESS_MAPPED_REGION_HPP
#define BOOST_INTERPROCESS_MAPPED_REGION_HPP

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
#include <boost/interprocess/exceptions.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <string>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/move/adl_move_swap.hpp>

#if defined(sun) || defined(__sun) || defined(__osf__) || defined(__osf) || defined(_hpux) || defined(hpux) || defined(_AIX)
#define BOOST_INTERPROCESS_MADVISE_USES_CADDR_T
#include <sys/types.h>
#endif

#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
#define BOOST_INTERPROCESS_MADV_DONTNEED_HAS_NONDESTRUCTIVE_SEMANTICS
#endif

#if defined (BOOST_INTERPROCESS_WINDOWS)
#  include <boost/interprocess/detail/win32_api.hpp>
#  include <boost/interprocess/sync/windows/sync_utils.hpp>
#else
#  ifdef BOOST_HAS_UNISTD_H
#    include <fcntl.h>
#    include <sys/mman.h>     
#    include <unistd.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    if defined(BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS)
#      include <sys/shm.h>      
#    endif
#    include <boost/assert.hpp>
#  else
#    error Unknown platform
#  endif

#endif   


namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#if (defined(sun) || defined(__sun)) && defined(MADV_NORMAL)
extern "C" int madvise(caddr_t, size_t, int);
#endif

namespace ipcdetail{ class interprocess_tester; }
namespace ipcdetail{ class raw_mapped_region_creator; }

#endif   

class mapped_region
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
BOOST_MOVABLE_BUT_NOT_COPYABLE(mapped_region)
#endif   

public:

template<class MemoryMappable>
mapped_region(const MemoryMappable& mapping
,mode_t mode
,offset_t offset = 0
,std::size_t size = 0
,const void *address = 0
,map_options_t map_options = default_map_options);

mapped_region();

mapped_region(BOOST_RV_REF(mapped_region) other)
#if defined (BOOST_INTERPROCESS_WINDOWS)
:  m_base(0), m_size(0)
,  m_page_offset(0)
,  m_mode(read_only)
,  m_file_or_mapping_hnd(ipcdetail::invalid_file())
#else
:  m_base(0), m_size(0), m_page_offset(0), m_mode(read_only), m_is_xsi(false)
#endif
{  this->swap(other);   }

~mapped_region();

mapped_region &operator=(BOOST_RV_REF(mapped_region) other)
{
mapped_region tmp(boost::move(other));
this->swap(tmp);
return *this;
}

void swap(mapped_region &other);

std::size_t get_size() const;

void*       get_address() const;

mode_t get_mode() const;

bool flush(std::size_t mapping_offset = 0, std::size_t numbytes = 0, bool async = true);

bool shrink_by(std::size_t bytes, bool from_back = true);

enum advice_types{
advice_normal,
advice_sequential,
advice_random,
advice_willneed,
advice_dontneed
};

bool advise(advice_types advise);

static std::size_t get_page_size();

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
void priv_close();

void* priv_map_address()  const;
std::size_t priv_map_size()  const;
bool priv_flush_param_check(std::size_t mapping_offset, void *&addr, std::size_t &numbytes) const;
bool priv_shrink_param_check(std::size_t bytes, bool from_back, void *&shrink_page_start, std::size_t &shrink_page_bytes);
static void priv_size_from_mapping_size
(offset_t mapping_size, offset_t offset, offset_t page_offset, std::size_t &size);
static offset_t priv_page_offset_addr_fixup(offset_t page_offset, const void *&addr);

template<int dummy>
struct page_size_holder
{
static const std::size_t PageSize;
static std::size_t get_page_size();
};

void*             m_base;
std::size_t       m_size;
std::size_t       m_page_offset;
mode_t            m_mode;
#if defined(BOOST_INTERPROCESS_WINDOWS)
file_handle_t     m_file_or_mapping_hnd;
#else
bool              m_is_xsi;
#endif

friend class ipcdetail::interprocess_tester;
friend class ipcdetail::raw_mapped_region_creator;
void dont_close_on_destruction();
#if defined(BOOST_INTERPROCESS_WINDOWS) && !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION)
template<int Dummy>
static void destroy_syncs_in_range(const void *addr, std::size_t size);
#endif
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

inline void swap(mapped_region &x, mapped_region &y)
{  x.swap(y);  }

inline mapped_region::~mapped_region()
{  this->priv_close(); }

inline std::size_t mapped_region::get_size()  const
{  return m_size; }

inline mode_t mapped_region::get_mode()  const
{  return m_mode;   }

inline void*    mapped_region::get_address()  const
{  return m_base; }

inline void*    mapped_region::priv_map_address()  const
{  return static_cast<char*>(m_base) - m_page_offset; }

inline std::size_t mapped_region::priv_map_size()  const
{  return m_size + m_page_offset; }

inline bool mapped_region::priv_flush_param_check
(std::size_t mapping_offset, void *&addr, std::size_t &numbytes) const
{
if(m_base == 0)
return false;

if(mapping_offset >= m_size || numbytes > (m_size - size_t(mapping_offset))){
return false;
}

if(numbytes == 0){
numbytes = m_size - mapping_offset;
}
addr = (char*)this->priv_map_address() + mapping_offset;
numbytes += m_page_offset;
return true;
}

inline bool mapped_region::priv_shrink_param_check
(std::size_t bytes, bool from_back, void *&shrink_page_start, std::size_t &shrink_page_bytes)
{
if(m_base == 0 || bytes > m_size){
return false;
}
else if(bytes == m_size){
this->priv_close();
return true;
}
else{
const std::size_t page_size = mapped_region::get_page_size();
if(from_back){
const std::size_t new_pages = (m_size + m_page_offset - bytes - 1)/page_size + 1;
shrink_page_start = static_cast<char*>(this->priv_map_address()) + new_pages*page_size;
shrink_page_bytes = m_page_offset + m_size - new_pages*page_size;
m_size -= bytes;
}
else{
shrink_page_start = this->priv_map_address();
m_page_offset += bytes;
shrink_page_bytes = (m_page_offset/page_size)*page_size;
m_page_offset = m_page_offset % page_size;
m_size -= bytes;
m_base  = static_cast<char *>(m_base) + bytes;
BOOST_ASSERT(shrink_page_bytes%page_size == 0);
}
return true;
}
}

inline void mapped_region::priv_size_from_mapping_size
(offset_t mapping_size, offset_t offset, offset_t page_offset, std::size_t &size)
{
if(mapping_size < offset ||
boost::uintmax_t(mapping_size - (offset - page_offset)) >
boost::uintmax_t(std::size_t(-1))){
error_info err(size_error);
throw interprocess_exception(err);
}
size = static_cast<std::size_t>(mapping_size - offset);
}

inline offset_t mapped_region::priv_page_offset_addr_fixup(offset_t offset, const void *&address)
{
const std::size_t page_size  = mapped_region::get_page_size();

const std::size_t page_offset =
static_cast<std::size_t>(offset - (offset / page_size) * page_size);
if(address){
address = static_cast<const char*>(address) - page_offset;
}
return page_offset;
}

#if defined (BOOST_INTERPROCESS_WINDOWS)

inline mapped_region::mapped_region()
:  m_base(0), m_size(0), m_page_offset(0), m_mode(read_only)
,  m_file_or_mapping_hnd(ipcdetail::invalid_file())
{}

template<int dummy>
inline std::size_t mapped_region::page_size_holder<dummy>::get_page_size()
{
winapi::interprocess_system_info info;
winapi::get_system_info(&info);
return std::size_t(info.dwAllocationGranularity);
}

template<class MemoryMappable>
inline mapped_region::mapped_region
(const MemoryMappable &mapping
,mode_t mode
,offset_t offset
,std::size_t size
,const void *address
,map_options_t map_options)
:  m_base(0), m_size(0), m_page_offset(0), m_mode(mode)
,  m_file_or_mapping_hnd(ipcdetail::invalid_file())
{
mapping_handle_t mhandle = mapping.get_mapping_handle();
{
file_handle_t native_mapping_handle = 0;

unsigned long protection = 0;
unsigned long map_access = map_options == default_map_options ? 0 : map_options;

switch(mode)
{
case read_only:
case read_private:
protection   |= winapi::page_readonly;
map_access   |= winapi::file_map_read;
break;
case read_write:
protection   |= winapi::page_readwrite;
map_access   |= winapi::file_map_write;
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

void * handle_to_close = winapi::invalid_handle_value;
if(!mhandle.is_shm){
native_mapping_handle = winapi::create_file_mapping
( ipcdetail::file_handle_from_mapping_handle(mapping.get_mapping_handle())
, protection, 0, 0, 0);

if(!native_mapping_handle){
error_info err = winapi::get_last_error();
throw interprocess_exception(err);
}
handle_to_close = native_mapping_handle;
}
else{
native_mapping_handle = mhandle.handle;
}
const winapi::handle_closer close_handle(handle_to_close);
(void)close_handle;

const offset_t page_offset = priv_page_offset_addr_fixup(offset, address);

if(size == 0){
offset_t mapping_size;
if(!winapi::get_file_mapping_size(native_mapping_handle, mapping_size)){
error_info err = winapi::get_last_error();
throw interprocess_exception(err);
}
priv_size_from_mapping_size(mapping_size, offset, page_offset, size);
}

void *base = winapi::map_view_of_file_ex
(native_mapping_handle,
map_access,
offset - page_offset,
static_cast<std::size_t>(page_offset + size),
const_cast<void*>(address));
if(!base){
error_info err = winapi::get_last_error();
throw interprocess_exception(err);
}

m_base = static_cast<char*>(base) + page_offset;
m_page_offset = page_offset;
m_size = size;
}
if(!winapi::duplicate_current_process_handle(mhandle.handle, &m_file_or_mapping_hnd)){
error_info err = winapi::get_last_error();
this->priv_close();
throw interprocess_exception(err);
}
}

inline bool mapped_region::flush(std::size_t mapping_offset, std::size_t numbytes, bool async)
{
void *addr;
if(!this->priv_flush_param_check(mapping_offset, addr, numbytes)){
return false;
}
if(!winapi::flush_view_of_file(addr, numbytes)){
return false;
}
else if(!async && m_file_or_mapping_hnd != winapi::invalid_handle_value &&
winapi::get_file_type(m_file_or_mapping_hnd) == winapi::file_type_disk){
return winapi::flush_file_buffers(m_file_or_mapping_hnd);
}
return true;
}

inline bool mapped_region::shrink_by(std::size_t bytes, bool from_back)
{
void *shrink_page_start = 0;
std::size_t shrink_page_bytes = 0;
if(!this->priv_shrink_param_check(bytes, from_back, shrink_page_start, shrink_page_bytes)){
return false;
}
else if(shrink_page_bytes){
unsigned long old_protect_ignored;
bool b_ret = winapi::virtual_unlock(shrink_page_start, shrink_page_bytes)
|| (winapi::get_last_error() == winapi::error_not_locked);
(void)old_protect_ignored;
b_ret = b_ret && winapi::virtual_protect
(shrink_page_start, shrink_page_bytes, winapi::page_noaccess, old_protect_ignored);
return b_ret;
}
else{
return true;
}
}

inline bool mapped_region::advise(advice_types)
{
return false;
}

inline void mapped_region::priv_close()
{
if(m_base){
void *addr = this->priv_map_address();
#if !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION)
mapped_region::destroy_syncs_in_range<0>(addr, m_size);
#endif
winapi::unmap_view_of_file(addr);
m_base = 0;
}
if(m_file_or_mapping_hnd != ipcdetail::invalid_file()){
winapi::close_handle(m_file_or_mapping_hnd);
m_file_or_mapping_hnd = ipcdetail::invalid_file();
}
}

inline void mapped_region::dont_close_on_destruction()
{}

#else    

inline mapped_region::mapped_region()
:  m_base(0), m_size(0), m_page_offset(0), m_mode(read_only), m_is_xsi(false)
{}

template<int dummy>
inline std::size_t mapped_region::page_size_holder<dummy>::get_page_size()
{  return std::size_t(sysconf(_SC_PAGESIZE)); }

template<class MemoryMappable>
inline mapped_region::mapped_region
( const MemoryMappable &mapping
, mode_t mode
, offset_t offset
, std::size_t size
, const void *address
, map_options_t map_options)
: m_base(0), m_size(0), m_page_offset(0), m_mode(mode), m_is_xsi(false)
{
mapping_handle_t map_hnd = mapping.get_mapping_handle();

#ifdef BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS
if(map_hnd.is_xsi){
::shmid_ds xsi_ds;
int ret = ::shmctl(map_hnd.handle, IPC_STAT, &xsi_ds);
if(ret == -1){
error_info err(system_error_code());
throw interprocess_exception(err);
}
if(size == 0){
size = (std::size_t)xsi_ds.shm_segsz;
}
else if(size != (std::size_t)xsi_ds.shm_segsz){
error_info err(size_error);
throw interprocess_exception(err);
}
int flag = map_options == default_map_options ? 0 : map_options;
if(m_mode == read_only){
flag |= SHM_RDONLY;
}
else if(m_mode != read_write){
error_info err(mode_error);
throw interprocess_exception(err);
}
void *const final_address = const_cast<void *>(address);
void *base = ::shmat(map_hnd.handle, final_address, flag);
if(base == (void*)-1){
error_info err(system_error_code());
throw interprocess_exception(err);
}
m_base   = base;
m_size   = size;
m_mode   = mode;
m_page_offset = 0;
m_is_xsi = true;
return;
}
#endif   

const offset_t page_offset = priv_page_offset_addr_fixup(offset, address);

if(size == 0){
struct ::stat buf;
if(0 != fstat(map_hnd.handle, &buf)){
error_info err(system_error_code());
throw interprocess_exception(err);
}
priv_size_from_mapping_size(buf.st_size, offset, page_offset, size);
}

#ifdef MAP_NOSYNC
#define BOOST_INTERPROCESS_MAP_NOSYNC MAP_NOSYNC
#else
#define BOOST_INTERPROCESS_MAP_NOSYNC 0
#endif   

int prot    = 0;
int flags   = map_options == default_map_options ? BOOST_INTERPROCESS_MAP_NOSYNC : map_options;

#undef BOOST_INTERPROCESS_MAP_NOSYNC

switch(mode)
{
case read_only:
prot  |= PROT_READ;
flags |= MAP_SHARED;
break;

case read_private:
prot  |= (PROT_READ);
flags |= MAP_PRIVATE;
break;

case read_write:
prot  |= (PROT_WRITE | PROT_READ);
flags |= MAP_SHARED;
break;

case copy_on_write:
prot  |= (PROT_WRITE | PROT_READ);
flags |= MAP_PRIVATE;
break;

default:
{
error_info err(mode_error);
throw interprocess_exception(err);
}
break;
}

void* base = mmap ( const_cast<void*>(address)
, static_cast<std::size_t>(page_offset + size)
, prot
, flags
, mapping.get_mapping_handle().handle
, offset - page_offset);

if(base == MAP_FAILED){
error_info err = system_error_code();
throw interprocess_exception(err);
}

m_base = static_cast<char*>(base) + page_offset;
m_page_offset = page_offset;
m_size   = size;

if(address && (base != address)){
error_info err(busy_error);
this->priv_close();
throw interprocess_exception(err);
}
}

inline bool mapped_region::shrink_by(std::size_t bytes, bool from_back)
{
void *shrink_page_start = 0;
std::size_t shrink_page_bytes = 0;
if(m_is_xsi || !this->priv_shrink_param_check(bytes, from_back, shrink_page_start, shrink_page_bytes)){
return false;
}
else if(shrink_page_bytes){
return 0 == munmap(shrink_page_start, shrink_page_bytes);
}
else{
return true;
}
}

inline bool mapped_region::flush(std::size_t mapping_offset, std::size_t numbytes, bool async)
{
void *addr;
if(m_is_xsi || !this->priv_flush_param_check(mapping_offset, addr, numbytes)){
return false;
}
return msync(addr, numbytes, async ? MS_ASYNC : MS_SYNC) == 0;
}

inline bool mapped_region::advise(advice_types advice)
{
int unix_advice = 0;
const unsigned int mode_none = 0;
const unsigned int mode_padv = 1;
const unsigned int mode_madv = 2;
(void)mode_padv;
(void)mode_madv;
unsigned int mode = mode_none;
switch(advice){
case advice_normal:
#if defined(POSIX_MADV_NORMAL)
unix_advice = POSIX_MADV_NORMAL;
mode = mode_padv;
#elif defined(MADV_NORMAL)
unix_advice = MADV_NORMAL;
mode = mode_madv;
#endif
break;
case advice_sequential:
#if defined(POSIX_MADV_SEQUENTIAL)
unix_advice = POSIX_MADV_SEQUENTIAL;
mode = mode_padv;
#elif defined(MADV_SEQUENTIAL)
unix_advice = MADV_SEQUENTIAL;
mode = mode_madv;
#endif
break;
case advice_random:
#if defined(POSIX_MADV_RANDOM)
unix_advice = POSIX_MADV_RANDOM;
mode = mode_padv;
#elif defined(MADV_RANDOM)
unix_advice = MADV_RANDOM;
mode = mode_madv;
#endif
break;
case advice_willneed:
#if defined(POSIX_MADV_WILLNEED)
unix_advice = POSIX_MADV_WILLNEED;
mode = mode_padv;
#elif defined(MADV_WILLNEED)
unix_advice = MADV_WILLNEED;
mode = mode_madv;
#endif
break;
case advice_dontneed:
#if defined(POSIX_MADV_DONTNEED)
unix_advice = POSIX_MADV_DONTNEED;
mode = mode_padv;
#elif defined(MADV_DONTNEED) && defined(BOOST_INTERPROCESS_MADV_DONTNEED_HAS_NONDESTRUCTIVE_SEMANTICS)
unix_advice = MADV_DONTNEED;
mode = mode_madv;
#endif
break;
default:
return false;
}
switch(mode){
#if defined(POSIX_MADV_NORMAL)
case mode_padv:
return 0 == posix_madvise(this->priv_map_address(), this->priv_map_size(), unix_advice);
#endif
#if defined(MADV_NORMAL)
case mode_madv:
return 0 == madvise(
#if defined(BOOST_INTERPROCESS_MADVISE_USES_CADDR_T)
(caddr_t)
#endif
this->priv_map_address(), this->priv_map_size(), unix_advice);
#endif
default:
return false;

}
}

inline void mapped_region::priv_close()
{
if(m_base != 0){
#ifdef BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS
if(m_is_xsi){
int ret = ::shmdt(m_base);
BOOST_ASSERT(ret == 0);
(void)ret;
return;
}
#endif 
munmap(this->priv_map_address(), this->priv_map_size());
m_base = 0;
}
}

inline void mapped_region::dont_close_on_destruction()
{  m_base = 0;   }

#endif   

template<int dummy>
const std::size_t mapped_region::page_size_holder<dummy>::PageSize
= mapped_region::page_size_holder<dummy>::get_page_size();

inline std::size_t mapped_region::get_page_size()
{
if(!page_size_holder<0>::PageSize)
return page_size_holder<0>::get_page_size();
else
return page_size_holder<0>::PageSize;
}

inline void mapped_region::swap(mapped_region &other)
{
::boost::adl_move_swap(this->m_base, other.m_base);
::boost::adl_move_swap(this->m_size, other.m_size);
::boost::adl_move_swap(this->m_page_offset, other.m_page_offset);
::boost::adl_move_swap(this->m_mode,  other.m_mode);
#if defined (BOOST_INTERPROCESS_WINDOWS)
::boost::adl_move_swap(this->m_file_or_mapping_hnd, other.m_file_or_mapping_hnd);
#else
::boost::adl_move_swap(this->m_is_xsi, other.m_is_xsi);
#endif
}

struct null_mapped_region_function
{
bool operator()(void *, std::size_t , bool) const
{   return true;   }

static std::size_t get_min_size()
{  return 0;  }
};

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#ifndef BOOST_INTERPROCESS_MAPPED_REGION_EXT_HPP
#define BOOST_INTERPROCESS_MAPPED_REGION_EXT_HPP

#if defined(BOOST_INTERPROCESS_WINDOWS) && !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION)
#  include <boost/interprocess/sync/windows/sync_utils.hpp>
#  include <boost/interprocess/detail/windows_intermodule_singleton.hpp>

namespace boost {
namespace interprocess {

template<int Dummy>
inline void mapped_region::destroy_syncs_in_range(const void *addr, std::size_t size)
{
ipcdetail::sync_handles &handles =
ipcdetail::windows_intermodule_singleton<ipcdetail::sync_handles>::get();
handles.destroy_syncs_in_range(addr, size);
}

}  
}  

#endif   

#endif   

#endif   

