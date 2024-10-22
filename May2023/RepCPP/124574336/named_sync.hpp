
#ifndef BOOST_INTERPROCESS_WINDOWS_NAMED_SYNC_HPP
#define BOOST_INTERPROCESS_WINDOWS_NAMED_SYNC_HPP

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
#include <boost/interprocess/detail/shared_dir_helpers.hpp>
#include <boost/interprocess/sync/windows/sync_utils.hpp>
#include <boost/interprocess/errors.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <string>
#include <boost/assert.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {

class windows_named_sync_interface
{
public:
virtual std::size_t get_data_size() const = 0;
virtual const void *buffer_with_final_data_to_file() = 0;
virtual const void *buffer_with_init_data_to_file() = 0;
virtual void *buffer_to_store_init_data_from_file() = 0;
virtual bool open(create_enum_t creation_type, const char *id_name) = 0;
virtual void close() = 0;
virtual ~windows_named_sync_interface() = 0;
};

inline windows_named_sync_interface::~windows_named_sync_interface()
{}

class windows_named_sync
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

windows_named_sync(const windows_named_sync &);
windows_named_sync &operator=(const windows_named_sync &);
#endif   

public:
windows_named_sync();
void open_or_create(create_enum_t creation_type, const char *name, const permissions &perm, windows_named_sync_interface &sync_interface);
void close(windows_named_sync_interface &sync_interface);

static bool remove(const char *name);

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
void *m_file_hnd;

#endif   
};

inline windows_named_sync::windows_named_sync()
: m_file_hnd(winapi::invalid_handle_value)
{}

inline void windows_named_sync::close(windows_named_sync_interface &sync_interface)
{
const std::size_t buflen = sync_interface.get_data_size();
const std::size_t sizeof_file_info = sizeof(sync_id::internal_type) + buflen;
winapi::interprocess_overlapped overlapped;
if(winapi::lock_file_ex
(m_file_hnd, winapi::lockfile_exclusive_lock, 0, sizeof_file_info, 0, &overlapped)){
if(winapi::set_file_pointer(m_file_hnd, sizeof(sync_id::internal_type), 0, winapi::file_begin)){
const void *buf = sync_interface.buffer_with_final_data_to_file();

unsigned long written_or_read = 0;
if(winapi::write_file(m_file_hnd, buf, buflen, &written_or_read, 0)){
}
}
}
sync_interface.close();
if(m_file_hnd != winapi::invalid_handle_value){
winapi::close_handle(m_file_hnd);
m_file_hnd = winapi::invalid_handle_value;
}
}

inline void windows_named_sync::open_or_create
( create_enum_t creation_type
, const char *name
, const permissions &perm
, windows_named_sync_interface &sync_interface)
{
std::string aux_str(name);
m_file_hnd  = winapi::invalid_handle_value;
{
create_shared_dir_cleaning_old_and_get_filepath(name, aux_str);
m_file_hnd = winapi::create_file
( aux_str.c_str()
, winapi::generic_read | winapi::generic_write
, creation_type == DoOpen ? winapi::open_existing :
(creation_type == DoCreate ? winapi::create_new : winapi::open_always)
, 0
, (winapi::interprocess_security_attributes*)perm.get_permissions());

error_info err;
bool success = false;
if(m_file_hnd != winapi::invalid_handle_value){
const std::size_t buflen = sync_interface.get_data_size();
typedef __int64 unique_id_type;
const std::size_t sizeof_file_info = sizeof(unique_id_type) + buflen;
winapi::interprocess_overlapped overlapped;
if(winapi::lock_file_ex
(m_file_hnd, winapi::lockfile_exclusive_lock, 0, sizeof_file_info, 0, &overlapped)){
__int64 filesize = 0;
if(winapi::get_file_size(m_file_hnd, filesize)){
unsigned long written_or_read = 0;
unique_id_type unique_id_val;
if(static_cast<std::size_t>(filesize) != sizeof_file_info){
winapi::set_end_of_file(m_file_hnd);
winapi::query_performance_counter(&unique_id_val);
const void *buf = sync_interface.buffer_with_init_data_to_file();
if(winapi::write_file(m_file_hnd, &unique_id_val, sizeof(unique_id_val), &written_or_read, 0)  &&
written_or_read == sizeof(unique_id_val) &&
winapi::write_file(m_file_hnd, buf, buflen, &written_or_read, 0) &&
written_or_read == buflen ){
success = true;
}
winapi::get_file_size(m_file_hnd, filesize);
BOOST_ASSERT(std::size_t(filesize) == sizeof_file_info);
}
else{
void *buf = sync_interface.buffer_to_store_init_data_from_file();
if(winapi::read_file(m_file_hnd, &unique_id_val, sizeof(unique_id_val), &written_or_read, 0)  &&
written_or_read == sizeof(unique_id_val) &&
winapi::read_file(m_file_hnd, buf, buflen, &written_or_read, 0)  &&
written_or_read == buflen   ){
success = true;
}
}
if(success){
char unique_id_name[sizeof(unique_id_val)*2+1];
std::size_t name_suffix_length = sizeof(unique_id_name);
bytes_to_str(&unique_id_val, sizeof(unique_id_val), &unique_id_name[0], name_suffix_length);
success = sync_interface.open(creation_type, unique_id_name);
}
}

err = system_error_code();

if(!winapi::unlock_file_ex(m_file_hnd, 0, sizeof_file_info, 0, &overlapped)){
err = system_error_code();
}
}
else{
err = system_error_code();
}
}
else{
err = system_error_code();
}

if(!success){
if(m_file_hnd != winapi::invalid_handle_value){
winapi::close_handle(m_file_hnd);
m_file_hnd = winapi::invalid_handle_value;
}
throw interprocess_exception(err);
}
}
}

inline bool windows_named_sync::remove(const char *name)
{
try{
std::string semfile;
ipcdetail::shared_filepath(name, semfile);
return winapi::unlink_file(semfile.c_str());
}
catch(...){
return false;
}
}

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
