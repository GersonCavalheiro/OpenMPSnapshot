
#ifndef BOOST_INTERPROCESS_ROBUST_EMULATION_HPP
#define BOOST_INTERPROCESS_ROBUST_EMULATION_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_recursive_mutex.hpp>
#include <boost/interprocess/detail/atomic.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/detail/shared_dir_helpers.hpp>
#include <boost/interprocess/detail/intermodule_singleton.hpp>
#include <boost/interprocess/detail/portable_intermodule_singleton.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/sync/spin/wait.hpp>
#include <boost/interprocess/sync/detail/common_algorithms.hpp>
#include <string>

namespace boost{
namespace interprocess{
namespace ipcdetail{

namespace robust_emulation_helpers {

template<class T>
class mutex_traits
{
public:
static void take_ownership(T &t)
{  t.take_ownership(); }
};

inline void remove_if_can_lock_file(const char *file_path)
{
file_handle_t fhnd = open_existing_file(file_path, read_write);

if(fhnd != invalid_file()){
bool acquired;
if(try_acquire_file_lock(fhnd, acquired) && acquired){
delete_file(file_path);
}
close_file(fhnd);
}
}

inline const char *robust_lock_subdir_path()
{  return "robust"; }

inline const char *robust_lock_prefix()
{  return "lck"; }

inline void robust_lock_path(std::string &s)
{
get_shared_dir(s);
s += "/";
s += robust_lock_subdir_path();
}

inline void create_and_get_robust_lock_file_path(std::string &s, OS_process_id_t pid)
{
intermodule_singleton_helpers::create_tmp_subdir_and_get_pid_based_filepath
(robust_lock_subdir_path(), robust_lock_prefix(), pid, s);
}

class robust_mutex_lock_file
{
file_handle_t fd;
std::string fname;
public:
robust_mutex_lock_file()
{
permissions p;
p.set_unrestricted();
remove_old_robust_lock_files();
create_and_get_robust_lock_file_path(fname, get_current_process_id());

fd = create_or_open_file(fname.c_str(), read_write, p);
if(fd == invalid_file()){
throw interprocess_exception(other_error, "Robust emulation robust_mutex_lock_file constructor failed: could not open or create file");
}

while(1){
bool acquired;
if(!try_acquire_file_lock(fd, acquired) || !acquired ){
throw interprocess_exception(other_error, "Robust emulation robust_mutex_lock_file constructor failed: try_acquire_file_lock");
}
file_handle_t fd2 = create_new_file(fname.c_str(), read_write, p);
if(fd2 != invalid_file()){
close_file(fd);
fd = fd2;
continue;
}
else if(error_info(system_error_code()).get_error_code() == already_exists_error){ 
break;
}
else{
close_file(fd);
throw interprocess_exception(other_error, "Robust emulation robust_mutex_lock_file constructor failed: create_file filed with unexpected error");
}
}
}

~robust_mutex_lock_file()
{
close_file(fd);
delete_file(fname.c_str());
}

private:
class other_process_lock_remover
{
public:
void operator()(const char *filepath, const char *filename)
{
std::string pid_str;
if(!intermodule_singleton_helpers::check_if_filename_complies_with_pid
(filename, robust_lock_prefix(), get_current_process_id(), pid_str)){
remove_if_can_lock_file(filepath);
}
}
};

bool remove_old_robust_lock_files()
{
std::string refcstrRootDirectory;
robust_lock_path(refcstrRootDirectory);
return for_each_file_in_dir(refcstrRootDirectory.c_str(), other_process_lock_remover());
}
};

}  

template<class Mutex>
class robust_spin_mutex
{
public:
static const boost::uint32_t correct_state = 0;
static const boost::uint32_t fixing_state  = 1;
static const boost::uint32_t broken_state  = 2;

typedef robust_emulation_helpers::mutex_traits<Mutex> mutex_traits_t;

robust_spin_mutex();
void lock();
bool try_lock();
bool timed_lock(const boost::posix_time::ptime &abs_time);
void unlock();
void consistent();
bool previous_owner_dead();

private:
static const unsigned int spin_threshold = 100u;
bool lock_own_unique_file();
bool robust_check();
bool check_if_owner_dead_and_take_ownership_atomically();
bool is_owner_dead(boost::uint32_t own);
void owner_to_filename(boost::uint32_t own, std::string &s);
Mutex mtx;
volatile boost::uint32_t owner;
volatile boost::uint32_t state;
};

template<class Mutex>
inline robust_spin_mutex<Mutex>::robust_spin_mutex()
: mtx(), owner(get_invalid_process_id()), state(correct_state)
{}

template<class Mutex>
inline void robust_spin_mutex<Mutex>::lock()
{  try_based_lock(*this);  }

template<class Mutex>
inline bool robust_spin_mutex<Mutex>::try_lock()
{
if(atomic_read32(&this->state) == broken_state){
throw interprocess_exception(lock_error, "Broken id");
}

if(!this->lock_own_unique_file()){
throw interprocess_exception(lock_error, "Broken id");
}

if (mtx.try_lock()){
atomic_write32(&this->owner, get_current_process_id());
return true;
}
else{
if(!this->robust_check()){
return false;
}
else{
return true;
}
}
}

template<class Mutex>
inline bool robust_spin_mutex<Mutex>::timed_lock
(const boost::posix_time::ptime &abs_time)
{  return try_based_timed_lock(*this, abs_time);   }

template<class Mutex>
inline void robust_spin_mutex<Mutex>::owner_to_filename(boost::uint32_t own, std::string &s)
{
robust_emulation_helpers::create_and_get_robust_lock_file_path(s, own);
}

template<class Mutex>
inline bool robust_spin_mutex<Mutex>::robust_check()
{
if(!this->check_if_owner_dead_and_take_ownership_atomically()){
return false;
}
atomic_write32(&this->state, fixing_state);
return true;
}

template<class Mutex>
inline bool robust_spin_mutex<Mutex>::check_if_owner_dead_and_take_ownership_atomically()
{
boost::uint32_t cur_owner = get_current_process_id();
boost::uint32_t old_owner = atomic_read32(&this->owner), old_owner2;
do{
if(!this->is_owner_dead(old_owner)){
return false;
}
old_owner2 = old_owner;
old_owner = atomic_cas32(&this->owner, cur_owner, old_owner);
}while(old_owner2 != old_owner);
mutex_traits_t::take_ownership(mtx);
return true;
}

template<class Mutex>
inline bool robust_spin_mutex<Mutex>::is_owner_dead(boost::uint32_t own)
{
if(own == (boost::uint32_t)get_invalid_process_id()){
return true;
}

std::string file;
this->owner_to_filename(own, file);

file_handle_t fhnd = open_existing_file(file.c_str(), read_write);

if(fhnd != invalid_file()){
bool acquired;
if(try_acquire_file_lock(fhnd, acquired) && acquired){
delete_file(file.c_str());
close_file(fhnd);
return true;
}
close_file(fhnd);
}
else{
if(error_info(system_error_code()).get_error_code() == not_found_error){
return true;
}
}
return false;
}

template<class Mutex>
inline void robust_spin_mutex<Mutex>::consistent()
{
if(atomic_read32(&this->state) != fixing_state &&
atomic_read32(&this->owner) != (boost::uint32_t)get_current_process_id()){
throw interprocess_exception(lock_error, "Broken id");
}
atomic_write32(&this->state, correct_state);
}

template<class Mutex>
inline bool robust_spin_mutex<Mutex>::previous_owner_dead()
{
return atomic_read32(&this->state) == fixing_state;
}

template<class Mutex>
inline void robust_spin_mutex<Mutex>::unlock()
{
if(atomic_read32(&this->state) == fixing_state){
atomic_write32(&this->state, broken_state);
}
atomic_write32(&this->owner, get_invalid_process_id());
mtx.unlock();
}

template<class Mutex>
inline bool robust_spin_mutex<Mutex>::lock_own_unique_file()
{
robust_emulation_helpers::robust_mutex_lock_file* dummy =
&ipcdetail::intermodule_singleton
<robust_emulation_helpers::robust_mutex_lock_file>::get();
return dummy != 0;
}

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif
