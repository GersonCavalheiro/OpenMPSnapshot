
#ifndef BOOST_INTERPROCESS_POSIX_SEMAPHORE_WRAPPER_HPP
#define BOOST_INTERPROCESS_POSIX_SEMAPHORE_WRAPPER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/detail/shared_dir_helpers.hpp>
#include <boost/interprocess/permissions.hpp>

#include <fcntl.h>      
#include <unistd.h>     
#include <string>       
#include <semaphore.h>  
#include <sys/stat.h>   
#include <boost/assert.hpp>

#ifdef SEM_FAILED
#define BOOST_INTERPROCESS_POSIX_SEM_FAILED (reinterpret_cast<sem_t*>(SEM_FAILED))
#else
#define BOOST_INTERPROCESS_POSIX_SEM_FAILED (reinterpret_cast<sem_t*>(-1))
#endif

#ifdef BOOST_INTERPROCESS_POSIX_TIMEOUTS
#include <boost/interprocess/sync/posix/ptime_to_timespec.hpp>
#else
#include <boost/interprocess/detail/os_thread_functions.hpp>
#include <boost/interprocess/sync/detail/locks.hpp>
#include <boost/interprocess/sync/detail/common_algorithms.hpp>
#endif

namespace boost {
namespace interprocess {
namespace ipcdetail {

#ifdef BOOST_INTERPROCESS_POSIX_NAMED_SEMAPHORES

inline bool semaphore_open
(sem_t *&handle, create_enum_t type, const char *origname,
unsigned int count = 0, const permissions &perm = permissions())
{
std::string name;
#ifndef BOOST_INTERPROCESS_FILESYSTEM_BASED_POSIX_SEMAPHORES
add_leading_slash(origname, name);
#else
create_shared_dir_cleaning_old_and_get_filepath(origname, name);
#endif

int oflag = 0;
switch(type){
case DoOpen:
{
handle = ::sem_open(name.c_str(), oflag);
}
break;
case DoOpenOrCreate:
case DoCreate:
{
while(1){
oflag = (O_CREAT | O_EXCL);
handle = ::sem_open(name.c_str(), oflag, perm.get_permissions(), count);
if(handle != BOOST_INTERPROCESS_POSIX_SEM_FAILED){
break;
}
else if(errno == EEXIST && type == DoOpenOrCreate){
oflag = 0;
if( (handle = ::sem_open(name.c_str(), oflag)) != BOOST_INTERPROCESS_POSIX_SEM_FAILED
|| (errno != ENOENT) ){
break;
}
}
else{
break;
}
}
}
break;
default:
{
error_info err(other_error);
throw interprocess_exception(err);
}
}

if(handle == BOOST_INTERPROCESS_POSIX_SEM_FAILED){
throw interprocess_exception(error_info(errno));
}

return true;
}

inline void semaphore_close(sem_t *handle)
{
int ret = sem_close(handle);
if(ret != 0){
BOOST_ASSERT(0);
}
}

inline bool semaphore_unlink(const char *semname)
{
try{
std::string sem_str;
#ifndef BOOST_INTERPROCESS_FILESYSTEM_BASED_POSIX_SEMAPHORES
add_leading_slash(semname, sem_str);
#else
shared_filepath(semname, sem_str);
#endif
return 0 == sem_unlink(sem_str.c_str());
}
catch(...){
return false;
}
}

#endif   

#ifdef BOOST_INTERPROCESS_POSIX_UNNAMED_SEMAPHORES

inline void semaphore_init(sem_t *handle, unsigned int initialCount)
{
int ret = sem_init(handle, 1, initialCount);
if(ret == -1){
error_info err = system_error_code();
throw interprocess_exception(err);
}
}

inline void semaphore_destroy(sem_t *handle)
{
int ret = sem_destroy(handle);
if(ret != 0){
BOOST_ASSERT(0);
}
}

#endif   

inline void semaphore_post(sem_t *handle)
{
int ret = sem_post(handle);
if(ret != 0){
error_info err = system_error_code();
throw interprocess_exception(err);
}
}

inline void semaphore_wait(sem_t *handle)
{
int ret = sem_wait(handle);
if(ret != 0){
error_info err = system_error_code();
throw interprocess_exception(err);
}
}

inline bool semaphore_try_wait(sem_t *handle)
{
int res = sem_trywait(handle);
if(res == 0)
return true;
if(system_error_code() == EAGAIN){
return false;
}
error_info err = system_error_code();
throw interprocess_exception(err);
}

#ifndef BOOST_INTERPROCESS_POSIX_TIMEOUTS

struct semaphore_wrapper_try_wrapper
{
explicit semaphore_wrapper_try_wrapper(sem_t *handle)
: m_handle(handle)
{}

void wait()
{  semaphore_wait(m_handle);  }

bool try_wait()
{  return semaphore_try_wait(m_handle);  }

private:
sem_t *m_handle;
};

#endif

inline bool semaphore_timed_wait(sem_t *handle, const boost::posix_time::ptime &abs_time)
{
#ifdef BOOST_INTERPROCESS_POSIX_TIMEOUTS
if(abs_time.is_pos_infinity()){
semaphore_wait(handle);
return true;
}

timespec tspec = ptime_to_timespec(abs_time);
for (;;){
int res = sem_timedwait(handle, &tspec);
if(res == 0)
return true;
if (res > 0){
errno = res;
}
if(system_error_code() == ETIMEDOUT){
return false;
}
error_info err = system_error_code();
throw interprocess_exception(err);
}
return false;
#else 

semaphore_wrapper_try_wrapper swtw(handle);
ipcdetail::lock_to_wait<semaphore_wrapper_try_wrapper> lw(swtw);
return ipcdetail::try_based_timed_lock(lw, abs_time);

#endif   
}

}  
}  
}  

#endif   
