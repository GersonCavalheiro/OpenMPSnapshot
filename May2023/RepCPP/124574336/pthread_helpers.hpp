
#ifndef BOOST_INTERPROCESS_PTHREAD_HELPERS_HPP
#define BOOST_INTERPROCESS_PTHREAD_HELPERS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <pthread.h>
#include <errno.h>
#include <boost/interprocess/exceptions.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail{

#if defined BOOST_INTERPROCESS_POSIX_PROCESS_SHARED

struct mutexattr_wrapper
{
mutexattr_wrapper(bool recursive = false)
{
if(pthread_mutexattr_init(&m_attr)!=0 ||
pthread_mutexattr_setpshared(&m_attr, PTHREAD_PROCESS_SHARED)!= 0 ||
(recursive &&
pthread_mutexattr_settype(&m_attr, PTHREAD_MUTEX_RECURSIVE)!= 0 ))
throw interprocess_exception("pthread_mutexattr_xxxx failed");
}

~mutexattr_wrapper()  {  pthread_mutexattr_destroy(&m_attr);  }

operator pthread_mutexattr_t&()  {  return m_attr;  }

pthread_mutexattr_t m_attr;
};

struct condattr_wrapper
{
condattr_wrapper()
{
if(pthread_condattr_init(&m_attr)!=0 ||
pthread_condattr_setpshared(&m_attr, PTHREAD_PROCESS_SHARED)!= 0)
throw interprocess_exception("pthread_condattr_xxxx failed");
}

~condattr_wrapper() { pthread_condattr_destroy(&m_attr); }

operator pthread_condattr_t&(){  return m_attr;  }

pthread_condattr_t m_attr;
};

class mutex_initializer
{
public:
mutex_initializer(pthread_mutex_t &mut, pthread_mutexattr_t &mut_attr)
: mp_mut(&mut)
{
if(pthread_mutex_init(mp_mut, &mut_attr) != 0)
throw interprocess_exception("pthread_mutex_init failed");
}

~mutex_initializer() {  if(mp_mut) pthread_mutex_destroy(mp_mut);  }

void release() {mp_mut = 0; }

private:
pthread_mutex_t *mp_mut;
};

class condition_initializer
{
public:
condition_initializer(pthread_cond_t &cond, pthread_condattr_t &cond_attr)
: mp_cond(&cond)
{
if(pthread_cond_init(mp_cond, &cond_attr)!= 0)
throw interprocess_exception("pthread_cond_init failed");
}

~condition_initializer()   {  if(mp_cond) pthread_cond_destroy(mp_cond);  }

void release()       { mp_cond = 0; }

private:
pthread_cond_t *mp_cond;
};

#endif   

#if defined(BOOST_INTERPROCESS_POSIX_BARRIERS) && defined(BOOST_INTERPROCESS_POSIX_PROCESS_SHARED)

struct barrierattr_wrapper
{
barrierattr_wrapper()
{
if(pthread_barrierattr_init(&m_attr)!=0 ||
pthread_barrierattr_setpshared(&m_attr, PTHREAD_PROCESS_SHARED)!= 0)
throw interprocess_exception("pthread_barrierattr_xxx failed");
}

~barrierattr_wrapper()  {  pthread_barrierattr_destroy(&m_attr);  }

operator pthread_barrierattr_t&()  {  return m_attr;  }

pthread_barrierattr_t m_attr;
};

class barrier_initializer
{
public:
barrier_initializer(pthread_barrier_t &mut,
pthread_barrierattr_t &mut_attr,
int count)
: mp_barrier(&mut)
{
if(pthread_barrier_init(mp_barrier, &mut_attr, count) != 0)
throw interprocess_exception("pthread_barrier_init failed");
}

~barrier_initializer() {  if(mp_barrier) pthread_barrier_destroy(mp_barrier);  }

void release() {mp_barrier = 0; }

private:
pthread_barrier_t *mp_barrier;
};

#endif   

}

}

}

#include <boost/interprocess/detail/config_end.hpp>

#endif 
