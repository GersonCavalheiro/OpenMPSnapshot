

#ifndef BOOST_FLYWEIGHT_DETAIL_RECURSIVE_LW_MUTEX_HPP
#define BOOST_FLYWEIGHT_DETAIL_RECURSIVE_LW_MUTEX_HPP

#if defined(_MSC_VER)
#pragma once
#endif



#include <boost/config.hpp>

#if !defined(BOOST_HAS_PTHREADS)
#include <boost/detail/lightweight_mutex.hpp>
namespace boost{

namespace flyweights{

namespace detail{

typedef boost::detail::lightweight_mutex recursive_lightweight_mutex;

} 

} 

} 
#else


#include <boost/assert.hpp>
#include <boost/noncopyable.hpp>
#include <pthread.h>

namespace boost{

namespace flyweights{

namespace detail{

struct recursive_lightweight_mutex:noncopyable
{
recursive_lightweight_mutex()
{
pthread_mutexattr_t attr;
BOOST_VERIFY(pthread_mutexattr_init(&attr)==0);
BOOST_VERIFY(pthread_mutexattr_settype(&attr,PTHREAD_MUTEX_RECURSIVE)==0);
BOOST_VERIFY(pthread_mutex_init(&m_,&attr)==0);
BOOST_VERIFY(pthread_mutexattr_destroy(&attr)==0);
}

~recursive_lightweight_mutex(){pthread_mutex_destroy(&m_);}

struct scoped_lock;
friend struct scoped_lock;
struct scoped_lock:noncopyable
{
public:
scoped_lock(recursive_lightweight_mutex& m):m_(m.m_)
{
BOOST_VERIFY(pthread_mutex_lock(&m_)==0);
}

~scoped_lock(){BOOST_VERIFY(pthread_mutex_unlock(&m_)==0);}

private:
pthread_mutex_t& m_;
};

private:
pthread_mutex_t m_;
};

} 

} 

} 
#endif

#endif
