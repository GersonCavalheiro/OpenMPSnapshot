
#ifndef BOOST_INTERPROCESS_CONDITION_ANY_HPP
#define BOOST_INTERPROCESS_CONDITION_ANY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/sync/detail/condition_any_algorithm.hpp>

#endif   


namespace boost {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace posix_time
{  class ptime;   }

#endif   

namespace interprocess {

class interprocess_condition_any
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
interprocess_condition_any(const interprocess_condition_any &);
interprocess_condition_any &operator=(const interprocess_condition_any &);

class members
{
public:
typedef interprocess_condition   condvar_type;
typedef interprocess_mutex       mutex_type;

condvar_type &get_condvar() {  return m_cond;  }
mutex_type   &get_mutex()   {  return m_mut; }

private:
condvar_type   m_cond;
mutex_type     m_mut;
};

ipcdetail::condition_any_wrapper<members>   m_cond;

#endif   
public:
interprocess_condition_any(){}

~interprocess_condition_any(){}

void notify_one()
{  m_cond.notify_one();  }

void notify_all()
{  m_cond.notify_all();  }

template <typename L>
void wait(L& lock)
{  m_cond.wait(lock);  }

template <typename L, typename Pr>
void wait(L& lock, Pr pred)
{  m_cond.wait(lock, pred);  }

template <typename L>
bool timed_wait(L& lock, const boost::posix_time::ptime &abs_time)
{  return m_cond.timed_wait(lock, abs_time);  }

template <typename L, typename Pr>
bool timed_wait(L& lock, const boost::posix_time::ptime &abs_time, Pr pred)
{  return m_cond.timed_wait(lock, abs_time, pred);  }
};

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif 
