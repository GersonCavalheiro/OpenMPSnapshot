
#ifndef BOOST_INTERPROCESS_DETAIL_CONDITION_ANY_ALGORITHM_HPP
#define BOOST_INTERPROCESS_DETAIL_CONDITION_ANY_ALGORITHM_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/detail/locks.hpp>
#include <limits>

namespace boost {
namespace interprocess {
namespace ipcdetail {



template<class ConditionAnyMembers>
class condition_any_algorithm
{
private:
condition_any_algorithm();
~condition_any_algorithm();
condition_any_algorithm(const condition_any_algorithm &);
condition_any_algorithm &operator=(const condition_any_algorithm &);

typedef typename ConditionAnyMembers::mutex_type      mutex_type;
typedef typename ConditionAnyMembers::condvar_type    condvar_type;

template <class Lock>
static void do_wait(ConditionAnyMembers &data, Lock& lock);

template <class Lock>
static bool do_timed_wait(ConditionAnyMembers &data, Lock& lock, const boost::posix_time::ptime &abs_time);

public:
template<class Lock>
static bool wait  ( ConditionAnyMembers &data, Lock &mut
, bool timeout_enabled, const boost::posix_time::ptime &abs_time);
static void signal( ConditionAnyMembers &data, bool broadcast);
};

template<class ConditionAnyMembers>
void condition_any_algorithm<ConditionAnyMembers>::signal(ConditionAnyMembers &data, bool broadcast)
{
scoped_lock<mutex_type> internal_lock(data.get_mutex());
if(broadcast){
data.get_condvar().notify_all();
}
else{
data.get_condvar().notify_one();
}
}

template<class ConditionAnyMembers>
template<class Lock>
bool condition_any_algorithm<ConditionAnyMembers>::wait
( ConditionAnyMembers &data
, Lock &lock
, bool tout_enabled
, const boost::posix_time::ptime &abs_time)
{
if(tout_enabled){
return condition_any_algorithm::do_timed_wait(data, lock, abs_time);
}
else{
condition_any_algorithm::do_wait(data, lock);
return true;
}
}

template<class ConditionAnyMembers>
template <class Lock>
void condition_any_algorithm<ConditionAnyMembers>::do_wait
(ConditionAnyMembers &data, Lock& lock)
{
scoped_lock<mutex_type> internal_lock(data.get_mutex());
{
lock_inverter<Lock> inverted_lock(lock);
scoped_lock<lock_inverter<Lock> >   external_unlock(inverted_lock);
{  
scoped_lock<mutex_type>     internal_unlock;
internal_lock.swap(internal_unlock);
data.get_condvar().wait(internal_unlock);
}
}
}

template<class ConditionAnyMembers>
template <class Lock>
bool condition_any_algorithm<ConditionAnyMembers>::do_timed_wait
(ConditionAnyMembers &data, Lock& lock, const boost::posix_time::ptime &abs_time)
{
scoped_lock<mutex_type> internal_lock(data.get_mutex());
{
lock_inverter<Lock> inverted_lock(lock);
scoped_lock<lock_inverter<Lock> >   external_unlock(inverted_lock);
{  
scoped_lock<mutex_type> internal_unlock;
internal_lock.swap(internal_unlock);
return data.get_condvar().timed_wait(internal_unlock, abs_time);
}
}
}


template<class ConditionAnyMembers>
class condition_any_wrapper
{
condition_any_wrapper(const condition_any_wrapper &);
condition_any_wrapper &operator=(const condition_any_wrapper &);

ConditionAnyMembers m_data;
typedef ipcdetail::condition_any_algorithm<ConditionAnyMembers> algo_type;

public:

condition_any_wrapper(){}

~condition_any_wrapper(){}

ConditionAnyMembers & get_members()
{  return m_data; }

const ConditionAnyMembers & get_members() const
{  return m_data; }

void notify_one()
{  algo_type::signal(m_data, false);  }

void notify_all()
{  algo_type::signal(m_data, true);  }

template <typename L>
void wait(L& lock)
{
if (!lock)
throw lock_exception();
algo_type::wait(m_data, lock, false, boost::posix_time::ptime());
}

template <typename L, typename Pr>
void wait(L& lock, Pr pred)
{
if (!lock)
throw lock_exception();

while (!pred())
algo_type::wait(m_data, lock, false, boost::posix_time::ptime());
}

template <typename L>
bool timed_wait(L& lock, const boost::posix_time::ptime &abs_time)
{
if (!lock)
throw lock_exception();
return algo_type::wait(m_data, lock, true, abs_time);
}

template <typename L, typename Pr>
bool timed_wait(L& lock, const boost::posix_time::ptime &abs_time, Pr pred)
{
if (!lock)
throw lock_exception();
while (!pred()){
if (!algo_type::wait(m_data, lock, true, abs_time))
return pred();
}
return true;
}
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
