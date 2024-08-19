
#ifndef BOOST_INTERPROCESS_DETAIL_CONDITION_ALGORITHM_8A_HPP
#define BOOST_INTERPROCESS_DETAIL_CONDITION_ALGORITHM_8A_HPP

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



template<class ConditionMembers>
class condition_algorithm_8a
{
private:
condition_algorithm_8a();
~condition_algorithm_8a();
condition_algorithm_8a(const condition_algorithm_8a &);
condition_algorithm_8a &operator=(const condition_algorithm_8a &);

typedef typename ConditionMembers::semaphore_type  semaphore_type;
typedef typename ConditionMembers::mutex_type      mutex_type;
typedef typename ConditionMembers::integer_type    integer_type;

public:
template<class Lock>
static bool wait  ( ConditionMembers &data, Lock &lock
, bool timeout_enabled, const boost::posix_time::ptime &abs_time);
static void signal(ConditionMembers &data, bool broadcast);
};

template<class ConditionMembers>
inline void condition_algorithm_8a<ConditionMembers>::signal(ConditionMembers &data, bool broadcast)
{
integer_type nsignals_to_issue;

{
scoped_lock<mutex_type> locker(data.get_mtx_unblock_lock());

if ( 0 != data.get_nwaiters_to_unblock() ) {        
if ( 0 == data.get_nwaiters_blocked() ) {        
return;
}
if (broadcast) {
data.get_nwaiters_to_unblock() += nsignals_to_issue = data.get_nwaiters_blocked();
data.get_nwaiters_blocked() = 0;
}
else {
nsignals_to_issue = 1;
data.get_nwaiters_to_unblock()++;
data.get_nwaiters_blocked()--;
}
}
else if ( data.get_nwaiters_blocked() > data.get_nwaiters_gone() ) { 
data.get_sem_block_lock().wait();                      
if ( 0 != data.get_nwaiters_gone() ) {
data.get_nwaiters_blocked() -= data.get_nwaiters_gone();
data.get_nwaiters_gone() = 0;
}
if (broadcast) {
nsignals_to_issue = data.get_nwaiters_to_unblock() = data.get_nwaiters_blocked();
data.get_nwaiters_blocked() = 0;
}
else {
nsignals_to_issue = data.get_nwaiters_to_unblock() = 1;
data.get_nwaiters_blocked()--;
}
}
else { 
return;
}
}
data.get_sem_block_queue().post(nsignals_to_issue);
}

template<class ConditionMembers>
template<class Lock>
inline bool condition_algorithm_8a<ConditionMembers>::wait
( ConditionMembers &data
, Lock &lock
, bool tout_enabled
, const boost::posix_time::ptime &abs_time
)
{
integer_type nsignals_was_left = 0;
integer_type nwaiters_was_gone = 0;

data.get_sem_block_lock().wait();
++data.get_nwaiters_blocked();
data.get_sem_block_lock().post();

lock_inverter<Lock> inverted_lock(lock);
scoped_lock<lock_inverter<Lock> >   external_unlock(inverted_lock);

bool bTimedOut = tout_enabled
? !data.get_sem_block_queue().timed_wait(abs_time)
: (data.get_sem_block_queue().wait(), false);

{
scoped_lock<mutex_type> locker(data.get_mtx_unblock_lock());
if ( 0 != (nsignals_was_left = data.get_nwaiters_to_unblock()) ) {
if ( bTimedOut ) {                       
if ( 0 != data.get_nwaiters_blocked() ) {
data.get_nwaiters_blocked()--;
}
else {
data.get_nwaiters_gone()++;                     
}
}
if ( 0 == --data.get_nwaiters_to_unblock() ) {
if ( 0 != data.get_nwaiters_blocked() ) {
data.get_sem_block_lock().post();          
nsignals_was_left = 0;          
}
else if ( 0 != (nwaiters_was_gone = data.get_nwaiters_gone()) ) {
data.get_nwaiters_gone() = 0;
}
}
}
else if ( (std::numeric_limits<integer_type>::max)()/2
== ++data.get_nwaiters_gone() ) { 
data.get_sem_block_lock().wait();
data.get_nwaiters_blocked() -= data.get_nwaiters_gone();       
data.get_sem_block_lock().post();
data.get_nwaiters_gone() = 0;
}
}

if ( 1 == nsignals_was_left ) {
if ( 0 != nwaiters_was_gone ) {
while ( nwaiters_was_gone-- ) {
data.get_sem_block_queue().wait();       
}
}
data.get_sem_block_lock().post(); 
}


return ( bTimedOut ) ? false : true;
}


template<class ConditionMembers>
class condition_8a_wrapper
{
condition_8a_wrapper(const condition_8a_wrapper &);
condition_8a_wrapper &operator=(const condition_8a_wrapper &);

ConditionMembers m_data;
typedef ipcdetail::condition_algorithm_8a<ConditionMembers> algo_type;

public:

condition_8a_wrapper(){}


ConditionMembers & get_members()
{  return m_data; }

const ConditionMembers & get_members() const
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
