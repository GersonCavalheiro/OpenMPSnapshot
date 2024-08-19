
#ifndef BOOST_INTERPROCESS_SHARABLE_MUTEX_HPP
#define BOOST_INTERPROCESS_SHARABLE_MUTEX_HPP

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
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <climits>



namespace boost {
namespace interprocess {

class interprocess_sharable_mutex
{
interprocess_sharable_mutex(const interprocess_sharable_mutex &);
interprocess_sharable_mutex &operator=(const interprocess_sharable_mutex &);

friend class interprocess_condition;
public:

interprocess_sharable_mutex();

~interprocess_sharable_mutex();


void lock();

bool try_lock();

bool timed_lock(const boost::posix_time::ptime &abs_time);

void unlock();


void lock_sharable();

bool try_lock_sharable();

bool timed_lock_sharable(const boost::posix_time::ptime &abs_time);

void unlock_sharable();

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef scoped_lock<interprocess_mutex> scoped_lock_t;

struct control_word_t
{
unsigned exclusive_in   : 1;
unsigned num_shared     : sizeof(unsigned)*CHAR_BIT-1;
}                       m_ctrl;

interprocess_mutex      m_mut;
interprocess_condition  m_first_gate;
interprocess_condition  m_second_gate;

private:
struct exclusive_rollback
{
exclusive_rollback(control_word_t         &ctrl
,interprocess_condition &first_gate)
:  mp_ctrl(&ctrl), m_first_gate(first_gate)
{}

void release()
{  mp_ctrl = 0;   }

~exclusive_rollback()
{
if(mp_ctrl){
mp_ctrl->exclusive_in = 0;
m_first_gate.notify_all();
}
}
control_word_t          *mp_ctrl;
interprocess_condition  &m_first_gate;
};

template<int Dummy>
struct base_constants_t
{
static const unsigned max_readers
= ~(unsigned(1) << (sizeof(unsigned)*CHAR_BIT-1));
};
typedef base_constants_t<0> constants;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <int Dummy>
const unsigned interprocess_sharable_mutex::base_constants_t<Dummy>::max_readers;

inline interprocess_sharable_mutex::interprocess_sharable_mutex()
{
this->m_ctrl.exclusive_in  = 0;
this->m_ctrl.num_shared   = 0;
}

inline interprocess_sharable_mutex::~interprocess_sharable_mutex()
{}

inline void interprocess_sharable_mutex::lock()
{
scoped_lock_t lck(m_mut);

while (this->m_ctrl.exclusive_in){
this->m_first_gate.wait(lck);
}

this->m_ctrl.exclusive_in = 1;

exclusive_rollback rollback(this->m_ctrl, this->m_first_gate);

while (this->m_ctrl.num_shared){
this->m_second_gate.wait(lck);
}
rollback.release();
}

inline bool interprocess_sharable_mutex::try_lock()
{
scoped_lock_t lck(m_mut, try_to_lock);

if(!lck.owns()
|| this->m_ctrl.exclusive_in
|| this->m_ctrl.num_shared){
return false;
}
this->m_ctrl.exclusive_in = 1;
return true;
}

inline bool interprocess_sharable_mutex::timed_lock
(const boost::posix_time::ptime &abs_time)
{
scoped_lock_t lck(m_mut, abs_time);
if(!lck.owns())   return false;

while (this->m_ctrl.exclusive_in){
if(!this->m_first_gate.timed_wait(lck, abs_time)){
if(this->m_ctrl.exclusive_in){
return false;
}
break;
}
}

this->m_ctrl.exclusive_in = 1;

exclusive_rollback rollback(this->m_ctrl, this->m_first_gate);

while (this->m_ctrl.num_shared){
if(!this->m_second_gate.timed_wait(lck, abs_time)){
if(this->m_ctrl.num_shared){
return false;
}
break;
}
}
rollback.release();
return true;
}

inline void interprocess_sharable_mutex::unlock()
{
scoped_lock_t lck(m_mut);
this->m_ctrl.exclusive_in = 0;
this->m_first_gate.notify_all();
}


inline void interprocess_sharable_mutex::lock_sharable()
{
scoped_lock_t lck(m_mut);

while(this->m_ctrl.exclusive_in
|| this->m_ctrl.num_shared == constants::max_readers){
this->m_first_gate.wait(lck);
}

++this->m_ctrl.num_shared;
}

inline bool interprocess_sharable_mutex::try_lock_sharable()
{
scoped_lock_t lck(m_mut, try_to_lock);

if(!lck.owns()
|| this->m_ctrl.exclusive_in
|| this->m_ctrl.num_shared == constants::max_readers){
return false;
}

++this->m_ctrl.num_shared;
return true;
}

inline bool interprocess_sharable_mutex::timed_lock_sharable
(const boost::posix_time::ptime &abs_time)
{
scoped_lock_t lck(m_mut, abs_time);
if(!lck.owns())   return false;

while (this->m_ctrl.exclusive_in
|| this->m_ctrl.num_shared == constants::max_readers){
if(!this->m_first_gate.timed_wait(lck, abs_time)){
if(this->m_ctrl.exclusive_in
|| this->m_ctrl.num_shared == constants::max_readers){
return false;
}
break;
}
}

++this->m_ctrl.num_shared;
return true;
}

inline void interprocess_sharable_mutex::unlock_sharable()
{
scoped_lock_t lck(m_mut);
--this->m_ctrl.num_shared;
if (this->m_ctrl.num_shared == 0){
this->m_second_gate.notify_one();
}
else if(this->m_ctrl.num_shared == (constants::max_readers-1)){
this->m_first_gate.notify_all();
}
}

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
