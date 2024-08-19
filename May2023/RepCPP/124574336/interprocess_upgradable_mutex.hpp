
#ifndef BOOST_INTERPROCESS_UPGRADABLE_MUTEX_HPP
#define BOOST_INTERPROCESS_UPGRADABLE_MUTEX_HPP

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

class interprocess_upgradable_mutex
{
interprocess_upgradable_mutex(const interprocess_upgradable_mutex &);
interprocess_upgradable_mutex &operator=(const interprocess_upgradable_mutex &);

friend class interprocess_condition;
public:

interprocess_upgradable_mutex();

~interprocess_upgradable_mutex();


void lock();

bool try_lock();

bool timed_lock(const boost::posix_time::ptime &abs_time);

void unlock();


void lock_sharable();

bool try_lock_sharable();

bool timed_lock_sharable(const boost::posix_time::ptime &abs_time);

void unlock_sharable();


void lock_upgradable();

bool try_lock_upgradable();

bool timed_lock_upgradable(const boost::posix_time::ptime &abs_time);

void unlock_upgradable();


void unlock_and_lock_upgradable();

void unlock_and_lock_sharable();

void unlock_upgradable_and_lock_sharable();


void unlock_upgradable_and_lock();

bool try_unlock_upgradable_and_lock();

bool timed_unlock_upgradable_and_lock(const boost::posix_time::ptime &abs_time);

bool try_unlock_sharable_and_lock();

bool try_unlock_sharable_and_lock_upgradable();

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef scoped_lock<interprocess_mutex> scoped_lock_t;

struct control_word_t
{
unsigned exclusive_in         : 1;
unsigned upgradable_in        : 1;
unsigned num_upr_shar         : sizeof(unsigned)*CHAR_BIT-2;
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

struct upgradable_to_exclusive_rollback
{
upgradable_to_exclusive_rollback(control_word_t         &ctrl)
:  mp_ctrl(&ctrl)
{}

void release()
{  mp_ctrl = 0;   }

~upgradable_to_exclusive_rollback()
{
if(mp_ctrl){
mp_ctrl->upgradable_in = 1;
++mp_ctrl->num_upr_shar;
mp_ctrl->exclusive_in = 0;
}
}
control_word_t          *mp_ctrl;
};

template<int Dummy>
struct base_constants_t
{
static const unsigned max_readers
= ~(unsigned(3) << (sizeof(unsigned)*CHAR_BIT-2));
};
typedef base_constants_t<0> constants;
#endif   
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <int Dummy>
const unsigned interprocess_upgradable_mutex::base_constants_t<Dummy>::max_readers;

inline interprocess_upgradable_mutex::interprocess_upgradable_mutex()
{
this->m_ctrl.exclusive_in  = 0;
this->m_ctrl.upgradable_in = 0;
this->m_ctrl.num_upr_shar   = 0;
}

inline interprocess_upgradable_mutex::~interprocess_upgradable_mutex()
{}

inline void interprocess_upgradable_mutex::lock()
{
scoped_lock_t lck(m_mut);

while (this->m_ctrl.exclusive_in || this->m_ctrl.upgradable_in){
this->m_first_gate.wait(lck);
}

this->m_ctrl.exclusive_in = 1;

exclusive_rollback rollback(this->m_ctrl, this->m_first_gate);

while (this->m_ctrl.num_upr_shar){
this->m_second_gate.wait(lck);
}
rollback.release();
}

inline bool interprocess_upgradable_mutex::try_lock()
{
scoped_lock_t lck(m_mut, try_to_lock);

if(!lck.owns()
|| this->m_ctrl.exclusive_in
|| this->m_ctrl.num_upr_shar){
return false;
}
this->m_ctrl.exclusive_in = 1;
return true;
}

inline bool interprocess_upgradable_mutex::timed_lock
(const boost::posix_time::ptime &abs_time)
{
scoped_lock_t lck(m_mut, abs_time);
if(!lck.owns())   return false;

while (this->m_ctrl.exclusive_in || this->m_ctrl.upgradable_in){
if(!this->m_first_gate.timed_wait(lck, abs_time)){
if(this->m_ctrl.exclusive_in || this->m_ctrl.upgradable_in){
return false;
}
break;
}
}

this->m_ctrl.exclusive_in = 1;

exclusive_rollback rollback(this->m_ctrl, this->m_first_gate);

while (this->m_ctrl.num_upr_shar){
if(!this->m_second_gate.timed_wait(lck, abs_time)){
if(this->m_ctrl.num_upr_shar){
return false;
}
break;
}
}
rollback.release();
return true;
}

inline void interprocess_upgradable_mutex::unlock()
{
scoped_lock_t lck(m_mut);
this->m_ctrl.exclusive_in = 0;
this->m_first_gate.notify_all();
}


inline void interprocess_upgradable_mutex::lock_upgradable()
{
scoped_lock_t lck(m_mut);

while(this->m_ctrl.exclusive_in || this->m_ctrl.upgradable_in
|| this->m_ctrl.num_upr_shar == constants::max_readers){
this->m_first_gate.wait(lck);
}

this->m_ctrl.upgradable_in = 1;
++this->m_ctrl.num_upr_shar;
}

inline bool interprocess_upgradable_mutex::try_lock_upgradable()
{
scoped_lock_t lck(m_mut, try_to_lock);

if(!lck.owns()
|| this->m_ctrl.exclusive_in
|| this->m_ctrl.upgradable_in
|| this->m_ctrl.num_upr_shar == constants::max_readers){
return false;
}

this->m_ctrl.upgradable_in = 1;
++this->m_ctrl.num_upr_shar;
return true;
}

inline bool interprocess_upgradable_mutex::timed_lock_upgradable
(const boost::posix_time::ptime &abs_time)
{
scoped_lock_t lck(m_mut, abs_time);
if(!lck.owns())   return false;

while(this->m_ctrl.exclusive_in
|| this->m_ctrl.upgradable_in
|| this->m_ctrl.num_upr_shar == constants::max_readers){
if(!this->m_first_gate.timed_wait(lck, abs_time)){
if((this->m_ctrl.exclusive_in
|| this->m_ctrl.upgradable_in
|| this->m_ctrl.num_upr_shar == constants::max_readers)){
return false;
}
break;
}
}

this->m_ctrl.upgradable_in = 1;
++this->m_ctrl.num_upr_shar;
return true;
}

inline void interprocess_upgradable_mutex::unlock_upgradable()
{
scoped_lock_t lck(m_mut);
this->m_ctrl.upgradable_in = 0;
--this->m_ctrl.num_upr_shar;
this->m_first_gate.notify_all();
}


inline void interprocess_upgradable_mutex::lock_sharable()
{
scoped_lock_t lck(m_mut);

while(this->m_ctrl.exclusive_in
|| this->m_ctrl.num_upr_shar == constants::max_readers){
this->m_first_gate.wait(lck);
}

++this->m_ctrl.num_upr_shar;
}

inline bool interprocess_upgradable_mutex::try_lock_sharable()
{
scoped_lock_t lck(m_mut, try_to_lock);

if(!lck.owns()
|| this->m_ctrl.exclusive_in
|| this->m_ctrl.num_upr_shar == constants::max_readers){
return false;
}

++this->m_ctrl.num_upr_shar;
return true;
}

inline bool interprocess_upgradable_mutex::timed_lock_sharable
(const boost::posix_time::ptime &abs_time)
{
scoped_lock_t lck(m_mut, abs_time);
if(!lck.owns())   return false;

while (this->m_ctrl.exclusive_in
|| this->m_ctrl.num_upr_shar == constants::max_readers){
if(!this->m_first_gate.timed_wait(lck, abs_time)){
if(this->m_ctrl.exclusive_in
|| this->m_ctrl.num_upr_shar == constants::max_readers){
return false;
}
break;
}
}

++this->m_ctrl.num_upr_shar;
return true;
}

inline void interprocess_upgradable_mutex::unlock_sharable()
{
scoped_lock_t lck(m_mut);
--this->m_ctrl.num_upr_shar;
if (this->m_ctrl.num_upr_shar == 0){
this->m_second_gate.notify_one();
}
else if(this->m_ctrl.num_upr_shar == (constants::max_readers-1)){
this->m_first_gate.notify_all();
}
}


inline void interprocess_upgradable_mutex::unlock_and_lock_upgradable()
{
scoped_lock_t lck(m_mut);
this->m_ctrl.exclusive_in     = 0;
this->m_ctrl.upgradable_in    = 1;
this->m_ctrl.num_upr_shar   = 1;
m_first_gate.notify_all();
}

inline void interprocess_upgradable_mutex::unlock_and_lock_sharable()
{
scoped_lock_t lck(m_mut);
this->m_ctrl.exclusive_in   = 0;
this->m_ctrl.num_upr_shar   = 1;
m_first_gate.notify_all();
}

inline void interprocess_upgradable_mutex::unlock_upgradable_and_lock_sharable()
{
scoped_lock_t lck(m_mut);
this->m_ctrl.upgradable_in    = 0;
m_first_gate.notify_all();
}


inline void interprocess_upgradable_mutex::unlock_upgradable_and_lock()
{
scoped_lock_t lck(m_mut);
this->m_ctrl.upgradable_in = 0;
--this->m_ctrl.num_upr_shar;
this->m_ctrl.exclusive_in = 1;

upgradable_to_exclusive_rollback rollback(m_ctrl);

while (this->m_ctrl.num_upr_shar){
this->m_second_gate.wait(lck);
}
rollback.release();
}

inline bool interprocess_upgradable_mutex::try_unlock_upgradable_and_lock()
{
scoped_lock_t lck(m_mut, try_to_lock);
if(!lck.owns()
|| this->m_ctrl.num_upr_shar != 1){
return false;
}
this->m_ctrl.upgradable_in = 0;
--this->m_ctrl.num_upr_shar;
this->m_ctrl.exclusive_in = 1;
return true;
}

inline bool interprocess_upgradable_mutex::timed_unlock_upgradable_and_lock
(const boost::posix_time::ptime &abs_time)
{
scoped_lock_t lck(m_mut, abs_time);
if(!lck.owns())   return false;

this->m_ctrl.upgradable_in = 0;
--this->m_ctrl.num_upr_shar;
this->m_ctrl.exclusive_in = 1;

upgradable_to_exclusive_rollback rollback(m_ctrl);

while (this->m_ctrl.num_upr_shar){
if(!this->m_second_gate.timed_wait(lck, abs_time)){
if(this->m_ctrl.num_upr_shar){
return false;
}
break;
}
}
rollback.release();
return true;
}

inline bool interprocess_upgradable_mutex::try_unlock_sharable_and_lock()
{
scoped_lock_t lck(m_mut, try_to_lock);

if(!lck.owns()
|| this->m_ctrl.exclusive_in
|| this->m_ctrl.upgradable_in
|| this->m_ctrl.num_upr_shar != 1){
return false;
}
this->m_ctrl.exclusive_in = 1;
this->m_ctrl.num_upr_shar = 0;
return true;
}

inline bool interprocess_upgradable_mutex::try_unlock_sharable_and_lock_upgradable()
{
scoped_lock_t lck(m_mut, try_to_lock);

if(!lck.owns()
|| this->m_ctrl.exclusive_in
|| this->m_ctrl.upgradable_in){
return false;
}

this->m_ctrl.upgradable_in = 1;
return true;
}

#endif   

}  
}  


#include <boost/interprocess/detail/config_end.hpp>

#endif   
