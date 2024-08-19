

#include "tbb/tbb_config.h"
#if __TBB_TSX_AVAILABLE
#include "tbb/spin_rw_mutex.h"
#include "tbb/tbb_machine.h"
#include "itt_notify.h"
#include "governor.h"
#include "tbb/atomic.h"

#ifndef __TBB_RW_MUTEX_DELAY_TEST
#define __TBB_RW_MUTEX_DELAY_TEST 0
#endif

#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (disable: 4244)
#endif

namespace tbb {

namespace interface8 {
namespace internal {

enum {
speculation_transaction_aborted = 0x01,
speculation_can_retry           = 0x02,
speculation_memadd_conflict     = 0x04,
speculation_buffer_overflow     = 0x08,
speculation_breakpoint_hit      = 0x10,
speculation_nested_abort        = 0x20,
speculation_xabort_mask         = 0xFF000000,
speculation_xabort_shift        = 24,
speculation_retry               = speculation_transaction_aborted
| speculation_can_retry
| speculation_memadd_conflict
};

static const int retry_threshold_read = 10;
static const int retry_threshold_write = 10;

void x86_rtm_rw_mutex::internal_release(x86_rtm_rw_mutex::scoped_lock& s) {
switch(s.transaction_state) {
case RTM_transacting_writer:
case RTM_transacting_reader:
{
__TBB_ASSERT(__TBB_machine_is_in_transaction(), "transaction_state && not speculating");
#if __TBB_RW_MUTEX_DELAY_TEST
if(s.transaction_state == RTM_transacting_reader) {
if(this->w_flag) __TBB_machine_transaction_conflict_abort();
} else {
if(this->state) __TBB_machine_transaction_conflict_abort();
}
#endif
__TBB_machine_end_transaction();
s.my_scoped_lock.internal_set_mutex(NULL);
}
break;
case RTM_real_reader:
__TBB_ASSERT(!this->w_flag, "w_flag set but read lock acquired");
s.my_scoped_lock.release();
break;
case RTM_real_writer:
__TBB_ASSERT(this->w_flag, "w_flag unset but write lock acquired");
this->w_flag = false;
s.my_scoped_lock.release();
break;
case RTM_not_in_mutex:
__TBB_ASSERT(false, "RTM_not_in_mutex, but in release");
default:
__TBB_ASSERT(false, "invalid transaction_state");
}
s.transaction_state = RTM_not_in_mutex;
}

void x86_rtm_rw_mutex::internal_acquire_writer(x86_rtm_rw_mutex::scoped_lock& s, bool only_speculate)
{
__TBB_ASSERT(s.transaction_state == RTM_not_in_mutex, "scoped_lock already in transaction");
if(tbb::internal::governor::speculation_enabled()) {
int num_retries = 0;
unsigned int abort_code;
do {
tbb::internal::atomic_backoff backoff;
if(this->state) {
if(only_speculate) return;
do {
backoff.pause();  
} while(this->state);
}
if(( abort_code = __TBB_machine_begin_transaction()) == ~(unsigned int)(0) )
{
#if !__TBB_RW_MUTEX_DELAY_TEST
if(this->state) {  
__TBB_machine_transaction_conflict_abort();
}
#endif
s.transaction_state = RTM_transacting_writer;
s.my_scoped_lock.internal_set_mutex(this);  
return;  
}
++num_retries;
} while( (abort_code & speculation_retry) != 0 && (num_retries < retry_threshold_write) );
}

if(only_speculate) return;              
s.my_scoped_lock.acquire(*this, true);  
__TBB_ASSERT(!w_flag, "After acquire for write, w_flag already true");
w_flag = true;                          
s.transaction_state = RTM_real_writer;
return;
}

void x86_rtm_rw_mutex::internal_acquire_reader(x86_rtm_rw_mutex::scoped_lock& s, bool only_speculate) {
__TBB_ASSERT(s.transaction_state == RTM_not_in_mutex, "scoped_lock already in transaction");
if(tbb::internal::governor::speculation_enabled()) {
int num_retries = 0;
unsigned int abort_code;
do {
tbb::internal::atomic_backoff backoff;
if(w_flag) {
if(only_speculate) return;
do {
backoff.pause();  
} while(w_flag);
}
if((abort_code = __TBB_machine_begin_transaction()) == ~(unsigned int)(0) )
{
#if !__TBB_RW_MUTEX_DELAY_TEST
if(w_flag) {  
__TBB_machine_transaction_conflict_abort();  
}
#endif
s.transaction_state = RTM_transacting_reader;
s.my_scoped_lock.internal_set_mutex(this);  
return;  
}
++num_retries;
} while( (abort_code & speculation_retry) != 0 && (num_retries < retry_threshold_read) );
}

if(only_speculate) return;
s.my_scoped_lock.acquire( *this, false );
s.transaction_state = RTM_real_reader;
}


bool x86_rtm_rw_mutex::internal_upgrade(x86_rtm_rw_mutex::scoped_lock& s)
{
switch(s.transaction_state) {
case RTM_real_reader: {
s.transaction_state = RTM_real_writer;
bool no_release = s.my_scoped_lock.upgrade_to_writer();
__TBB_ASSERT(!w_flag, "After upgrade_to_writer, w_flag already true");
w_flag = true;
return no_release;
}
case RTM_transacting_reader:
#if !__TBB_RW_MUTEX_DELAY_TEST
if(this->state) {  
internal_release(s);
internal_acquire_writer(s);
return false;
} else
#endif
{
s.transaction_state = RTM_transacting_writer;
return true;
}
default:
__TBB_ASSERT(false, "Invalid state for upgrade");
return false;
}
}

bool x86_rtm_rw_mutex::internal_downgrade(x86_rtm_rw_mutex::scoped_lock& s) {
switch(s.transaction_state) {
case RTM_real_writer:
s.transaction_state = RTM_real_reader;
__TBB_ASSERT(w_flag, "Before downgrade_to_reader w_flag not true");
w_flag = false;
return s.my_scoped_lock.downgrade_to_reader();
case RTM_transacting_writer:
#if __TBB_RW_MUTEX_DELAY_TEST
if(this->state) {  
__TBB_machine_transaction_conflict_abort();
}
#endif
s.transaction_state = RTM_transacting_reader;
return true;
default:
__TBB_ASSERT(false, "Invalid state for downgrade");
return false;
}
}

bool x86_rtm_rw_mutex::internal_try_acquire_writer(x86_rtm_rw_mutex::scoped_lock& s)
{
internal_acquire_writer(s, true);
if(s.transaction_state == RTM_transacting_writer) {
return true;
}
__TBB_ASSERT(s.transaction_state == RTM_not_in_mutex, "Trying to acquire writer which is already allocated");
bool result = s.my_scoped_lock.try_acquire(*this, true);
if(result) {
__TBB_ASSERT(!w_flag, "After try_acquire_writer, w_flag already true");
w_flag = true;
s.transaction_state = RTM_real_writer;
}
return result;
}

void x86_rtm_rw_mutex::internal_construct() {
ITT_SYNC_CREATE(this, _T("tbb::x86_rtm_rw_mutex"), _T(""));
}

} 
} 
} 

#endif 
