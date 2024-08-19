

#include "tbb/reader_writer_lock.h"
#include "tbb/tbb_machine.h"
#include "tbb/tbb_exception.h"
#include "itt_notify.h"

#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (disable: 4244)
#endif

namespace tbb {
namespace interface5 {

const uintptr_t WFLAG1 = 0x1;  
const uintptr_t WFLAG2 = 0x2;  
const uintptr_t RFLAG = 0x4;   
const uintptr_t RC_INCR = 0x8; 


inline uintptr_t fetch_and_or(atomic<uintptr_t>& operand, uintptr_t value) {
for (tbb::internal::atomic_backoff b;;b.pause()) {
uintptr_t old = operand;
uintptr_t result = operand.compare_and_swap(old|value, old);
if (result==old) return result;
}
}

inline uintptr_t fetch_and_and(atomic<uintptr_t>& operand, uintptr_t value) {
for (tbb::internal::atomic_backoff b;;b.pause()) {
uintptr_t old = operand;
uintptr_t result = operand.compare_and_swap(old&value, old);
if (result==old) return result;
}
}


template<typename T, typename U>
void spin_wait_while_geq( const volatile T& location, U value ) {
tbb::internal::atomic_backoff backoff;
while( location>=value ) backoff.pause();
}


template<typename T, typename U>
void spin_wait_until_and( const volatile T& location, U value ) {
tbb::internal::atomic_backoff backoff;
while( !(location & value) ) backoff.pause();
}


void reader_writer_lock::internal_construct() {
reader_head = NULL;
writer_head = NULL;
writer_tail = NULL;
rdr_count_and_flags = 0;
my_current_writer = tbb_thread::id();
#if TBB_USE_THREADING_TOOLS
ITT_SYNC_CREATE(this, _T("tbb::reader_writer_lock"), _T(""));
#endif 
}

void reader_writer_lock::internal_destroy() {
__TBB_ASSERT(rdr_count_and_flags==0, "reader_writer_lock destroyed with pending readers/writers.");
__TBB_ASSERT(reader_head==NULL, "reader_writer_lock destroyed with pending readers.");
__TBB_ASSERT(writer_tail==NULL, "reader_writer_lock destroyed with pending writers.");
__TBB_ASSERT(writer_head==NULL, "reader_writer_lock destroyed with pending/active writers.");
}

void reader_writer_lock::lock() {
if (is_current_writer()) { 
tbb::internal::throw_exception(tbb::internal::eid_improper_lock);
}
else {
scoped_lock *a_writer_lock = new scoped_lock();
(void) start_write(a_writer_lock);
}
}

bool reader_writer_lock::try_lock() {
if (is_current_writer()) { 
return false;
}
else {
scoped_lock *a_writer_lock = new scoped_lock();
a_writer_lock->status = waiting_nonblocking;
return start_write(a_writer_lock);
}
}

bool reader_writer_lock::start_write(scoped_lock *I) {
tbb_thread::id id = this_tbb_thread::get_id();
scoped_lock *pred = NULL;
if (I->status == waiting_nonblocking) {
if ((pred = writer_tail.compare_and_swap(I, NULL)) != NULL) {
delete I;
return false;
}
}
else {
ITT_NOTIFY(sync_prepare, this);
pred = writer_tail.fetch_and_store(I);
}
if (pred)
pred->next = I;
else {
set_next_writer(I);
if (I->status == waiting_nonblocking) {
if (I->next) { 
set_next_writer(I->next);
}
else { 
writer_head.fetch_and_store(NULL);
if (I != writer_tail.compare_and_swap(NULL, I)) { 
spin_wait_while_eq(I->next, (scoped_lock *)NULL);  
__TBB_ASSERT(I->next, "There should be a node following the last writer.");
set_next_writer(I->next);
}
}
delete I;
return false;
}
}
spin_wait_while_eq(I->status, waiting);
ITT_NOTIFY(sync_acquired, this);
my_current_writer = id;
return true;
}

void reader_writer_lock::set_next_writer(scoped_lock *W) {
writer_head = W;
if (W->status == waiting_nonblocking) {
if (rdr_count_and_flags.compare_and_swap(WFLAG1+WFLAG2, 0) == 0) {
W->status = active;
}
}
else {
if (fetch_and_or(rdr_count_and_flags, WFLAG1) & RFLAG) { 
spin_wait_until_and(rdr_count_and_flags, WFLAG2); 
}
else { 
__TBB_AtomicOR(&rdr_count_and_flags, WFLAG2);
}
spin_wait_while_geq(rdr_count_and_flags, RC_INCR); 
W->status = active;
}
}

void reader_writer_lock::lock_read() {
if (is_current_writer()) { 
tbb::internal::throw_exception(tbb::internal::eid_improper_lock);
}
else {
scoped_lock_read a_reader_lock;
start_read(&a_reader_lock);
}
}

bool reader_writer_lock::try_lock_read() {
if (is_current_writer()) { 
return false;
}
else {
if (rdr_count_and_flags.fetch_and_add(RC_INCR) & (WFLAG1+WFLAG2)) { 
rdr_count_and_flags -= RC_INCR;
return false;
}
else { 
ITT_NOTIFY(sync_acquired, this);
return true;
}
}
}

void reader_writer_lock::start_read(scoped_lock_read *I) {
ITT_NOTIFY(sync_prepare, this);
I->next = reader_head.fetch_and_store(I);
if (!I->next) { 
if (!(fetch_and_or(rdr_count_and_flags, RFLAG) & (WFLAG1+WFLAG2))) { 
unblock_readers();
}
}
__TBB_ASSERT(I->status == waiting || I->status == active, "Lock requests should be waiting or active before blocking.");
spin_wait_while_eq(I->status, waiting); 
if (I->next) {
__TBB_ASSERT(I->next->status == waiting, NULL);
rdr_count_and_flags += RC_INCR;
I->next->status = active; 
}
ITT_NOTIFY(sync_acquired, this);
}

void reader_writer_lock::unblock_readers() {
__TBB_ASSERT(rdr_count_and_flags&RFLAG, NULL);
rdr_count_and_flags += RC_INCR-RFLAG;
__TBB_ASSERT(rdr_count_and_flags >= RC_INCR, NULL);
if (rdr_count_and_flags & WFLAG1 && !(rdr_count_and_flags & WFLAG2)) {
__TBB_AtomicOR(&rdr_count_and_flags, WFLAG2);
}
scoped_lock_read *head = reader_head.fetch_and_store(NULL);
__TBB_ASSERT(head, NULL);
__TBB_ASSERT(head->status == waiting, NULL);
head->status = active;
}

void reader_writer_lock::unlock() {
if( my_current_writer!=tbb_thread::id() ) {
__TBB_ASSERT(is_current_writer(), "caller of reader_writer_lock::unlock() does not own the lock.");
__TBB_ASSERT(writer_head, NULL);
__TBB_ASSERT(writer_head->status==active, NULL);
scoped_lock *a_writer_lock = writer_head;
end_write(a_writer_lock);
__TBB_ASSERT(a_writer_lock != writer_head, "Internal error: About to turn writer_head into dangling reference.");
delete a_writer_lock;
} else {
end_read();
}
}

void reader_writer_lock::end_write(scoped_lock *I) {
__TBB_ASSERT(I==writer_head, "Internal error: can't unlock a thread that is not holding the lock.");
my_current_writer = tbb_thread::id();
ITT_NOTIFY(sync_releasing, this);
if (I->next) { 
writer_head = I->next;
writer_head->status = active;
}
else { 
__TBB_ASSERT(writer_head, NULL);
if (fetch_and_and(rdr_count_and_flags, ~(WFLAG1+WFLAG2)) & RFLAG) {
unblock_readers();
}
writer_head.fetch_and_store(NULL);
if (I != writer_tail.compare_and_swap(NULL, I)) { 
spin_wait_while_eq(I->next, (scoped_lock *)NULL);  
__TBB_ASSERT(I->next, "There should be a node following the last writer.");
set_next_writer(I->next);
}
}
}

void reader_writer_lock::end_read() {
ITT_NOTIFY(sync_releasing, this);
__TBB_ASSERT(rdr_count_and_flags >= RC_INCR, "unlock() called but no readers hold the lock.");
rdr_count_and_flags -= RC_INCR;
}

inline bool reader_writer_lock::is_current_writer() {
return my_current_writer==this_tbb_thread::get_id();
}

void reader_writer_lock::scoped_lock::internal_construct (reader_writer_lock& lock) {
mutex = &lock;
next = NULL;
status = waiting;
if (mutex->is_current_writer()) { 
tbb::internal::throw_exception(tbb::internal::eid_improper_lock);
}
else { 
(void) mutex->start_write(this);
}
}

inline reader_writer_lock::scoped_lock::scoped_lock() : mutex(NULL), next(NULL) {
status = waiting;
}

void reader_writer_lock::scoped_lock_read::internal_construct (reader_writer_lock& lock) {
mutex = &lock;
next = NULL;
status = waiting;
if (mutex->is_current_writer()) { 
tbb::internal::throw_exception(tbb::internal::eid_improper_lock);
}
else { 
mutex->start_read(this);
}
}

inline reader_writer_lock::scoped_lock_read::scoped_lock_read() : mutex(NULL), next(NULL) {
status = waiting;
}

void reader_writer_lock::scoped_lock::internal_destroy() {
if (mutex) {
__TBB_ASSERT(mutex->is_current_writer(), "~scoped_lock() destroyed by thread different than thread that holds lock.");
mutex->end_write(this);
}
status = invalid;
}

void reader_writer_lock::scoped_lock_read::internal_destroy() {
if (mutex)
mutex->end_read();
status = invalid;
}

} 
} 
