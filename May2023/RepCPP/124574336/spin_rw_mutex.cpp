

#include "tbb/spin_rw_mutex.h"
#include "tbb/tbb_machine.h"
#include "tbb/atomic.h"
#include "itt_notify.h"

#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (disable: 4244)
#endif

namespace tbb {

template<typename T> 
static inline T CAS(volatile T &addr, T newv, T oldv) {
return tbb::internal::as_atomic(addr).compare_and_swap( newv, oldv );
}

bool spin_rw_mutex_v3::internal_acquire_writer()
{
ITT_NOTIFY(sync_prepare, this);
for( internal::atomic_backoff backoff;;backoff.pause() ){
state_t s = const_cast<volatile state_t&>(state); 
if( !(s & BUSY) ) { 
if( CAS(state, WRITER, s)==s )
break; 
backoff.reset(); 
} else if( !(s & WRITER_PENDING) ) { 
__TBB_AtomicOR(&state, WRITER_PENDING);
}
}
ITT_NOTIFY(sync_acquired, this);
return false;
}

void spin_rw_mutex_v3::internal_release_writer()
{
ITT_NOTIFY(sync_releasing, this);
__TBB_AtomicAND( &state, READERS );
}

void spin_rw_mutex_v3::internal_acquire_reader()
{
ITT_NOTIFY(sync_prepare, this);
for( internal::atomic_backoff b;;b.pause() ){
state_t s = const_cast<volatile state_t&>(state); 
if( !(s & (WRITER|WRITER_PENDING)) ) { 
state_t t = (state_t)__TBB_FetchAndAddW( &state, (intptr_t) ONE_READER );
if( !( t&WRITER ))
break; 
__TBB_FetchAndAddW( &state, -(intptr_t)ONE_READER );
}
}

ITT_NOTIFY(sync_acquired, this);
__TBB_ASSERT( state & READERS, "invalid state of a read lock: no readers" );
}


bool spin_rw_mutex_v3::internal_upgrade()
{
state_t s = state;
__TBB_ASSERT( s & READERS, "invalid state before upgrade: no readers " );
while( (s & READERS)==ONE_READER || !(s & WRITER_PENDING) ) {
state_t old_s = s;
if( (s=CAS(state, s | WRITER | WRITER_PENDING, s))==old_s ) {
ITT_NOTIFY(sync_prepare, this);
internal::atomic_backoff backoff;
while( (state & READERS) != ONE_READER ) backoff.pause();
__TBB_ASSERT((state&(WRITER_PENDING|WRITER))==(WRITER_PENDING|WRITER),"invalid state when upgrading to writer");
__TBB_FetchAndAddW( &state,  - (intptr_t)(ONE_READER+WRITER_PENDING));
ITT_NOTIFY(sync_acquired, this);
return true; 
}
}
internal_release_reader();
return internal_acquire_writer(); 
}

void spin_rw_mutex_v3::internal_downgrade() {
ITT_NOTIFY(sync_releasing, this);
__TBB_FetchAndAddW( &state, (intptr_t)(ONE_READER-WRITER));
__TBB_ASSERT( state & READERS, "invalid state after downgrade: no readers" );
}

void spin_rw_mutex_v3::internal_release_reader()
{
__TBB_ASSERT( state & READERS, "invalid state of a read lock: no readers" );
ITT_NOTIFY(sync_releasing, this); 
__TBB_FetchAndAddWrelease( &state,-(intptr_t)ONE_READER);
}

bool spin_rw_mutex_v3::internal_try_acquire_writer()
{
state_t s = state;
if( !(s & BUSY) ) 
if( CAS(state, WRITER, s)==s ) {
ITT_NOTIFY(sync_acquired, this);
return true; 
}
return false;
}

bool spin_rw_mutex_v3::internal_try_acquire_reader()
{
state_t s = state;
if( !(s & (WRITER|WRITER_PENDING)) ) { 
state_t t = (state_t)__TBB_FetchAndAddW( &state, (intptr_t) ONE_READER );
if( !( t&WRITER )) {  
ITT_NOTIFY(sync_acquired, this);
return true; 
}
__TBB_FetchAndAddW( &state, -(intptr_t)ONE_READER );
}
return false;
}

void spin_rw_mutex_v3::internal_construct() {
ITT_SYNC_CREATE(this, _T("tbb::spin_rw_mutex"), _T(""));
}
} 
