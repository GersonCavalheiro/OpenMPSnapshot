

#ifndef __RML_job_automaton_H
#define __RML_job_automaton_H

#include "rml_base.h"
#include "tbb/atomic.h"

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4244)
#endif

namespace rml {

namespace internal {


class job_automaton: no_copy {
private:
tbb::atomic<intptr_t> my_job;
public:

job_automaton() {
my_job = 0;
}

~job_automaton() {
__TBB_ASSERT( my_job==-1, "must plug before destroying" );
}


bool try_acquire() {
intptr_t snapshot = my_job;
if( snapshot==-1 ) {
return false;
} else {
__TBB_ASSERT( (snapshot&1)==0, "already marked that way" );
intptr_t old = my_job.compare_and_swap( snapshot|1, snapshot );
__TBB_ASSERT( old==snapshot || old==-1, "unexpected interference" );  
return old==snapshot;
}
}

void release() {
intptr_t snapshot = my_job;
__TBB_ASSERT( snapshot&1, NULL );
my_job = snapshot&~1;
}


void set_and_release( rml::job* job ) {
intptr_t value = reinterpret_cast<intptr_t>(job);
__TBB_ASSERT( (value&1)==0, "job misaligned" );
__TBB_ASSERT( value!=0, "null job" );
__TBB_ASSERT( my_job==1, "already set, or not marked busy?" );
my_job = value;
}


bool try_plug_null() {
return my_job.compare_and_swap( -1, 0 )==0;
}


bool try_plug( rml::job*&j ) {
for(;;) {
intptr_t snapshot = my_job;
if( snapshot&1 ) {
j = NULL;
return false;
} 
if( my_job.compare_and_swap( -1, snapshot )==snapshot ) {
j = reinterpret_cast<rml::job*>(snapshot);
return true;
} 
}
}


rml::job* wait_for_job() const {
intptr_t snapshot;
for(;;) {
snapshot = my_job;
if( snapshot&~1 ) break;
__TBB_Yield();
}
__TBB_ASSERT( snapshot!=-1, "wait on plugged job_automaton" );
return reinterpret_cast<rml::job*>(snapshot&~1);
}
};

} 
} 


#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif 

#endif 
