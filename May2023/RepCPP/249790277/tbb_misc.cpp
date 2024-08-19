


#include "tbb/tbb_stddef.h"
#include "tbb_assert_impl.h" 
#include "tbb/tbb_exception.h"
#include "tbb/tbb_machine.h"
#include "tbb_misc.h"
#include "tbb_version.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#if _WIN32||_WIN64
#include "tbb/machine/windows_api.h"
#endif

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <cstring>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#define __TBB_STD_RETHROW_EXCEPTION_POSSIBLY_BROKEN                             \
(__GLIBCXX__ && __TBB_GLIBCXX_VERSION>=40700 && __TBB_GLIBCXX_VERSION<60000 \
&& TBB_USE_EXCEPTIONS && !TBB_USE_CAPTURED_EXCEPTION)

#if __TBB_STD_RETHROW_EXCEPTION_POSSIBLY_BROKEN
#include <cxxabi.h>
#endif

using namespace std;

namespace tbb {

const char* bad_last_alloc::what() const throw() { return "bad allocation in previous or concurrent attempt"; }
const char* improper_lock::what() const throw() { return "attempted recursive lock on critical section or non-recursive mutex"; }
const char* user_abort::what() const throw() { return "User-initiated abort has terminated this operation"; }
const char* invalid_multiple_scheduling::what() const throw() { return "The same task_handle object cannot be executed more than once"; }
const char* missing_wait::what() const throw() { return "wait() was not called on the structured_task_group"; }

namespace internal {

#if TBB_USE_EXCEPTIONS
#define DO_THROW(exc, init_args) throw exc init_args;
#else 
#define PRINT_ERROR_AND_ABORT(exc_name, msg) \
fprintf (stderr, "Exception %s with message %s would've been thrown, "  \
"if exception handling were not disabled. Aborting.\n", exc_name, msg); \
fflush(stderr); \
abort();
#define DO_THROW(exc, init_args) PRINT_ERROR_AND_ABORT(#exc, #init_args)
#endif 



void handle_perror( int error_code, const char* what ) {
char buf[256];
#if _MSC_VER
#define snprintf _snprintf
#endif
int written = snprintf(buf, sizeof(buf), "%s: %s", what, strerror( error_code ));
__TBB_ASSERT_EX( written>0 && written<(int)sizeof(buf), "Error description is too long" );
buf[sizeof(buf)-1] = 0;
#if TBB_USE_EXCEPTIONS
throw runtime_error(buf);
#else
PRINT_ERROR_AND_ABORT( "runtime_error", buf);
#endif 
}

#if _WIN32||_WIN64
void handle_win_error( int error_code ) {
char buf[512];
#if !__TBB_WIN8UI_SUPPORT
FormatMessageA( FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
NULL, error_code, 0, buf, sizeof(buf), NULL );
#else
sprintf_s((char*)&buf, 512, "error code %d", error_code);
#endif
#if TBB_USE_EXCEPTIONS
throw runtime_error(buf);
#else
PRINT_ERROR_AND_ABORT( "runtime_error", buf);
#endif 
}
#endif 

void throw_bad_last_alloc_exception_v4() {
throw_exception_v4(eid_bad_last_alloc);
}

void throw_exception_v4 ( exception_id eid ) {
__TBB_ASSERT ( eid > 0 && eid < eid_max, "Unknown exception ID" );
switch ( eid ) {
case eid_bad_alloc: DO_THROW( bad_alloc, () );
case eid_bad_last_alloc: DO_THROW( bad_last_alloc, () );
case eid_nonpositive_step: DO_THROW( invalid_argument, ("Step must be positive") );
case eid_out_of_range: DO_THROW( out_of_range, ("Index out of requested size range") );
case eid_segment_range_error: DO_THROW( range_error, ("Index out of allocated segment slots") );
case eid_index_range_error: DO_THROW( range_error, ("Index is not allocated") );
case eid_missing_wait: DO_THROW( missing_wait, () );
case eid_invalid_multiple_scheduling: DO_THROW( invalid_multiple_scheduling, () );
case eid_improper_lock: DO_THROW( improper_lock, () );
case eid_possible_deadlock: DO_THROW( runtime_error, ("Resource deadlock would occur") );
case eid_operation_not_permitted: DO_THROW( runtime_error, ("Operation not permitted") );
case eid_condvar_wait_failed: DO_THROW( runtime_error, ("Wait on condition variable failed") );
case eid_invalid_load_factor: DO_THROW( out_of_range, ("Invalid hash load factor") );
case eid_reserved: DO_THROW( out_of_range, ("[backward compatibility] Invalid number of buckets") );
case eid_invalid_swap: DO_THROW( invalid_argument, ("swap() is invalid on non-equal allocators") );
case eid_reservation_length_error: DO_THROW( length_error, ("reservation size exceeds permitted max size") );
case eid_invalid_key: DO_THROW( out_of_range, ("invalid key") );
case eid_user_abort: DO_THROW( user_abort, () );
case eid_bad_tagged_msg_cast: DO_THROW( runtime_error, ("Illegal tagged_msg cast") );
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
case eid_blocking_thread_join_impossible: DO_THROW( runtime_error, ("Blocking terminate failed") );
#endif
default: break;
}
#if !TBB_USE_EXCEPTIONS && __APPLE__
out_of_range e1("");
length_error e2("");
range_error e3("");
invalid_argument e4("");
#endif 
}

#if __TBB_STD_RETHROW_EXCEPTION_POSSIBLY_BROKEN

void fix_broken_rethrow() {
struct gcc_eh_data {
void *       caughtExceptions;
unsigned int uncaughtExceptions;
};
gcc_eh_data* eh_data = punned_cast<gcc_eh_data*>( abi::__cxa_get_globals() );
++eh_data->uncaughtExceptions;
}

bool gcc_rethrow_exception_broken() {
bool is_broken;
__TBB_ASSERT( !std::uncaught_exception(),
"gcc_rethrow_exception_broken() must not be called when an exception is active" );
try {
try {
throw __TBB_GLIBCXX_VERSION;
} catch(...) {
std::rethrow_exception( std::current_exception() );
}
} catch(...) {
is_broken = std::uncaught_exception();
}
if( is_broken ) fix_broken_rethrow();
__TBB_ASSERT( !std::uncaught_exception(), NULL );
return is_broken;
}
#else
void fix_broken_rethrow() {}
bool gcc_rethrow_exception_broken() { return false; }
#endif 

#if __TBB_WIN8UI_SUPPORT
bool GetBoolEnvironmentVariable( const char * ) { return false;}
#else  
bool GetBoolEnvironmentVariable( const char * name ) {
if( const char* s = getenv(name) )
return strcmp(s,"0") != 0;
return false;
}
#endif 


static const char VersionString[] = "\0" TBB_VERSION_STRINGS;

static bool PrintVersionFlag = false;

void PrintVersion() {
PrintVersionFlag = true;
fputs(VersionString+1,stderr);
}

void PrintExtraVersionInfo( const char* category, const char* format, ... ) {
if( PrintVersionFlag ) {
char str[1024]; memset(str, 0, 1024);
va_list args; va_start(args, format);
vsnprintf( str, 1024-1, format, args);
va_end(args);
fprintf(stderr, "TBB: %s\t%s\n", category, str );
}
}

void PrintRMLVersionInfo( void* arg, const char* server_info ) {
PrintExtraVersionInfo( server_info, (const char *)arg );
}

#if _MSC_VER
#include <intrin.h> 
#endif
bool cpu_has_speculation() {
#if __TBB_TSX_AVAILABLE
#if (__INTEL_COMPILER || __GNUC__ || _MSC_VER || __SUNPRO_CC)
bool result = false;
const int rtm_ebx_mask = 1<<11;
#if _MSC_VER
int info[4] = {0,0,0,0};
const int reg_ebx = 1;
__cpuidex(info, 7, 0);
result = (info[reg_ebx] & rtm_ebx_mask)!=0;
#elif __GNUC__ || __SUNPRO_CC
int32_t reg_ebx = 0;
int32_t reg_eax = 7;
int32_t reg_ecx = 0;
__asm__ __volatile__ ( "movl %%ebx, %%esi\n"
"cpuid\n"
"movl %%ebx, %0\n"
"movl %%esi, %%ebx\n"
: "=a"(reg_ebx) : "0" (reg_eax), "c" (reg_ecx) : "esi",
#if __TBB_x86_64
"ebx",
#endif
"edx"
);
result = (reg_ebx & rtm_ebx_mask)!=0 ;
#endif
return result;
#else
#error Speculation detection not enabled for compiler
#endif 
#else  
return false;
#endif 
}

} 

extern "C" int TBB_runtime_interface_version() {
return TBB_INTERFACE_VERSION;
}

} 

#if !__TBB_RML_STATIC
#if __TBB_x86_32

#include "tbb/atomic.h"

#if _MSC_VER
using tbb::internal::int64_t;
#endif

extern "C" void __TBB_machine_store8_slow_perf_warning( volatile void *ptr ) {
const unsigned n = 4;
static tbb::atomic<void*> cache[n];
static tbb::atomic<unsigned> k;
for( unsigned i=0; i<n; ++i )
if( ptr==cache[i] )
goto done;
cache[(k++)%n] = const_cast<void*>(ptr);
tbb::internal::runtime_warning( "atomic store on misaligned 8-byte location %p is slow", ptr );
done:;
}

extern "C" void __TBB_machine_store8_slow( volatile void *ptr, int64_t value ) {
for( tbb::internal::atomic_backoff b;;b.pause() ) {
int64_t tmp = *(int64_t*)ptr;
if( __TBB_machine_cmpswp8(ptr,value,tmp)==tmp )
break;
}
}

#endif 
#endif 

#if __TBB_ipf

extern "C" intptr_t __TBB_machine_lockbyte( volatile unsigned char& flag ) {
tbb::internal::atomic_backoff backoff;
while( !__TBB_TryLockByte(flag) ) backoff.pause();
return 0;
}
#endif
