

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_tbb_thread_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_tbb_thread_H
#pragma message("TBB Warning: tbb/tbb_thread.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_tbb_thread_H
#define __TBB_tbb_thread_H

#define __TBB_tbb_thread_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"

#if _WIN32||_WIN64
#include "machine/windows_api.h"
#define __TBB_NATIVE_THREAD_ROUTINE unsigned WINAPI
#define __TBB_NATIVE_THREAD_ROUTINE_PTR(r) unsigned (WINAPI* r)( void* )
namespace tbb { namespace internal {
#if __TBB_WIN8UI_SUPPORT
typedef size_t thread_id_type;
#else  
typedef DWORD thread_id_type;
#endif 
}} 
#else
#define __TBB_NATIVE_THREAD_ROUTINE void*
#define __TBB_NATIVE_THREAD_ROUTINE_PTR(r) void* (*r)( void* )
#include <pthread.h>
namespace tbb { namespace internal {
typedef pthread_t thread_id_type;
}} 
#endif 

#include "atomic.h"
#include "internal/_tbb_hash_compare_impl.h"
#include "tick_count.h"

#include __TBB_STD_SWAP_HEADER
#include <iosfwd>

namespace tbb {

namespace internal {
class tbb_thread_v3;
}

inline void swap( internal::tbb_thread_v3& t1, internal::tbb_thread_v3& t2 ) __TBB_NOEXCEPT(true);

namespace internal {

void* __TBB_EXPORTED_FUNC allocate_closure_v3( size_t size );
void __TBB_EXPORTED_FUNC free_closure_v3( void* );

struct thread_closure_base {
void* operator new( size_t size ) {return allocate_closure_v3(size);}
void operator delete( void* ptr ) {free_closure_v3(ptr);}
};

template<class F> struct thread_closure_0: thread_closure_base {
F function;

static __TBB_NATIVE_THREAD_ROUTINE start_routine( void* c ) {
thread_closure_0 *self = static_cast<thread_closure_0*>(c);
self->function();
delete self;
return 0;
}
thread_closure_0( const F& f ) : function(f) {}
};
template<class F, class X> struct thread_closure_1: thread_closure_base {
F function;
X arg1;
static __TBB_NATIVE_THREAD_ROUTINE start_routine( void* c ) {
thread_closure_1 *self = static_cast<thread_closure_1*>(c);
self->function(self->arg1);
delete self;
return 0;
}
thread_closure_1( const F& f, const X& x ) : function(f), arg1(x) {}
};
template<class F, class X, class Y> struct thread_closure_2: thread_closure_base {
F function;
X arg1;
Y arg2;
static __TBB_NATIVE_THREAD_ROUTINE start_routine( void* c ) {
thread_closure_2 *self = static_cast<thread_closure_2*>(c);
self->function(self->arg1, self->arg2);
delete self;
return 0;
}
thread_closure_2( const F& f, const X& x, const Y& y ) : function(f), arg1(x), arg2(y) {}
};

class tbb_thread_v3 {
#if __TBB_IF_NO_COPY_CTOR_MOVE_SEMANTICS_BROKEN
public:
#endif
tbb_thread_v3(const tbb_thread_v3&); 
public:
#if _WIN32||_WIN64
typedef HANDLE native_handle_type;
#else
typedef pthread_t native_handle_type;
#endif 

class id;
tbb_thread_v3() __TBB_NOEXCEPT(true) : my_handle(0)
#if _WIN32||_WIN64
, my_thread_id(0)
#endif 
{}

template <class F> explicit tbb_thread_v3(F f) {
typedef internal::thread_closure_0<F> closure_type;
internal_start(closure_type::start_routine, new closure_type(f));
}
template <class F, class X> tbb_thread_v3(F f, X x) {
typedef internal::thread_closure_1<F,X> closure_type;
internal_start(closure_type::start_routine, new closure_type(f,x));
}
template <class F, class X, class Y> tbb_thread_v3(F f, X x, Y y) {
typedef internal::thread_closure_2<F,X,Y> closure_type;
internal_start(closure_type::start_routine, new closure_type(f,x,y));
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
tbb_thread_v3(tbb_thread_v3&& x) __TBB_NOEXCEPT(true)
: my_handle(x.my_handle)
#if _WIN32||_WIN64
, my_thread_id(x.my_thread_id)
#endif
{
x.internal_wipe();
}
tbb_thread_v3& operator=(tbb_thread_v3&& x) __TBB_NOEXCEPT(true) {
internal_move(x);
return *this;
}
private:
tbb_thread_v3& operator=(const tbb_thread_v3& x); 
public:
#else  
tbb_thread_v3& operator=(tbb_thread_v3& x) {
internal_move(x);
return *this;
}
#endif 

void swap( tbb_thread_v3& t ) __TBB_NOEXCEPT(true) {tbb::swap( *this, t );}
bool joinable() const __TBB_NOEXCEPT(true) {return my_handle!=0; }
void __TBB_EXPORTED_METHOD join();
void __TBB_EXPORTED_METHOD detach();
~tbb_thread_v3() {if( joinable() ) detach();}
inline id get_id() const __TBB_NOEXCEPT(true);
native_handle_type native_handle() { return my_handle; }


static unsigned __TBB_EXPORTED_FUNC hardware_concurrency() __TBB_NOEXCEPT(true);
private:
native_handle_type my_handle;
#if _WIN32||_WIN64
thread_id_type my_thread_id;
#endif 

void internal_wipe() __TBB_NOEXCEPT(true) {
my_handle = 0;
#if _WIN32||_WIN64
my_thread_id = 0;
#endif
}
void internal_move(tbb_thread_v3& x) __TBB_NOEXCEPT(true) {
if (joinable()) detach();
my_handle = x.my_handle;
#if _WIN32||_WIN64
my_thread_id = x.my_thread_id;
#endif 
x.internal_wipe();
}


void __TBB_EXPORTED_METHOD internal_start( __TBB_NATIVE_THREAD_ROUTINE_PTR(start_routine),
void* closure );
friend void __TBB_EXPORTED_FUNC move_v3( tbb_thread_v3& t1, tbb_thread_v3& t2 );
friend void tbb::swap( tbb_thread_v3& t1, tbb_thread_v3& t2 ) __TBB_NOEXCEPT(true);
};

class tbb_thread_v3::id {
thread_id_type my_id;
id( thread_id_type id_ ) : my_id(id_) {}

friend class tbb_thread_v3;
public:
id() __TBB_NOEXCEPT(true) : my_id(0) {}

friend bool operator==( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
friend bool operator!=( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
friend bool operator<( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
friend bool operator<=( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
friend bool operator>( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
friend bool operator>=( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);

template<class charT, class traits>
friend std::basic_ostream<charT, traits>&
operator<< (std::basic_ostream<charT, traits> &out,
tbb_thread_v3::id id)
{
out << id.my_id;
return out;
}
friend tbb_thread_v3::id __TBB_EXPORTED_FUNC thread_get_id_v3();

friend inline size_t tbb_hasher( const tbb_thread_v3::id& id ) {
__TBB_STATIC_ASSERT(sizeof(id.my_id) <= sizeof(size_t), "Implementation assumes that thread_id_type fits into machine word");
return tbb::tbb_hasher(id.my_id);
}

friend id atomic_compare_and_swap(id& location, const id& value, const id& comparand){
return as_atomic(location.my_id).compare_and_swap(value.my_id, comparand.my_id);
}
}; 

tbb_thread_v3::id tbb_thread_v3::get_id() const __TBB_NOEXCEPT(true) {
#if _WIN32||_WIN64
return id(my_thread_id);
#else
return id(my_handle);
#endif 
}

void __TBB_EXPORTED_FUNC move_v3( tbb_thread_v3& t1, tbb_thread_v3& t2 );
tbb_thread_v3::id __TBB_EXPORTED_FUNC thread_get_id_v3();
void __TBB_EXPORTED_FUNC thread_yield_v3();
void __TBB_EXPORTED_FUNC thread_sleep_v3(const tick_count::interval_t &i);

inline bool operator==(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
{
return x.my_id == y.my_id;
}
inline bool operator!=(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
{
return x.my_id != y.my_id;
}
inline bool operator<(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
{
return x.my_id < y.my_id;
}
inline bool operator<=(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
{
return x.my_id <= y.my_id;
}
inline bool operator>(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
{
return x.my_id > y.my_id;
}
inline bool operator>=(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
{
return x.my_id >= y.my_id;
}

} 

__TBB_DEPRECATED_VERBOSE_MSG("tbb::thread is deprecated, use std::thread") typedef internal::tbb_thread_v3 tbb_thread;

using internal::operator==;
using internal::operator!=;
using internal::operator<;
using internal::operator>;
using internal::operator<=;
using internal::operator>=;

inline void move( tbb_thread& t1, tbb_thread& t2 ) {
internal::move_v3(t1, t2);
}

inline void swap( internal::tbb_thread_v3& t1, internal::tbb_thread_v3& t2 )  __TBB_NOEXCEPT(true) {
std::swap(t1.my_handle, t2.my_handle);
#if _WIN32||_WIN64
std::swap(t1.my_thread_id, t2.my_thread_id);
#endif 
}

namespace this_tbb_thread {
__TBB_DEPRECATED_VERBOSE inline tbb_thread::id get_id() { return internal::thread_get_id_v3(); }
__TBB_DEPRECATED_VERBOSE inline void yield() { internal::thread_yield_v3(); }
__TBB_DEPRECATED_VERBOSE inline void sleep(const tick_count::interval_t &i) {
internal::thread_sleep_v3(i);
}
}  

} 

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_tbb_thread_H_include_area

#endif 
