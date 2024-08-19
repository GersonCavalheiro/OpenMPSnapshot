
#ifndef BOOST_INTERPROCESS_INTERMODULE_SINGLETON_COMMON_HPP
#define BOOST_INTERPROCESS_INTERMODULE_SINGLETON_COMMON_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/detail/atomic.hpp>
#include <boost/interprocess/detail/os_thread_functions.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/container/detail/type_traits.hpp>  
#include <boost/interprocess/detail/mpl.hpp>
#include <boost/interprocess/sync/spin/wait.hpp>
#include <boost/assert.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>

namespace boost{
namespace interprocess{
namespace ipcdetail{

namespace intermodule_singleton_helpers {

inline void get_pid_creation_time_str(std::string &s)
{
std::stringstream stream;
stream << get_current_process_id() << '_';
stream.precision(6);
stream << std::fixed << get_current_process_creation_time();
s = stream.str();
}

inline const char *get_map_base_name()
{  return "bip.gmem.map.";  }

inline void get_map_name(std::string &map_name)
{
get_pid_creation_time_str(map_name);
map_name.insert(0, get_map_base_name());
}

inline std::size_t get_map_size()
{  return 65536;  }

template<class ThreadSafeGlobalMap>
struct thread_safe_global_map_dependant;

}  

template<class ThreadSafeGlobalMap>
class intermodule_singleton_common
{
public:
typedef void*(singleton_constructor_t)(ThreadSafeGlobalMap &);
typedef void (singleton_destructor_t)(void *, ThreadSafeGlobalMap &);

static const ::boost::uint32_t Uninitialized       = 0u;
static const ::boost::uint32_t Initializing        = 1u;
static const ::boost::uint32_t Initialized         = 2u;
static const ::boost::uint32_t Broken              = 3u;
static const ::boost::uint32_t Destroyed           = 4u;

static void initialize_singleton_logic
(void *&ptr, volatile boost::uint32_t &this_module_singleton_initialized, singleton_constructor_t constructor, bool phoenix)
{
if(atomic_read32(&this_module_singleton_initialized) != Initialized){
::boost::uint32_t previous_module_singleton_initialized = atomic_cas32
(&this_module_singleton_initialized, Initializing, Uninitialized);
if(previous_module_singleton_initialized == Destroyed){
if(phoenix){
atomic_cas32(&this_module_singleton_initialized, Uninitialized, Destroyed);
previous_module_singleton_initialized = atomic_cas32
(&this_module_singleton_initialized, Initializing, Uninitialized);
}
else{
throw interprocess_exception("Boost.Interprocess: Dead reference on non-Phoenix singleton of type");
}
}
if(previous_module_singleton_initialized == Uninitialized){
try{
initialize_global_map_handle();
ThreadSafeGlobalMap *const pmap = get_map_ptr();
void *tmp = constructor(*pmap);
atomic_inc32(&this_module_singleton_count);
atomic_write32(&this_module_singleton_initialized, Initializing);
ptr = tmp;
atomic_write32(&this_module_singleton_initialized, Initialized);
}
catch(...){
atomic_write32(&this_module_singleton_initialized, Broken);
throw;
}
}
else if(previous_module_singleton_initialized == Initializing){
spin_wait swait;
while(1){
previous_module_singleton_initialized = atomic_read32(&this_module_singleton_initialized);
if(previous_module_singleton_initialized >= Initialized){
break;
}
else if(previous_module_singleton_initialized == Initializing){
swait.yield();
}
else{
BOOST_ASSERT(0);
}
}
}
else if(previous_module_singleton_initialized == Initialized){
}
else{
throw interprocess_exception("boost::interprocess::intermodule_singleton initialization failed");
}
}
BOOST_ASSERT(ptr != 0);
}

static void finalize_singleton_logic(void *&ptr, volatile boost::uint32_t &this_module_singleton_initialized, singleton_destructor_t destructor)
{
if(ptr){
ThreadSafeGlobalMap * const pmap = get_map_ptr();
destructor(ptr, *pmap);
ptr = 0;

atomic_write32(&this_module_singleton_initialized, Destroyed);

if(1 == atomic_dec32(&this_module_singleton_count)){
destroy_global_map_handle();
}
}
}

private:
static ThreadSafeGlobalMap *get_map_ptr()
{
return static_cast<ThreadSafeGlobalMap *>(static_cast<void*>(mem_holder.map_mem));
}

static void initialize_global_map_handle()
{
spin_wait swait;
while(1){
::boost::uint32_t tmp = atomic_cas32(&this_module_map_initialized, Initializing, Uninitialized);
if(tmp == Initialized || tmp == Broken){
break;
}
else if(tmp == Destroyed){
tmp = atomic_cas32(&this_module_map_initialized, Uninitialized, Destroyed);
continue;
}
else if(tmp == Initializing){
swait.yield();
}
else{ 
try{
intermodule_singleton_helpers::thread_safe_global_map_dependant<ThreadSafeGlobalMap>::remove_old_gmem();
ThreadSafeGlobalMap * const pmap = get_map_ptr();
intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::construct_map(static_cast<void*>(pmap));
typename intermodule_singleton_helpers::thread_safe_global_map_dependant<ThreadSafeGlobalMap>::
lock_file_logic f(*pmap);
if(f.retry()){
pmap->~ThreadSafeGlobalMap();
atomic_write32(&this_module_map_initialized, Destroyed);
}
else{
atomic_write32(&this_module_map_initialized, Initialized);
break;
}
}
catch(...){
throw;
}
}
}
}

static void destroy_global_map_handle()
{
if(!atomic_read32(&this_module_singleton_count)){
ThreadSafeGlobalMap * const pmap = get_map_ptr();
typename intermodule_singleton_helpers::thread_safe_global_map_dependant<ThreadSafeGlobalMap>::
unlink_map_logic f(*pmap);
pmap->~ThreadSafeGlobalMap();
atomic_write32(&this_module_map_initialized, Destroyed);
intermodule_singleton_helpers::thread_safe_global_map_dependant<ThreadSafeGlobalMap>::remove_old_gmem();
}
}

static volatile boost::uint32_t this_module_singleton_count;

static volatile boost::uint32_t this_module_map_initialized;

static union mem_holder_t
{
unsigned char map_mem [sizeof(ThreadSafeGlobalMap)];
::boost::container::dtl::max_align_t aligner;
} mem_holder;
};

template<class ThreadSafeGlobalMap>
volatile boost::uint32_t intermodule_singleton_common<ThreadSafeGlobalMap>::this_module_singleton_count;

template<class ThreadSafeGlobalMap>
volatile boost::uint32_t intermodule_singleton_common<ThreadSafeGlobalMap>::this_module_map_initialized;

template<class ThreadSafeGlobalMap>
typename intermodule_singleton_common<ThreadSafeGlobalMap>::mem_holder_t
intermodule_singleton_common<ThreadSafeGlobalMap>::mem_holder;

struct ref_count_ptr
{
ref_count_ptr(void *p, boost::uint32_t count)
: ptr(p), singleton_ref_count(count)
{}
void *ptr;
volatile boost::uint32_t singleton_ref_count;
};


template<typename C, bool LazyInit, bool Phoenix, class ThreadSafeGlobalMap>
class intermodule_singleton_impl
{
public:

static C& get()   
{
if(!this_module_singleton_ptr){
if(lifetime.dummy_function()){  
atentry_work();
}
}
return *static_cast<C*>(this_module_singleton_ptr);
}

private:

static void atentry_work()
{
intermodule_singleton_common<ThreadSafeGlobalMap>::initialize_singleton_logic
(this_module_singleton_ptr, this_module_singleton_initialized, singleton_constructor, Phoenix);
}

static void atexit_work()
{
intermodule_singleton_common<ThreadSafeGlobalMap>::finalize_singleton_logic
(this_module_singleton_ptr, this_module_singleton_initialized, singleton_destructor);
}

static void*                      this_module_singleton_ptr;

static volatile boost::uint32_t   this_module_singleton_initialized;

struct lifetime_type_lazy
{
bool dummy_function()
{  return m_dummy == 0; }

~lifetime_type_lazy()
{
}

static volatile int m_dummy;
};

struct lifetime_type_static
: public lifetime_type_lazy
{
lifetime_type_static()
{  atentry_work();  }
};

typedef typename if_c
<LazyInit, lifetime_type_lazy, lifetime_type_static>::type lifetime_type;

static lifetime_type lifetime;

struct init_atomic_func
{
init_atomic_func(ThreadSafeGlobalMap &m)
: m_map(m), ret_ptr()
{}

void operator()()
{
ref_count_ptr *rcount = intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::find(m_map, typeid(C).name());
if(!rcount){
C *p = new C;
try{
ref_count_ptr val(p, 0u);
rcount = intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::insert(m_map, typeid(C).name(), val);
}
catch(...){
intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::erase(m_map, typeid(C).name());
delete p;
throw;
}
}
std::atexit(&atexit_work);
atomic_inc32(&rcount->singleton_ref_count);
ret_ptr = rcount->ptr;
}
void *data() const
{ return ret_ptr;  }

private:
ThreadSafeGlobalMap &m_map;
void *ret_ptr;
};

struct fini_atomic_func
{
fini_atomic_func(ThreadSafeGlobalMap &m)
: m_map(m)
{}

void operator()()
{
ref_count_ptr *rcount = intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::find(m_map, typeid(C).name());
BOOST_ASSERT(rcount);
BOOST_ASSERT(rcount->singleton_ref_count > 0);
if(atomic_dec32(&rcount->singleton_ref_count) == 1){
BOOST_ASSERT(rcount->ptr != 0);
C *pc = static_cast<C*>(rcount->ptr);
bool destroyed = intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::erase(m_map, typeid(C).name());
(void)destroyed;  BOOST_ASSERT(destroyed == true);
delete pc;
}
}

private:
ThreadSafeGlobalMap &m_map;
};

static void *singleton_constructor(ThreadSafeGlobalMap &map)
{
init_atomic_func f(map);
intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::atomic_func(map, f);
return f.data();
}

static void singleton_destructor(void *p, ThreadSafeGlobalMap &map)
{  (void)p;
fini_atomic_func f(map);
intermodule_singleton_helpers::thread_safe_global_map_dependant
<ThreadSafeGlobalMap>::atomic_func(map, f);
}
};

template <typename C, bool L, bool P, class ThreadSafeGlobalMap>
volatile int intermodule_singleton_impl<C, L, P, ThreadSafeGlobalMap>::lifetime_type_lazy::m_dummy = 0;

template <typename C, bool L, bool P, class ThreadSafeGlobalMap>
void *intermodule_singleton_impl<C, L, P, ThreadSafeGlobalMap>::this_module_singleton_ptr = 0;

template <typename C, bool L, bool P, class ThreadSafeGlobalMap>
volatile boost::uint32_t intermodule_singleton_impl<C, L, P, ThreadSafeGlobalMap>::this_module_singleton_initialized = 0;

template <typename C, bool L, bool P, class ThreadSafeGlobalMap>
typename intermodule_singleton_impl<C, L, P, ThreadSafeGlobalMap>::lifetime_type
intermodule_singleton_impl<C, L, P, ThreadSafeGlobalMap>::lifetime;

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
