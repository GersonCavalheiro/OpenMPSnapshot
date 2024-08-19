
#ifndef BOOST_INTERPROCESS_WINDOWS_INTERMODULE_SINGLETON_HPP
#define BOOST_INTERPROCESS_WINDOWS_INTERMODULE_SINGLETON_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/container/string.hpp>

#if !defined(BOOST_INTERPROCESS_WINDOWS)
#error "This header can't be included from non-windows operating systems"
#endif

#include <boost/assert.hpp>
#include <boost/interprocess/detail/intermodule_singleton_common.hpp>
#include <boost/interprocess/sync/windows/winapi_semaphore_wrapper.hpp>
#include <boost/interprocess/sync/windows/winapi_mutex_wrapper.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/cstdint.hpp>
#include <string>
#include <boost/container/map.hpp>

namespace boost{
namespace interprocess{
namespace ipcdetail{

namespace intermodule_singleton_helpers {

class windows_semaphore_based_map
{
typedef boost::container::map<boost::container::string, ref_count_ptr> map_type;

public:
windows_semaphore_based_map()
{
map_type *m = new map_type;
boost::uint32_t initial_count = 0;
boost::uint32_t max_count = 0;

if(sizeof(void*) == sizeof(boost::uint32_t)){
union caster_union
{
void *addr;
boost::uint32_t addr_uint32;
} caster;
caster.addr = m;
BOOST_ASSERT((caster.addr_uint32 & boost::uint32_t(3)) == 0);
max_count = caster.addr_uint32 >> 2;
}
else if(sizeof(void*) == sizeof(boost::uint64_t)){
union caster_union
{
void *addr;
boost::uint64_t addr_uint64;
} caster;
caster.addr = m;
caster.addr = m;
BOOST_ASSERT((caster.addr_uint64 & boost::uint64_t(3)) == 0);
max_count = boost::uint32_t(caster.addr_uint64 >> 32);
initial_count = boost::uint32_t(caster.addr_uint64);
initial_count = initial_count/4;
BOOST_ASSERT((max_count & boost::uint32_t(0xC0000000)) == 0);
max_count |= boost::uint32_t(0x40000000);
}
bool created = false;
const permissions & perm = permissions();
std::string pid_creation_time, name;
get_pid_creation_time_str(pid_creation_time);
name = "bipc_gmap_sem_lock_";
name += pid_creation_time;
bool success = m_mtx_lock.open_or_create(name.c_str(), perm);
name = "bipc_gmap_sem_count_";
name += pid_creation_time;
scoped_lock<winapi_mutex_wrapper> lck(m_mtx_lock);
{
success = success && m_sem_count.open_or_create
( name.c_str(), static_cast<long>(0), winapi_semaphore_wrapper::MaxCount, perm, created);
name = "bipc_gmap_sem_map_";
name += pid_creation_time;
success = success && m_sem_map.open_or_create
(name.c_str(), initial_count, max_count, perm, created);
if(!success){
delete m;
throw int(0);
}
if(!created){
delete m;
}
else{
BOOST_ASSERT(&get_map_unlocked() == m);
}
m_sem_count.post();
}
}

map_type &get_map_unlocked()
{
if(sizeof(void*) == sizeof(boost::uint32_t)){
union caster_union
{
void *addr;
boost::uint32_t addr_uint32;
} caster;
caster.addr = 0;
caster.addr_uint32 = m_sem_map.limit();
caster.addr_uint32 = caster.addr_uint32 << 2;
return *static_cast<map_type*>(caster.addr);
}
else{
union caster_union
{
void *addr;
boost::uint64_t addr_uint64;
} caster;
boost::uint32_t max_count(m_sem_map.limit()), initial_count(m_sem_map.value());
max_count &= boost::uint32_t(0xBFFFFFFF);
caster.addr_uint64 = max_count;
caster.addr_uint64 =  caster.addr_uint64 << 32;
caster.addr_uint64 |= boost::uint64_t(initial_count) << 2;
return *static_cast<map_type*>(caster.addr);
}
}

ref_count_ptr *find(const char *name)
{
scoped_lock<winapi_mutex_wrapper> lck(m_mtx_lock);
map_type &map = this->get_map_unlocked();
map_type::iterator it = map.find(boost::container::string(name));
if(it != map.end()){
return &it->second;
}
else{
return 0;
}
}

ref_count_ptr * insert(const char *name, const ref_count_ptr &ref)
{
scoped_lock<winapi_mutex_wrapper> lck(m_mtx_lock);
map_type &map = this->get_map_unlocked();
map_type::iterator it = map.insert(map_type::value_type(boost::container::string(name), ref)).first;
return &it->second;
}

bool erase(const char *name)
{
scoped_lock<winapi_mutex_wrapper> lck(m_mtx_lock);
map_type &map = this->get_map_unlocked();
return map.erase(boost::container::string(name)) != 0;
}

template<class F>
void atomic_func(F &f)
{
scoped_lock<winapi_mutex_wrapper> lck(m_mtx_lock);
f();
}

~windows_semaphore_based_map()
{
scoped_lock<winapi_mutex_wrapper> lck(m_mtx_lock);
m_sem_count.wait();
if(0 == m_sem_count.value()){
map_type &map = this->get_map_unlocked();
BOOST_ASSERT(map.empty());
delete &map;
}
m_sem_map.close();
m_sem_count.close();
}

private:
winapi_mutex_wrapper     m_mtx_lock;
winapi_semaphore_wrapper m_sem_map;
winapi_semaphore_wrapper m_sem_count;
};

template<>
struct thread_safe_global_map_dependant<windows_semaphore_based_map>
{
static void apply_gmem_erase_logic(const char *, const char *){}

static bool remove_old_gmem()
{ return true; }

struct lock_file_logic
{
lock_file_logic(windows_semaphore_based_map &)
: retry_with_new_map(false)
{}

void operator()(void){}
bool retry() const { return retry_with_new_map; }
private:
const bool retry_with_new_map;
};

static void construct_map(void *addr)
{
::new (addr)windows_semaphore_based_map;
}

struct unlink_map_logic
{
unlink_map_logic(windows_semaphore_based_map &)
{}
void operator()(){}
};

static ref_count_ptr *find(windows_semaphore_based_map &map, const char *name)
{
return map.find(name);
}

static ref_count_ptr * insert(windows_semaphore_based_map &map, const char *name, const ref_count_ptr &ref)
{
return map.insert(name, ref);
}

static bool erase(windows_semaphore_based_map &map, const char *name)
{
return map.erase(name);
}

template<class F>
static void atomic_func(windows_semaphore_based_map &map, F &f)
{
map.atomic_func(f);
}
};

}  

template<typename C, bool LazyInit = true, bool Phoenix = false>
class windows_intermodule_singleton
: public intermodule_singleton_impl
< C
, LazyInit
, Phoenix
, intermodule_singleton_helpers::windows_semaphore_based_map
>
{};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
