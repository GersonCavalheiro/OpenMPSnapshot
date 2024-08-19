
#ifndef BOOST_INTERPROCESS_MANAGED_MULTI_SHARED_MEMORY_HPP
#define BOOST_INTERPROCESS_MANAGED_MULTI_SHARED_MEMORY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/detail/managed_memory_impl.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/interprocess/detail/multi_segment_services.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/containers/list.hpp>
#include <boost/interprocess/mapped_region.hpp> 
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/detail/managed_open_or_create_impl.hpp> 
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/streams/vectorstream.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>
#include <string> 
#include <new>    
#include <ostream>

#include <boost/assert.hpp>
#include <boost/interprocess/mem_algo/rbtree_best_fit.hpp>
#include <boost/interprocess/sync/mutex_family.hpp>


namespace boost {

namespace interprocess {


template
<
class CharType,
class MemoryAlgorithm,
template<class IndexConfig> class IndexType
>
class basic_managed_multi_shared_memory
:  public ipcdetail::basic_managed_memory_impl
<CharType, MemoryAlgorithm, IndexType>
{

typedef basic_managed_multi_shared_memory
<CharType, MemoryAlgorithm, IndexType>    self_t;
typedef ipcdetail::basic_managed_memory_impl
<CharType, MemoryAlgorithm, IndexType>             base_t;

typedef typename MemoryAlgorithm::void_pointer        void_pointer;
typedef typename ipcdetail::
managed_open_or_create_impl<shared_memory_object, MemoryAlgorithm::Alignment, true, false>  managed_impl;
typedef typename void_pointer::segment_group_id       segment_group_id;
typedef typename base_t::size_type                   size_type;



class group_services
:  public multi_segment_services
{
public:
typedef std::pair<void *, size_type>                  result_type;
typedef basic_managed_multi_shared_memory             frontend_t;
typedef typename
basic_managed_multi_shared_memory::void_pointer    void_pointer;
typedef typename void_pointer::segment_group_id       segment_group_id;
group_services(frontend_t *const frontend)
:  mp_frontend(frontend), m_group(0), m_min_segment_size(0){}

virtual std::pair<void *, size_type> create_new_segment(size_type alloc_size)
{  (void)alloc_size;

return result_type(static_cast<void *>(0), 0);
}

virtual bool update_segments ()
{  return true;   }

virtual ~group_services(){}

void set_group(segment_group_id group)
{  m_group = group;  }

segment_group_id get_group() const
{  return m_group;  }

void set_min_segment_size(size_type min_segment_size)
{  m_min_segment_size = min_segment_size;  }

size_type get_min_segment_size() const
{  return m_min_segment_size;  }

private:

frontend_t * const   mp_frontend;
segment_group_id     m_group;
size_type            m_min_segment_size;
};

struct create_open_func
{
enum type_t {  DoCreate, DoOpen, DoOpenOrCreate  };
typedef typename
basic_managed_multi_shared_memory::void_pointer   void_pointer;

create_open_func(self_t * const    frontend,
type_t type, size_type segment_number)
: mp_frontend(frontend), m_type(type), m_segment_number(segment_number){}

bool operator()(void *addr, size_type size, bool created) const
{
if(((m_type == DoOpen)   &&  created) ||
((m_type == DoCreate) && !created))
return false;
segment_group_id group = mp_frontend->m_group_services.get_group();
bool mapped       = false;
bool impl_done    = false;

void_pointer::insert_mapping
( group
, static_cast<char*>(addr) - managed_impl::ManagedOpenOrCreateUserOffset
, size + managed_impl::ManagedOpenOrCreateUserOffset);
if(!m_segment_number){
if((impl_done = created ?
mp_frontend->create_impl(addr, size) : mp_frontend->open_impl(addr, size))){
return true;
}
}
else{
return true;
}

if(impl_done){
mp_frontend->close_impl();
}
if(mapped){
bool ret = void_pointer::erase_last_mapping(group);
BOOST_ASSERT(ret);(void)ret;
}
return false;
}

static std::size_t get_min_size()
{
const size_type sz = self_t::segment_manager::get_min_size();
if(sz > std::size_t(-1)){
BOOST_ASSERT(false);
return std::size_t(-1);
}
else{
return static_cast<std::size_t>(sz);
}
}

self_t * const    mp_frontend;
type_t            m_type;
size_type         m_segment_number;
};

struct close_func
{
typedef typename
basic_managed_multi_shared_memory::void_pointer   void_pointer;

close_func(self_t * const frontend)
: mp_frontend(frontend){}

void operator()(const mapped_region &region, bool last) const
{
if(last) mp_frontend->destroy_impl();
else     mp_frontend->close_impl();
}
self_t * const    mp_frontend;
};

friend struct basic_managed_multi_shared_memory::create_open_func;
friend struct basic_managed_multi_shared_memory::close_func;
friend class basic_managed_multi_shared_memory::group_services;

typedef list<managed_impl> shmem_list_t;

basic_managed_multi_shared_memory *get_this_pointer()
{  return this;   }

public:

basic_managed_multi_shared_memory(create_only_t,
const char *name,
size_type size,
const permissions &perm = permissions())
:  m_group_services(get_this_pointer())
{
priv_open_or_create(create_open_func::DoCreate,name, size, perm);
}

basic_managed_multi_shared_memory(open_or_create_t,
const char *name,
size_type size,
const permissions &perm = permissions())
:  m_group_services(get_this_pointer())
{
priv_open_or_create(create_open_func::DoOpenOrCreate, name, size, perm);
}

basic_managed_multi_shared_memory(open_only_t, const char *name)
:  m_group_services(get_this_pointer())
{
priv_open_or_create(create_open_func::DoOpen, name, 0, permissions());
}

~basic_managed_multi_shared_memory()
{  this->priv_close(); }

private:
bool  priv_open_or_create(typename create_open_func::type_t type,
const char *name,
size_type size,
const permissions &perm)
{
if(!m_shmem_list.empty())
return false;
typename void_pointer::segment_group_id group = 0;
BOOST_TRY{
m_root_name = name;
group = void_pointer::new_segment_group(&m_group_services);
size = void_pointer::round_size(size);
m_group_services.set_group(group);
m_group_services.set_min_segment_size(size);

if(group){
if(this->priv_new_segment(type, size, 0, perm)){
return true;
}
}
}
BOOST_CATCH(const std::bad_alloc&){
}
BOOST_CATCH_END
if(group){
void_pointer::delete_group(group);
}
return false;
}

bool  priv_new_segment(typename create_open_func::type_t type,
size_type size,
const void *addr,
const permissions &perm)
{
BOOST_TRY{
size_type segment_id  = m_shmem_list.size();
boost::interprocess::basic_ovectorstream<boost::interprocess::string> formatter;
size_type str_size = m_root_name.length()+10;
if(formatter.vector().size() < str_size){
formatter.reserve(str_size);
}
formatter << m_root_name
<< static_cast<unsigned int>(segment_id) << std::ends;
create_open_func func(this, type, segment_id);
const char *name = formatter.vector().c_str();
managed_impl mshm;

switch(type){
case create_open_func::DoCreate:
{
managed_impl shm(create_only, name, size, read_write, addr, func, perm);
mshm = boost::move(shm);
}
break;

case create_open_func::DoOpen:
{
managed_impl shm(open_only, name,read_write, addr, func);
mshm = boost::move(shm);
}
break;

case create_open_func::DoOpenOrCreate:
{
managed_impl shm(open_or_create, name, size, read_write, addr, func, perm);
mshm = boost::move(shm);
}
break;

default:
return false;
break;
}

m_shmem_list.push_back(boost::move(mshm));
return true;
}
BOOST_CATCH(const std::bad_alloc&){
}
BOOST_CATCH_END
return false;
}

void priv_close()
{
if(!m_shmem_list.empty()){
bool ret;
segment_group_id group = m_group_services.get_group();
ret = void_pointer::delete_group(group);
(void)ret;
BOOST_ASSERT(ret);
m_shmem_list.clear();
}
}

private:
shmem_list_t   m_shmem_list;
group_services m_group_services;
std::string    m_root_name;
};

typedef basic_managed_multi_shared_memory
< char
, rbtree_best_fit<mutex_family, intersegment_ptr<void> >
, iset_index>
managed_multi_shared_memory;

}  

}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

