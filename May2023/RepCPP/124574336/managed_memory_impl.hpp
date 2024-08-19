
#ifndef BOOST_INTERPROCESS_DETAIL_MANAGED_MEMORY_IMPL_HPP
#define BOOST_INTERPROCESS_DETAIL_MANAGED_MEMORY_IMPL_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/segment_manager.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/detail/nothrow.hpp>
#include <boost/interprocess/detail/simple_swap.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>
#include <boost/assert.hpp>


namespace boost {
namespace interprocess {
namespace ipcdetail {

template<class BasicManagedMemoryImpl>
class create_open_func;

template<
class CharType,
class MemoryAlgorithm,
template<class IndexConfig> class IndexType
>
struct segment_manager_type
{
typedef segment_manager<CharType, MemoryAlgorithm, IndexType> type;
};

template <  class CharType
,  class MemoryAlgorithm
,  template<class IndexConfig> class IndexType
,  std::size_t Offset = 0
>
class basic_managed_memory_impl
{
basic_managed_memory_impl(const basic_managed_memory_impl &);
basic_managed_memory_impl &operator=(const basic_managed_memory_impl &);

template<class BasicManagedMemoryImpl>
friend class create_open_func;

public:
typedef typename segment_manager_type
<CharType, MemoryAlgorithm, IndexType>::type    segment_manager;
typedef CharType                                   char_type;
typedef MemoryAlgorithm                            memory_algorithm;
typedef typename MemoryAlgorithm::mutex_family     mutex_family;
typedef CharType                                   char_t;
typedef typename MemoryAlgorithm::size_type        size_type;
typedef typename MemoryAlgorithm::difference_type  difference_type;
typedef difference_type                            handle_t;
typedef typename segment_manager::
const_named_iterator                            const_named_iterator;
typedef typename segment_manager::
const_unique_iterator                           const_unique_iterator;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

typedef typename
segment_manager::char_ptr_holder_t         char_ptr_holder_t;

typedef typename segment_manager::multiallocation_chain  multiallocation_chain;

#endif   

static const size_type PayloadPerAllocation = segment_manager::PayloadPerAllocation;

private:
typedef basic_managed_memory_impl
<CharType, MemoryAlgorithm, IndexType, Offset> self_t;
protected:
template<class ManagedMemory>
static bool grow(const char *filename, size_type extra_bytes)
{
typedef typename ManagedMemory::device_type device_type;
try{
offset_t old_size;
{
device_type f(open_or_create, filename, read_write);
if(!f.get_size(old_size))
return false;
f.truncate(old_size + extra_bytes);
}
ManagedMemory managed_memory(open_only, filename);
managed_memory.self_t::grow(extra_bytes);
}
catch(...){
return false;
}
return true;
}

template<class ManagedMemory>
static bool shrink_to_fit(const char *filename)
{
typedef typename ManagedMemory::device_type device_type;
size_type new_size;
try{
ManagedMemory managed_memory(open_only, filename);
managed_memory.get_size();
managed_memory.self_t::shrink_to_fit();
new_size = managed_memory.get_size();
}
catch(...){
return false;
}

{
device_type f(open_or_create, filename, read_write);
f.truncate(new_size);
}
return true;
}

basic_managed_memory_impl()
: mp_header(0){}

~basic_managed_memory_impl()
{  this->close_impl(); }

bool  create_impl   (void *addr, size_type size)
{
if(mp_header)  return false;

if(size < segment_manager::get_min_size())
return false;

BOOST_TRY{
BOOST_ASSERT((0 == (std::size_t)addr % boost::move_detail::alignment_of<segment_manager>::value));
mp_header       = ::new(addr, boost_container_new_t()) segment_manager(size);
}
BOOST_CATCH(...){
return false;
}
BOOST_CATCH_END
return true;
}

bool  open_impl     (void *addr, size_type)
{
if(mp_header)  return false;
mp_header = static_cast<segment_manager*>(addr);
return true;
}

bool close_impl()
{
bool ret = mp_header != 0;
mp_header = 0;
return ret;
}

bool destroy_impl()
{
if(mp_header == 0)
return false;
mp_header->~segment_manager();
this->close_impl();
return true;
}

void grow(size_type extra_bytes)
{  mp_header->grow(extra_bytes); }

void shrink_to_fit()
{  mp_header->shrink_to_fit(); }

public:

segment_manager *get_segment_manager() const
{   return mp_header; }

void *   get_address   () const
{   return reinterpret_cast<char*>(mp_header) - Offset; }

size_type   get_size   () const
{   return mp_header->get_size() + Offset;  }

size_type get_free_memory() const
{  return mp_header->get_free_memory();  }

bool all_memory_deallocated()
{   return mp_header->all_memory_deallocated(); }

bool check_sanity()
{   return mp_header->check_sanity(); }

void zero_free_memory()
{   mp_header->zero_free_memory(); }

handle_t get_handle_from_address   (const void *ptr) const
{
return (handle_t)(reinterpret_cast<const char*>(ptr) -
reinterpret_cast<const char*>(this->get_address()));
}

bool belongs_to_segment (const void *ptr) const
{
return ptr >= this->get_address() &&
ptr <  (reinterpret_cast<const char*>(this->get_address()) + this->get_size());
}

void *    get_address_from_handle (handle_t offset) const
{  return reinterpret_cast<char*>(this->get_address()) + offset; }

void* allocate             (size_type nbytes)
{   return mp_header->allocate(nbytes);   }

void* allocate             (size_type nbytes, const std::nothrow_t &tag)
{   return mp_header->allocate(nbytes, tag);  }

void * allocate_aligned (size_type nbytes, size_type alignment, const std::nothrow_t &tag)
{   return mp_header->allocate_aligned(nbytes, alignment, tag);  }

template<class T>
T * allocation_command  (boost::interprocess::allocation_type command,   size_type limit_size,
size_type &prefer_in_recvd_out_size, T *&reuse)
{  return mp_header->allocation_command(command, limit_size, prefer_in_recvd_out_size, reuse);  }

void * allocate_aligned(size_type nbytes, size_type alignment)
{   return mp_header->allocate_aligned(nbytes, alignment);  }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)


void allocate_many(size_type elem_bytes, size_type n_elements, multiallocation_chain &chain)
{  mp_header->allocate_many(elem_bytes, n_elements, chain); }

void allocate_many(const size_type *element_lengths, size_type n_elements, size_type sizeof_element, multiallocation_chain &chain)
{  mp_header->allocate_many(element_lengths, n_elements, sizeof_element, chain); }

void allocate_many(const std::nothrow_t &tag, size_type elem_bytes, size_type n_elements, multiallocation_chain &chain)
{  mp_header->allocate_many(tag, elem_bytes, n_elements, chain); }

void allocate_many(const std::nothrow_t &tag, const size_type *elem_sizes, size_type n_elements, size_type sizeof_element, multiallocation_chain &chain)
{  mp_header->allocate_many(tag, elem_sizes, n_elements, sizeof_element, chain); }

void deallocate_many(multiallocation_chain &chain)
{  mp_header->deallocate_many(chain); }

#endif   

void  deallocate           (void *addr)
{   if (mp_header) mp_header->deallocate(addr);  }

template <class T>
std::pair<T*, size_type> find  (char_ptr_holder_t name)
{   return mp_header->template find<T>(name); }

template <class T>
typename segment_manager::template construct_proxy<T>::type
construct(char_ptr_holder_t name)
{   return mp_header->template construct<T>(name);  }

template <class T>
typename segment_manager::template construct_proxy<T>::type
find_or_construct(char_ptr_holder_t name)
{   return mp_header->template find_or_construct<T>(name);  }

template <class T>
typename segment_manager::template construct_proxy<T>::type
construct(char_ptr_holder_t name, const std::nothrow_t &tag)
{   return mp_header->template construct<T>(name, tag);  }

template <class T>
typename segment_manager::template construct_proxy<T>::type
find_or_construct(char_ptr_holder_t name, const std::nothrow_t &tag)
{   return mp_header->template find_or_construct<T>(name, tag);  }

template <class T>
typename segment_manager::template construct_iter_proxy<T>::type
construct_it(char_ptr_holder_t name)
{   return mp_header->template construct_it<T>(name);  }

template <class T>
typename segment_manager::template construct_iter_proxy<T>::type
find_or_construct_it(char_ptr_holder_t name)
{   return mp_header->template find_or_construct_it<T>(name);  }

template <class T>
typename segment_manager::template construct_iter_proxy<T>::type
construct_it(char_ptr_holder_t name, const std::nothrow_t &tag)
{   return mp_header->template construct_it<T>(name, tag);  }

template <class T>
typename segment_manager::template construct_iter_proxy<T>::type
find_or_construct_it(char_ptr_holder_t name, const std::nothrow_t &tag)
{   return mp_header->template find_or_construct_it<T>(name, tag);  }

template <class Func>
void atomic_func(Func &f)
{   mp_header->atomic_func(f);  }

template <class Func>
bool try_atomic_func(Func &f)
{   return mp_header->try_atomic_func(f); }

template <class T>
bool destroy(const CharType *name)
{   return mp_header->template destroy<T>(name); }

template <class T>
bool destroy(const unique_instance_t *const )
{   return mp_header->template destroy<T>(unique_instance);  }

template <class T>
void destroy_ptr(const T *ptr)
{  mp_header->template destroy_ptr<T>(ptr); }

template<class T>
static const char_type *get_instance_name(const T *ptr)
{  return segment_manager::get_instance_name(ptr);   }

template<class T>
static instance_type get_instance_type(const T *ptr)
{  return segment_manager::get_instance_type(ptr); }

template<class T>
static size_type get_instance_length(const T *ptr)
{  return segment_manager::get_instance_length(ptr); }

void reserve_named_objects(size_type num)
{  mp_header->reserve_named_objects(num);  }

void reserve_unique_objects(size_type num)
{  mp_header->reserve_unique_objects(num);  }

void shrink_to_fit_indexes()
{  mp_header->shrink_to_fit_indexes();  }

size_type get_num_named_objects()
{  return mp_header->get_num_named_objects();  }

size_type get_num_unique_objects()
{  return mp_header->get_num_unique_objects();  }

const_named_iterator named_begin() const
{  return mp_header->named_begin(); }

const_named_iterator named_end() const
{  return mp_header->named_end(); }

const_unique_iterator unique_begin() const
{  return mp_header->unique_begin(); }

const_unique_iterator unique_end() const
{  return mp_header->unique_end(); }

template<class T>
struct allocator
{
typedef typename segment_manager::template allocator<T>::type type;
};

template<class T>
typename allocator<T>::type
get_allocator()
{   return mp_header->template get_allocator<T>(); }

template<class T>
struct deleter
{
typedef typename segment_manager::template deleter<T>::type type;
};

template<class T>
typename deleter<T>::type
get_deleter()
{   return mp_header->template get_deleter<T>(); }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
template <class T>
std::pair<T*, size_type> find_no_lock  (char_ptr_holder_t name)
{   return mp_header->template find_no_lock<T>(name); }
#endif   

protected:
void swap(basic_managed_memory_impl &other)
{  (simple_swap)(mp_header, other.mp_header); }

private:
segment_manager *mp_header;
};

template<class BasicManagedMemoryImpl>
class create_open_func
{
typedef typename BasicManagedMemoryImpl::size_type size_type;

public:

create_open_func(BasicManagedMemoryImpl * const frontend, create_enum_t type)
: m_frontend(frontend), m_type(type){}

bool operator()(void *addr, std::size_t size, bool created) const
{
if( ((m_type == DoOpen)   &&  created) ||
((m_type == DoCreate) && !created) ||
size_type(-1) < size ){
return false;
}
else if(created){
return m_frontend->create_impl(addr, static_cast<size_type>(size));
}
else{
return m_frontend->open_impl  (addr, static_cast<size_type>(size));
}
}

static std::size_t get_min_size()
{
const size_type sz = BasicManagedMemoryImpl::segment_manager::get_min_size();
if(sz > std::size_t(-1)){
BOOST_ASSERT(false);
return std::size_t(-1);
}
else{
return static_cast<std::size_t>(sz);
}
}

private:
BasicManagedMemoryImpl *m_frontend;
create_enum_t           m_type;
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

