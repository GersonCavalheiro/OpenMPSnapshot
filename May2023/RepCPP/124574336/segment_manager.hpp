
#ifndef BOOST_INTERPROCESS_SEGMENT_MANAGER_HPP
#define BOOST_INTERPROCESS_SEGMENT_MANAGER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/core/no_exceptions_support.hpp>
#include <boost/interprocess/detail/type_traits.hpp>

#include <boost/interprocess/detail/transform_iterator.hpp>

#include <boost/interprocess/detail/mpl.hpp>
#include <boost/interprocess/detail/nothrow.hpp>
#include <boost/interprocess/detail/segment_manager_helper.hpp>
#include <boost/interprocess/detail/named_proxy.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/indexes/iset_index.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/smart_ptr/deleter.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/container/detail/minimal_char_traits_header.hpp>
#include <boost/container/detail/placement_new.hpp>
#include <cstddef>   
#include <boost/intrusive/detail/minimal_pair_header.hpp>
#include <boost/assert.hpp>
#ifndef BOOST_NO_EXCEPTIONS
#include <exception>
#endif


namespace boost{
namespace interprocess{

template<class MemoryAlgorithm>
class segment_manager_base
:  private MemoryAlgorithm
{
public:
typedef segment_manager_base<MemoryAlgorithm> segment_manager_base_type;
typedef typename MemoryAlgorithm::void_pointer  void_pointer;
typedef typename MemoryAlgorithm::mutex_family  mutex_family;
typedef MemoryAlgorithm memory_algorithm;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

typedef typename MemoryAlgorithm::multiallocation_chain    multiallocation_chain;
typedef typename MemoryAlgorithm::difference_type  difference_type;
typedef typename MemoryAlgorithm::size_type        size_type;

#endif   

static const size_type PayloadPerAllocation = MemoryAlgorithm::PayloadPerAllocation;

segment_manager_base(size_type sz, size_type reserved_bytes)
:  MemoryAlgorithm(sz, reserved_bytes)
{
BOOST_ASSERT((sizeof(segment_manager_base<MemoryAlgorithm>) == sizeof(MemoryAlgorithm)));
}

size_type get_size() const
{  return MemoryAlgorithm::get_size();  }

size_type get_free_memory() const
{  return MemoryAlgorithm::get_free_memory();  }

static size_type get_min_size (size_type size)
{  return MemoryAlgorithm::get_min_size(size);  }

void * allocate (size_type nbytes, const std::nothrow_t &)
{  return MemoryAlgorithm::allocate(nbytes);   }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

void allocate_many(size_type elem_bytes, size_type n_elements, multiallocation_chain &chain)
{
size_type prev_size = chain.size();
MemoryAlgorithm::allocate_many(elem_bytes, n_elements, chain);
if(!elem_bytes || chain.size() == prev_size){
throw bad_alloc();
}
}

void allocate_many(const size_type *element_lengths, size_type n_elements, size_type sizeof_element, multiallocation_chain &chain)
{
size_type prev_size = chain.size();
MemoryAlgorithm::allocate_many(element_lengths, n_elements, sizeof_element, chain);
if(!sizeof_element || chain.size() == prev_size){
throw bad_alloc();
}
}

void allocate_many(const std::nothrow_t &, size_type elem_bytes, size_type n_elements, multiallocation_chain &chain)
{  MemoryAlgorithm::allocate_many(elem_bytes, n_elements, chain); }

void allocate_many(const std::nothrow_t &, const size_type *elem_sizes, size_type n_elements, size_type sizeof_element, multiallocation_chain &chain)
{  MemoryAlgorithm::allocate_many(elem_sizes, n_elements, sizeof_element, chain); }

void deallocate_many(multiallocation_chain &chain)
{  MemoryAlgorithm::deallocate_many(chain); }

#endif   

void * allocate(size_type nbytes)
{
void * ret = MemoryAlgorithm::allocate(nbytes);
if(!ret)
throw bad_alloc();
return ret;
}

void * allocate_aligned (size_type nbytes, size_type alignment, const std::nothrow_t &)
{  return MemoryAlgorithm::allocate_aligned(nbytes, alignment);   }

void * allocate_aligned(size_type nbytes, size_type alignment)
{
void * ret = MemoryAlgorithm::allocate_aligned(nbytes, alignment);
if(!ret)
throw bad_alloc();
return ret;
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template<class T>
T *allocation_command  (boost::interprocess::allocation_type command, size_type limit_size,
size_type &prefer_in_recvd_out_size, T *&reuse)
{
T *ret = MemoryAlgorithm::allocation_command
(command | boost::interprocess::nothrow_allocation, limit_size, prefer_in_recvd_out_size, reuse);
if(!(command & boost::interprocess::nothrow_allocation) && !ret)
throw bad_alloc();
return ret;
}

void *raw_allocation_command  (boost::interprocess::allocation_type command,   size_type limit_objects,
size_type &prefer_in_recvd_out_size, void *&reuse, size_type sizeof_object = 1)
{
void *ret = MemoryAlgorithm::raw_allocation_command
( command | boost::interprocess::nothrow_allocation, limit_objects,
prefer_in_recvd_out_size, reuse, sizeof_object);
if(!(command & boost::interprocess::nothrow_allocation) && !ret)
throw bad_alloc();
return ret;
}

#endif   

void   deallocate          (void *addr)
{  MemoryAlgorithm::deallocate(addr);   }

void grow(size_type extra_size)
{  MemoryAlgorithm::grow(extra_size);   }

void shrink_to_fit()
{  MemoryAlgorithm::shrink_to_fit();   }

bool all_memory_deallocated()
{   return MemoryAlgorithm::all_memory_deallocated(); }

bool check_sanity()
{   return MemoryAlgorithm::check_sanity(); }

void zero_free_memory()
{   MemoryAlgorithm::zero_free_memory(); }

size_type size(const void *ptr) const
{   return MemoryAlgorithm::size(ptr); }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
protected:
void * prot_anonymous_construct
(size_type num, bool dothrow, ipcdetail::in_place_interface &table)
{
typedef ipcdetail::block_header<size_type> block_header_t;
block_header_t block_info (  size_type(table.size*num)
, size_type(table.alignment)
, anonymous_type
, 1
, 0);

void *ptr_struct = this->allocate(block_info.total_size(), nothrow<>::get());

if(!ptr_struct){
if(dothrow){
throw bad_alloc();
}
else{
return 0;
}
}

ipcdetail::mem_algo_deallocator<MemoryAlgorithm> mem(ptr_struct, *this);

block_header_t * hdr = ::new(ptr_struct, boost_container_new_t()) block_header_t(block_info);
void *ptr = 0; 
ptr = hdr->value();

ipcdetail::array_construct(ptr, num, table);

mem.release();
return ptr;
}

void prot_anonymous_destroy(const void *object, ipcdetail::in_place_interface &table)
{

typedef ipcdetail::block_header<size_type> block_header_t;
block_header_t *ctrl_data = block_header_t::block_header_from_value(object, table.size, table.alignment);


if(ctrl_data->alloc_type() != anonymous_type){
BOOST_ASSERT(0);
}

std::size_t destroyed = 0;
table.destroy_n(const_cast<void*>(object), ctrl_data->m_value_bytes/table.size, destroyed);
this->deallocate(ctrl_data);
}
#endif   
};

template<class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class segment_manager
:  public segment_manager_base<MemoryAlgorithm>
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
segment_manager();
segment_manager(const segment_manager &);
segment_manager &operator=(const segment_manager &);
typedef segment_manager_base<MemoryAlgorithm> segment_manager_base_t;
#endif   

public:
typedef MemoryAlgorithm                                  memory_algorithm;
typedef typename segment_manager_base_t::void_pointer    void_pointer;
typedef typename segment_manager_base_t::size_type       size_type;
typedef typename segment_manager_base_t::difference_type difference_type;
typedef CharType                                         char_type;

typedef segment_manager_base<MemoryAlgorithm>   segment_manager_base_type;

static const size_type PayloadPerAllocation = segment_manager_base_t::PayloadPerAllocation;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef ipcdetail::block_header<size_type> block_header_t;
typedef ipcdetail::index_config<CharType, MemoryAlgorithm>  index_config_named;
typedef ipcdetail::index_config<char, MemoryAlgorithm>      index_config_unique;
typedef IndexType<index_config_named>                    index_type;
typedef ipcdetail::bool_<is_intrusive_index<index_type>::value >    is_intrusive_t;
typedef ipcdetail::bool_<is_node_index<index_type>::value>          is_node_index_t;

public:
typedef IndexType<index_config_named>                    named_index_t;
typedef IndexType<index_config_unique>                   unique_index_t;
typedef ipcdetail::char_ptr_holder<CharType>                char_ptr_holder_t;
typedef ipcdetail::segment_manager_iterator_transform
<typename named_index_t::const_iterator
,is_intrusive_index<index_type>::value>   named_transform;

typedef ipcdetail::segment_manager_iterator_transform
<typename unique_index_t::const_iterator
,is_intrusive_index<index_type>::value>   unique_transform;
#endif   

typedef typename segment_manager_base_t::mutex_family       mutex_family;

typedef transform_iterator
<typename named_index_t::const_iterator, named_transform> const_named_iterator;
typedef transform_iterator
<typename unique_index_t::const_iterator, unique_transform> const_unique_iterator;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template<class T>
struct construct_proxy
{
typedef ipcdetail::named_proxy<segment_manager, T, false>   type;
};

template<class T>
struct construct_iter_proxy
{
typedef ipcdetail::named_proxy<segment_manager, T, true>   type;
};

#endif   

explicit segment_manager(size_type segment_size)
:  segment_manager_base_t(segment_size, priv_get_reserved_bytes())
,  m_header(static_cast<segment_manager_base_t*>(get_this_pointer()))
{
(void) anonymous_instance;   (void) unique_instance;
const void * const this_addr = this;
const void *const segm_addr  = static_cast<segment_manager_base_t*>(this);
(void)this_addr;  (void)segm_addr;
BOOST_ASSERT( this_addr == segm_addr);
const std::size_t void_ptr_alignment = boost::move_detail::alignment_of<void_pointer>::value; (void)void_ptr_alignment;
BOOST_ASSERT((0 == (std::size_t)this_addr % boost::move_detail::alignment_of<segment_manager>::value));
}

template <class T>
std::pair<T*, size_type> find  (char_ptr_holder_t name)
{  return this->priv_find_impl<T>(name, true);  }

template <class T>
std::pair<T*, size_type> find_no_lock  (char_ptr_holder_t name)
{  return this->priv_find_impl<T>(name, false);  }

template <class T>
typename construct_proxy<T>::type
construct(char_ptr_holder_t name)
{  return typename construct_proxy<T>::type (this, name, false, true);  }

template <class T>
typename construct_proxy<T>::type find_or_construct(char_ptr_holder_t name)
{  return typename construct_proxy<T>::type (this, name, true, true);  }

template <class T>
typename construct_proxy<T>::type
construct(char_ptr_holder_t name, const std::nothrow_t &)
{  return typename construct_proxy<T>::type (this, name, false, false);  }

template <class T>
typename construct_proxy<T>::type
find_or_construct(char_ptr_holder_t name, const std::nothrow_t &)
{  return typename construct_proxy<T>::type (this, name, true, false);  }

template <class T>
typename construct_iter_proxy<T>::type
construct_it(char_ptr_holder_t name)
{  return typename construct_iter_proxy<T>::type (this, name, false, true);  }

template <class T>
typename construct_iter_proxy<T>::type
find_or_construct_it(char_ptr_holder_t name)
{  return typename construct_iter_proxy<T>::type (this, name, true, true);  }

template <class T>
typename construct_iter_proxy<T>::type
construct_it(char_ptr_holder_t name, const std::nothrow_t &)
{  return typename construct_iter_proxy<T>::type (this, name, false, false);  }

template <class T>
typename construct_iter_proxy<T>::type
find_or_construct_it(char_ptr_holder_t name, const std::nothrow_t &)
{  return typename construct_iter_proxy<T>::type (this, name, true, false);  }

template <class Func>
void atomic_func(Func &f)
{  scoped_lock<rmutex> guard(m_header);  f();  }

template <class Func>
bool try_atomic_func(Func &f)
{
scoped_lock<rmutex> guard(m_header, try_to_lock);
if(guard){
f();
return true;
}
else{
return false;
}
}

template <class T>
bool destroy(char_ptr_holder_t name)
{
BOOST_ASSERT(!name.is_anonymous());
ipcdetail::placement_destroy<T> dtor;

if(name.is_unique()){
return this->priv_generic_named_destroy<char>
( typeid(T).name(), m_header.m_unique_index , dtor, is_intrusive_t());
}
else{
return this->priv_generic_named_destroy<CharType>
( name.get(), m_header.m_named_index, dtor, is_intrusive_t());
}
}

template <class T>
void destroy_ptr(const T *p)
{
typedef typename ipcdetail::char_if_void<T>::type data_t;
ipcdetail::placement_destroy<data_t> dtor;
priv_destroy_ptr(p, dtor);
}

template<class T>
static const CharType *get_instance_name(const T *ptr)
{ return priv_get_instance_name(block_header_t::block_header_from_value(ptr));  }

template<class T>
static size_type get_instance_length(const T *ptr)
{  return priv_get_instance_length(block_header_t::block_header_from_value(ptr), sizeof(T));  }

template<class T>
static instance_type get_instance_type(const T *ptr)
{  return priv_get_instance_type(block_header_t::block_header_from_value(ptr));  }

void reserve_named_objects(size_type num)
{
scoped_lock<rmutex> guard(m_header);
m_header.m_named_index.reserve(num);
}

void reserve_unique_objects(size_type num)
{
scoped_lock<rmutex> guard(m_header);
m_header.m_unique_index.reserve(num);
}

void shrink_to_fit_indexes()
{
scoped_lock<rmutex> guard(m_header);
m_header.m_named_index.shrink_to_fit();
m_header.m_unique_index.shrink_to_fit();
}

size_type get_num_named_objects()
{
scoped_lock<rmutex> guard(m_header);
return m_header.m_named_index.size();
}

size_type get_num_unique_objects()
{
scoped_lock<rmutex> guard(m_header);
return m_header.m_unique_index.size();
}

static size_type get_min_size()
{  return segment_manager_base_t::get_min_size(priv_get_reserved_bytes());  }

const_named_iterator named_begin() const
{
return (make_transform_iterator)
(m_header.m_named_index.begin(), named_transform());
}

const_named_iterator named_end() const
{
return (make_transform_iterator)
(m_header.m_named_index.end(), named_transform());
}

const_unique_iterator unique_begin() const
{
return (make_transform_iterator)
(m_header.m_unique_index.begin(), unique_transform());
}

const_unique_iterator unique_end() const
{
return (make_transform_iterator)
(m_header.m_unique_index.end(), unique_transform());
}

template<class T>
struct allocator
{
typedef boost::interprocess::allocator<T, segment_manager> type;
};

template<class T>
typename allocator<T>::type
get_allocator()
{   return typename allocator<T>::type(this); }

template<class T>
struct deleter
{
typedef boost::interprocess::deleter<T, segment_manager> type;
};

template<class T>
typename deleter<T>::type
get_deleter()
{   return typename deleter<T>::type(this); }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template<class T>
T *generic_construct(const CharType *name,
size_type num,
bool try2find,
bool dothrow,
ipcdetail::in_place_interface &table)
{
return static_cast<T*>
(priv_generic_construct(name, num, try2find, dothrow, table));
}

private:
template <class T>
std::pair<T*, size_type> priv_find_impl (const CharType* name, bool lock)
{
BOOST_ASSERT(name != 0);
ipcdetail::placement_destroy<T> table;
size_type sz;
void *ret;

if(name == reinterpret_cast<const CharType*>(-1)){
ret = priv_generic_find<char> (typeid(T).name(), m_header.m_unique_index, table, sz, is_intrusive_t(), lock);
}
else{
ret = priv_generic_find<CharType> (name, m_header.m_named_index, table, sz, is_intrusive_t(), lock);
}
return std::pair<T*, size_type>(static_cast<T*>(ret), sz);
}

template <class T>
std::pair<T*, size_type> priv_find_impl (const ipcdetail::unique_instance_t* name, bool lock)
{
ipcdetail::placement_destroy<T> table;
size_type size;
void *ret = priv_generic_find<char>(name, m_header.m_unique_index, table, size, is_intrusive_t(), lock);
return std::pair<T*, size_type>(static_cast<T*>(ret), size);
}

void *priv_generic_construct
(const CharType *name, size_type num, bool try2find, bool dothrow, ipcdetail::in_place_interface &table)
{
void *ret;
if(num > ((std::size_t)-1)/table.size){
if(dothrow)
throw bad_alloc();
else
return 0;
}
if(name == 0){
ret = this->prot_anonymous_construct(num, dothrow, table);
}
else if(name == reinterpret_cast<const CharType*>(-1)){
ret = this->priv_generic_named_construct<char>
(unique_type, table.type_name, num, try2find, dothrow, table, m_header.m_unique_index, is_intrusive_t());
}
else{
ret = this->priv_generic_named_construct<CharType>
(named_type, name, num, try2find, dothrow, table, m_header.m_named_index, is_intrusive_t());
}
return ret;
}

void priv_destroy_ptr(const void *ptr, ipcdetail::in_place_interface &dtor)
{
block_header_t *ctrl_data = block_header_t::block_header_from_value(ptr, dtor.size, dtor.alignment);
switch(ctrl_data->alloc_type()){
case anonymous_type:
this->prot_anonymous_destroy(ptr, dtor);
break;

case named_type:
this->priv_generic_named_destroy<CharType>
(ctrl_data, m_header.m_named_index, dtor, is_node_index_t());
break;

case unique_type:
this->priv_generic_named_destroy<char>
(ctrl_data, m_header.m_unique_index, dtor, is_node_index_t());
break;

default:
BOOST_ASSERT(0);
break;
}
}

static const CharType *priv_get_instance_name(block_header_t *ctrl_data)
{
boost::interprocess::allocation_type type = ctrl_data->alloc_type();
if(type == anonymous_type){
BOOST_ASSERT((type == anonymous_type && ctrl_data->m_num_char == 0) ||
(type == unique_type    && ctrl_data->m_num_char != 0) );
return 0;
}
CharType *name = static_cast<CharType*>(ctrl_data->template name<CharType>());

BOOST_ASSERT(ctrl_data->sizeof_char() == sizeof(CharType));
BOOST_ASSERT(ctrl_data->m_num_char == std::char_traits<CharType>::length(name));
return name;
}

static size_type priv_get_instance_length(block_header_t *ctrl_data, size_type sizeofvalue)
{
BOOST_ASSERT((ctrl_data->value_bytes() %sizeofvalue) == 0);
return ctrl_data->value_bytes()/sizeofvalue;
}

static instance_type priv_get_instance_type(block_header_t *ctrl_data)
{
BOOST_ASSERT((instance_type)ctrl_data->alloc_type() < max_allocation_type);
return (instance_type)ctrl_data->alloc_type();
}

static size_type priv_get_reserved_bytes()
{
return sizeof(segment_manager) - sizeof(segment_manager_base_t);
}

template <class CharT>
void *priv_generic_find
(const CharT* name,
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index,
ipcdetail::in_place_interface &table,
size_type &length, ipcdetail::true_ is_intrusive, bool use_lock)
{
(void)is_intrusive;
typedef IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >         index_type;
typedef typename index_type::iterator           index_it;

scoped_lock<rmutex> guard(priv_get_lock(use_lock));
ipcdetail::intrusive_compare_key<CharT> key
(name, std::char_traits<CharT>::length(name));
index_it it = index.find(key);

void *ret_ptr  = 0;
length         = 0;

if(it != index.end()){
block_header_t *ctrl_data = it->get_block_header();

BOOST_ASSERT((ctrl_data->m_value_bytes % table.size) == 0);
BOOST_ASSERT(ctrl_data->sizeof_char() == sizeof(CharT));
ret_ptr  = ctrl_data->value();
length  = ctrl_data->m_value_bytes/table.size;
}
return ret_ptr;
}

template <class CharT>
void *priv_generic_find
(const CharT* name,
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index,
ipcdetail::in_place_interface &table,
size_type &length, ipcdetail::false_ is_intrusive, bool use_lock)
{
(void)is_intrusive;
typedef IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >      index_type;
typedef typename index_type::key_type        key_type;
typedef typename index_type::iterator        index_it;

scoped_lock<rmutex> guard(priv_get_lock(use_lock));
index_it it = index.find(key_type(name, std::char_traits<CharT>::length(name)));

void *ret_ptr  = 0;
length         = 0;

if(it != index.end()){
block_header_t *ctrl_data = reinterpret_cast<block_header_t*>
(ipcdetail::to_raw_pointer(it->second.m_ptr));

BOOST_ASSERT((ctrl_data->m_value_bytes % table.size) == 0);
BOOST_ASSERT(ctrl_data->sizeof_char() == sizeof(CharT));
ret_ptr  = ctrl_data->value();
length  = ctrl_data->m_value_bytes/table.size;
}
return ret_ptr;
}

template <class CharT>
bool priv_generic_named_destroy
(block_header_t *block_header,
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index,
ipcdetail::in_place_interface &table, ipcdetail::true_ is_node_index)
{
(void)is_node_index;
typedef typename IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >::iterator index_it;

index_it *ihdr = block_header_t::template to_first_header<index_it>(block_header);
return this->priv_generic_named_destroy_impl<CharT>(*ihdr, index, table);
}

template <class CharT>
bool priv_generic_named_destroy
(block_header_t *block_header,
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index,
ipcdetail::in_place_interface &table,
ipcdetail::false_ is_node_index)
{
(void)is_node_index;
CharT *name = static_cast<CharT*>(block_header->template name<CharT>());
return this->priv_generic_named_destroy<CharT>(name, index, table, is_intrusive_t());
}

template <class CharT>
bool priv_generic_named_destroy(const CharT *name,
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index,
ipcdetail::in_place_interface &table, ipcdetail::true_ is_intrusive_index)
{
(void)is_intrusive_index;
typedef IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >         index_type;
typedef typename index_type::iterator           index_it;
typedef typename index_type::value_type         intrusive_value_type;

scoped_lock<rmutex> guard(m_header);
ipcdetail::intrusive_compare_key<CharT> key
(name, std::char_traits<CharT>::length(name));
index_it it = index.find(key);

if(it == index.end()){
return false;
}

block_header_t *ctrl_data = it->get_block_header();
intrusive_value_type *iv = intrusive_value_type::get_intrusive_value_type(ctrl_data);
void *memory = iv;
void *values = ctrl_data->value();
std::size_t num = ctrl_data->m_value_bytes/table.size;

BOOST_ASSERT((ctrl_data->m_value_bytes % table.size) == 0);
BOOST_ASSERT(sizeof(CharT) == ctrl_data->sizeof_char());

index.erase(it);

ctrl_data->~block_header_t();
iv->~intrusive_value_type();

std::size_t destroyed;
table.destroy_n(values, num, destroyed);
this->deallocate(memory);
return true;
}

template <class CharT>
bool priv_generic_named_destroy(const CharT *name,
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index,
ipcdetail::in_place_interface &table,
ipcdetail::false_ is_intrusive_index)
{
(void)is_intrusive_index;
typedef IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >            index_type;
typedef typename index_type::iterator              index_it;
typedef typename index_type::key_type              key_type;

scoped_lock<rmutex> guard(m_header);
index_it it = index.find(key_type (name,
std::char_traits<CharT>::length(name)));

if(it == index.end()){
return false;
}
return this->priv_generic_named_destroy_impl<CharT>(it, index, table);
}

template <class CharT>
bool priv_generic_named_destroy_impl
(const typename IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >::iterator &it,
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index,
ipcdetail::in_place_interface &table)
{
typedef IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >      index_type;
typedef typename index_type::iterator        index_it;

block_header_t *ctrl_data = reinterpret_cast<block_header_t*>
(ipcdetail::to_raw_pointer(it->second.m_ptr));
char *stored_name       = static_cast<char*>(static_cast<void*>(const_cast<CharT*>(it->first.name())));
(void)stored_name;

std::size_t num = ctrl_data->m_value_bytes/table.size;
void *values = ctrl_data->value();

BOOST_ASSERT((ctrl_data->m_value_bytes % table.size) == 0);
BOOST_ASSERT(static_cast<void*>(stored_name) == static_cast<void*>(ctrl_data->template name<CharT>()));
BOOST_ASSERT(sizeof(CharT) == ctrl_data->sizeof_char());

index.erase(it);

ctrl_data->~block_header_t();

void *memory;
if(is_node_index_t::value){
index_it *ihdr = block_header_t::template
to_first_header<index_it>(ctrl_data);
ihdr->~index_it();
memory = ihdr;
}
else{
memory = ctrl_data;
}

std::size_t destroyed;
table.destroy_n(values, num, destroyed);
this->deallocate(memory);
return true;
}

template<class CharT>
void * priv_generic_named_construct
(unsigned char type, const CharT *name, size_type num, bool try2find,
bool dothrow, ipcdetail::in_place_interface &table, 
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index, ipcdetail::true_ is_intrusive)
{
(void)is_intrusive;
std::size_t namelen  = std::char_traits<CharT>::length(name);

block_header_t block_info ( size_type(table.size*num)
, size_type(table.alignment)
, type
, sizeof(CharT)
, namelen);

typedef IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >            index_type;
typedef typename index_type::iterator              index_it;
typedef std::pair<index_it, bool>                  index_ib;

scoped_lock<rmutex> guard(m_header);
index_ib insert_ret;

typename index_type::insert_commit_data   commit_data;
typedef typename index_type::value_type   intrusive_value_type;

BOOST_TRY{
ipcdetail::intrusive_compare_key<CharT> key(name, namelen);
insert_ret = index.insert_check(key, commit_data);
}
BOOST_CATCH(...){
if(dothrow)
BOOST_RETHROW
return 0;
}
BOOST_CATCH_END

index_it it = insert_ret.first;

if(!insert_ret.second){
if(try2find){
return it->get_block_header()->value();
}
if(dothrow){
throw interprocess_exception(already_exists_error);
}
else{
return 0;
}
}

void *buffer_ptr;

if(dothrow){
buffer_ptr = this->allocate
(block_info.template total_size_with_header<intrusive_value_type>());
}
else{
buffer_ptr = this->allocate
(block_info.template total_size_with_header<intrusive_value_type>(), nothrow<>::get());
if(!buffer_ptr)
return 0;
}

intrusive_value_type * intrusive_hdr = ::new(buffer_ptr, boost_container_new_t()) intrusive_value_type();
block_header_t * hdr = ::new(intrusive_hdr->get_block_header(), boost_container_new_t())block_header_t(block_info);
void *ptr = 0; 
ptr = hdr->value();

CharT *name_ptr = static_cast<CharT *>(hdr->template name<CharT>());
std::char_traits<CharT>::copy(name_ptr, name, namelen+1);

BOOST_TRY{
it = index.insert_commit(*intrusive_hdr, commit_data);
}
BOOST_CATCH(...){
if(dothrow)
BOOST_RETHROW
return 0;
}
BOOST_CATCH_END

ipcdetail::mem_algo_deallocator<segment_manager_base_type> mem
(buffer_ptr, *static_cast<segment_manager_base_type*>(this));

value_eraser<index_type> v_eraser(index, it);

ipcdetail::array_construct(ptr, num, table);

v_eraser.release();
mem.release();
return ptr;
}

template<class CharT>
void * priv_generic_named_construct
(unsigned char type, const CharT *name, size_type num, bool try2find, bool dothrow,
ipcdetail::in_place_interface &table, 
IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> > &index, ipcdetail::false_ is_intrusive)
{
(void)is_intrusive;
std::size_t namelen  = std::char_traits<CharT>::length(name);

block_header_t block_info ( size_type(table.size*num)
, size_type(table.alignment)
, type
, sizeof(CharT)
, namelen);

typedef IndexType<ipcdetail::index_config<CharT, MemoryAlgorithm> >            index_type;
typedef typename index_type::key_type              key_type;
typedef typename index_type::mapped_type           mapped_type;
typedef typename index_type::value_type            value_type;
typedef typename index_type::iterator              index_it;
typedef std::pair<index_it, bool>                  index_ib;

scoped_lock<rmutex> guard(m_header);
index_ib insert_ret;
BOOST_TRY{
insert_ret = index.insert(value_type(key_type (name, namelen), mapped_type(0)));
}
BOOST_CATCH(...){
if(dothrow)
BOOST_RETHROW;
return 0;
}
BOOST_CATCH_END

index_it it = insert_ret.first;

if(!insert_ret.second){
if(try2find){
block_header_t *hdr = static_cast<block_header_t*>
(ipcdetail::to_raw_pointer(it->second.m_ptr));
return hdr->value();
}
return 0;
}
value_eraser<index_type> v_eraser(index, it);

void *buffer_ptr;
block_header_t * hdr;

if(is_node_index_t::value){
size_type total_size = block_info.template total_size_with_header<index_it>();
if(dothrow){
buffer_ptr = this->allocate(total_size);
}
else{
buffer_ptr = this->allocate(total_size, nothrow<>::get());
if(!buffer_ptr)
return 0;
}
index_it *idr = ::new(buffer_ptr, boost_container_new_t()) index_it(it);
hdr = block_header_t::template from_first_header<index_it>(idr);
}
else{
if(dothrow){
buffer_ptr = this->allocate(block_info.total_size());
}
else{
buffer_ptr = this->allocate(block_info.total_size(), nothrow<>::get());
if(!buffer_ptr)
return 0;
}
hdr = static_cast<block_header_t*>(buffer_ptr);
}

hdr = ::new(hdr, boost_container_new_t())block_header_t(block_info);
void *ptr = 0; 
ptr = hdr->value();

CharT *name_ptr = static_cast<CharT *>(hdr->template name<CharT>());
std::char_traits<CharT>::copy(name_ptr, name, namelen+1);

const_cast<key_type &>(it->first).name(name_ptr);
it->second.m_ptr  = hdr;

ipcdetail::mem_algo_deallocator<segment_manager_base_type> mem
(buffer_ptr, *static_cast<segment_manager_base_type*>(this));

ipcdetail::array_construct(ptr, num, table);

mem.release();

v_eraser.release();
return ptr;
}

private:
segment_manager *get_this_pointer()
{  return this;  }

typedef typename MemoryAlgorithm::mutex_family::recursive_mutex_type   rmutex;

scoped_lock<rmutex> priv_get_lock(bool use_lock)
{
scoped_lock<rmutex> local(m_header, defer_lock);
if(use_lock){
local.lock();
}
return scoped_lock<rmutex>(boost::move(local));
}

struct header_t
:  public rmutex
{
named_index_t           m_named_index;
unique_index_t          m_unique_index;

header_t(segment_manager_base_t *segment_mngr_base)
:  m_named_index (segment_mngr_base)
,  m_unique_index(segment_mngr_base)
{}
}  m_header;

#endif   
};


}} 

#include <boost/interprocess/detail/config_end.hpp>

#endif 

