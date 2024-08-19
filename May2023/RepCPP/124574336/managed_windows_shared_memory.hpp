
#ifndef BOOST_INTERPROCESS_MANAGED_WINDOWS_SHARED_MEMORY_HPP
#define BOOST_INTERPROCESS_MANAGED_WINDOWS_SHARED_MEMORY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/managed_open_or_create_impl.hpp>
#include <boost/interprocess/detail/managed_memory_impl.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/windows_shared_memory.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/mem_algo/rbtree_best_fit.hpp>
#include <boost/interprocess/sync/mutex_family.hpp>
#include <boost/interprocess/indexes/iset_index.hpp>

namespace boost {
namespace interprocess {

namespace ipcdetail {

template<class AllocationAlgorithm>
struct wshmem_open_or_create
{
typedef  ipcdetail::managed_open_or_create_impl
< windows_shared_memory, AllocationAlgorithm::Alignment, false, false> type;
};

}  

template
<
class CharType,
class AllocationAlgorithm,
template<class IndexConfig> class IndexType
>
class basic_managed_windows_shared_memory
: public ipcdetail::basic_managed_memory_impl
< CharType, AllocationAlgorithm, IndexType
, ipcdetail::wshmem_open_or_create<AllocationAlgorithm>::type::ManagedOpenOrCreateUserOffset>
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef ipcdetail::basic_managed_memory_impl
<CharType, AllocationAlgorithm, IndexType,
ipcdetail::wshmem_open_or_create<AllocationAlgorithm>::type::ManagedOpenOrCreateUserOffset>   base_t;
typedef ipcdetail::create_open_func<base_t>        create_open_func_t;

basic_managed_windows_shared_memory *get_this_pointer()
{  return this;   }

private:
typedef typename base_t::char_ptr_holder_t   char_ptr_holder_t;
BOOST_MOVABLE_BUT_NOT_COPYABLE(basic_managed_windows_shared_memory)
#endif   

public: 
typedef typename base_t::size_type              size_type;

basic_managed_windows_shared_memory()
{}

basic_managed_windows_shared_memory
(create_only_t, const char *name,
size_type size, const void *addr = 0, const permissions &perm = permissions())
: m_wshm(create_only, name, size, read_write, addr,
create_open_func_t(get_this_pointer(), ipcdetail::DoCreate), perm)
{}

basic_managed_windows_shared_memory
(open_or_create_t,
const char *name, size_type size,
const void *addr = 0,
const permissions &perm = permissions())
: m_wshm(open_or_create, name, size, read_write, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpenOrCreate), perm)
{}

basic_managed_windows_shared_memory
(open_only_t, const char* name, const void *addr = 0)
: m_wshm(open_only, name, read_write, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpen))
{}

basic_managed_windows_shared_memory
(open_copy_on_write_t, const char* name, const void *addr = 0)
: m_wshm(open_only, name, copy_on_write, addr,
create_open_func_t(get_this_pointer(), ipcdetail::DoOpen))
{}

basic_managed_windows_shared_memory
(open_read_only_t, const char* name, const void *addr = 0)
: base_t()
, m_wshm(open_only, name, read_only, addr,
create_open_func_t(get_this_pointer(), ipcdetail::DoOpen))
{}

basic_managed_windows_shared_memory
(BOOST_RV_REF(basic_managed_windows_shared_memory) moved)
{  this->swap(moved);   }

basic_managed_windows_shared_memory &operator=(BOOST_RV_REF(basic_managed_windows_shared_memory) moved)
{
basic_managed_windows_shared_memory tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

~basic_managed_windows_shared_memory()
{}

void swap(basic_managed_windows_shared_memory &other)
{
base_t::swap(other);
m_wshm.swap(other.m_wshm);
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <class T>
std::pair<T*, size_type> find  (char_ptr_holder_t name)
{
if(m_wshm.get_mapped_region().get_mode() == read_only){
return base_t::template find_no_lock<T>(name);
}
else{
return base_t::template find<T>(name);
}
}

private:
typename ipcdetail::wshmem_open_or_create<AllocationAlgorithm>::type m_wshm;
#endif   
};

#ifdef BOOST_INTERPROCESS_DOXYGEN_INVOKED

typedef basic_managed_windows_shared_memory
<char
,rbtree_best_fit<mutex_family>
,iset_index>
managed_windows_shared_memory;

typedef basic_managed_windows_shared_memory
<wchar_t
,rbtree_best_fit<mutex_family>
,iset_index>
wmanaged_windows_shared_memory;

#endif   


}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
