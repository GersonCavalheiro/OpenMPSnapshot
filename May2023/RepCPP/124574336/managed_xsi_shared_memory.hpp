
#ifndef BOOST_INTERPROCESS_MANAGED_XSI_SHARED_MEMORY_HPP
#define BOOST_INTERPROCESS_MANAGED_XSI_SHARED_MEMORY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#if !defined(BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS)
#error "This header can't be used in operating systems without XSI (System V) shared memory support"
#endif

#include <boost/interprocess/detail/managed_memory_impl.hpp>
#include <boost/interprocess/detail/managed_open_or_create_impl.hpp>
#include <boost/interprocess/detail/xsi_shared_memory_file_wrapper.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/mem_algo/rbtree_best_fit.hpp>
#include <boost/interprocess/sync/mutex_family.hpp>
#include <boost/interprocess/indexes/iset_index.hpp>

namespace boost {

namespace interprocess {

namespace ipcdetail {

template<class AllocationAlgorithm>
struct xsishmem_open_or_create
{
typedef  ipcdetail::managed_open_or_create_impl                 
< xsi_shared_memory_file_wrapper, AllocationAlgorithm::Alignment, false, true> type;
};

}  

template
<
class CharType,
class AllocationAlgorithm,
template<class IndexConfig> class IndexType
>
class basic_managed_xsi_shared_memory
: public ipcdetail::basic_managed_memory_impl
<CharType, AllocationAlgorithm, IndexType
,ipcdetail::xsishmem_open_or_create<AllocationAlgorithm>::type::ManagedOpenOrCreateUserOffset>
, private ipcdetail::xsishmem_open_or_create<AllocationAlgorithm>::type
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
public:
typedef xsi_shared_memory_file_wrapper device_type;

public:
typedef typename ipcdetail::xsishmem_open_or_create<AllocationAlgorithm>::type base2_t;
typedef ipcdetail::basic_managed_memory_impl
<CharType, AllocationAlgorithm, IndexType,
base2_t::ManagedOpenOrCreateUserOffset>   base_t;

typedef ipcdetail::create_open_func<base_t>        create_open_func_t;

basic_managed_xsi_shared_memory *get_this_pointer()
{  return this;   }

private:
typedef typename base_t::char_ptr_holder_t   char_ptr_holder_t;
BOOST_MOVABLE_BUT_NOT_COPYABLE(basic_managed_xsi_shared_memory)
#endif   

public: 
typedef typename base_t::size_type              size_type;

~basic_managed_xsi_shared_memory()
{}

basic_managed_xsi_shared_memory()
{}

basic_managed_xsi_shared_memory(create_only_t, const xsi_key &key,
std::size_t size, const void *addr = 0, const permissions& perm = permissions())
: base_t()
, base2_t(create_only, key, size, read_write, addr,
create_open_func_t(get_this_pointer(), ipcdetail::DoCreate), perm)
{}

basic_managed_xsi_shared_memory (open_or_create_t,
const xsi_key &key, std::size_t size,
const void *addr = 0, const permissions& perm = permissions())
: base_t()
, base2_t(open_or_create, key, size, read_write, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpenOrCreate), perm)
{}

basic_managed_xsi_shared_memory (open_read_only_t, const xsi_key &key,
const void *addr = 0)
: base_t()
, base2_t(open_only, key, read_only, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpen))
{}

basic_managed_xsi_shared_memory (open_only_t, const xsi_key &key,
const void *addr = 0)
: base_t()
, base2_t(open_only, key, read_write, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpen))
{}

basic_managed_xsi_shared_memory(BOOST_RV_REF(basic_managed_xsi_shared_memory) moved)
{
basic_managed_xsi_shared_memory tmp;
this->swap(moved);
tmp.swap(moved);
}

basic_managed_xsi_shared_memory &operator=(BOOST_RV_REF(basic_managed_xsi_shared_memory) moved)
{
basic_managed_xsi_shared_memory tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

void swap(basic_managed_xsi_shared_memory &other)
{
base_t::swap(other);
base2_t::swap(other);
}

static bool remove(int shmid)
{  return device_type::remove(shmid); }

int get_shmid() const
{  return base2_t::get_device().get_shmid(); }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <class T>
std::pair<T*, std::size_t> find  (char_ptr_holder_t name)
{
if(base2_t::get_mapped_region().get_mode() == read_only){
return base_t::template find_no_lock<T>(name);
}
else{
return base_t::template find<T>(name);
}
}

#endif   
};

#ifdef BOOST_INTERPROCESS_DOXYGEN_INVOKED

typedef basic_managed_xsi_shared_memory
<char
,rbtree_best_fit<mutex_family>
,iset_index>
managed_xsi_shared_memory;

typedef basic_managed_xsi_shared_memory
<wchar_t
,rbtree_best_fit<mutex_family>
,iset_index>
wmanaged_xsi_shared_memory;

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

