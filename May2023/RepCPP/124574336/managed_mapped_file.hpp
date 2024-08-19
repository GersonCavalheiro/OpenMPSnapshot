
#ifndef BOOST_INTERPROCESS_MANAGED_MAPPED_FILE_HPP
#define BOOST_INTERPROCESS_MANAGED_MAPPED_FILE_HPP

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
#include <boost/interprocess/detail/file_wrapper.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/mem_algo/rbtree_best_fit.hpp>
#include <boost/interprocess/sync/mutex_family.hpp>
#include <boost/interprocess/indexes/iset_index.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {

template<class AllocationAlgorithm>
struct mfile_open_or_create
{
typedef  ipcdetail::managed_open_or_create_impl
< file_wrapper, AllocationAlgorithm::Alignment, true, false> type;
};

}  

template
<
class CharType,
class AllocationAlgorithm,
template<class IndexConfig> class IndexType
>
class basic_managed_mapped_file
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
: public ipcdetail::basic_managed_memory_impl
<CharType, AllocationAlgorithm, IndexType
,ipcdetail::mfile_open_or_create<AllocationAlgorithm>::type::ManagedOpenOrCreateUserOffset>
#endif   
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
public:
typedef ipcdetail::basic_managed_memory_impl
<CharType, AllocationAlgorithm, IndexType,
ipcdetail::mfile_open_or_create<AllocationAlgorithm>::type::ManagedOpenOrCreateUserOffset>   base_t;
typedef ipcdetail::file_wrapper device_type;

private:

typedef ipcdetail::create_open_func<base_t>        create_open_func_t;

basic_managed_mapped_file *get_this_pointer()
{  return this;   }

private:
typedef typename base_t::char_ptr_holder_t   char_ptr_holder_t;
BOOST_MOVABLE_BUT_NOT_COPYABLE(basic_managed_mapped_file)
#endif   

public: 

typedef typename BOOST_INTERPROCESS_IMPDEF(base_t::size_type) size_type;

basic_managed_mapped_file()
{}

basic_managed_mapped_file(create_only_t, const char *name,
size_type size, const void *addr = 0, const permissions &perm = permissions())
: m_mfile(create_only, name, size, read_write, addr,
create_open_func_t(get_this_pointer(), ipcdetail::DoCreate), perm)
{}

basic_managed_mapped_file (open_or_create_t,
const char *name, size_type size,
const void *addr = 0, const permissions &perm = permissions())
: m_mfile(open_or_create, name, size, read_write, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpenOrCreate), perm)
{}

basic_managed_mapped_file (open_only_t, const char* name,
const void *addr = 0)
: m_mfile(open_only, name, read_write, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpen))
{}

basic_managed_mapped_file (open_copy_on_write_t, const char* name,
const void *addr = 0)
: m_mfile(open_only, name, copy_on_write, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpen))
{}

basic_managed_mapped_file (open_read_only_t, const char* name,
const void *addr = 0)
: m_mfile(open_only, name, read_only, addr,
create_open_func_t(get_this_pointer(),
ipcdetail::DoOpen))
{}

basic_managed_mapped_file(BOOST_RV_REF(basic_managed_mapped_file) moved)
{
this->swap(moved);
}

basic_managed_mapped_file &operator=(BOOST_RV_REF(basic_managed_mapped_file) moved)
{
basic_managed_mapped_file tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

~basic_managed_mapped_file()
{}

void swap(basic_managed_mapped_file &other)
{
base_t::swap(other);
m_mfile.swap(other.m_mfile);
}

bool flush()
{  return m_mfile.flush();  }

static bool grow(const char *filename, size_type extra_bytes)
{
return base_t::template grow
<basic_managed_mapped_file>(filename, extra_bytes);
}

static bool shrink_to_fit(const char *filename)
{
return base_t::template shrink_to_fit
<basic_managed_mapped_file>(filename);
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <class T>
std::pair<T*, size_type> find  (char_ptr_holder_t name)
{
if(m_mfile.get_mapped_region().get_mode() == read_only){
return base_t::template find_no_lock<T>(name);
}
else{
return base_t::template find<T>(name);
}
}

private:
typename ipcdetail::mfile_open_or_create<AllocationAlgorithm>::type m_mfile;
#endif   
};

#ifdef BOOST_INTERPROCESS_DOXYGEN_INVOKED

typedef basic_managed_mapped_file
<char
,rbtree_best_fit<mutex_family>
,iset_index>
managed_mapped_file;

typedef basic_managed_mapped_file
<wchar_t
,rbtree_best_fit<mutex_family>
,iset_index>
wmanaged_mapped_file;

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
