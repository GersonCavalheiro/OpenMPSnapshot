
#ifndef BOOST_INTERPROCESS_MANAGED_EXTERNAL_BUFFER_HPP
#define BOOST_INTERPROCESS_MANAGED_EXTERNAL_BUFFER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/managed_memory_impl.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/assert.hpp>
#include <boost/interprocess/mem_algo/rbtree_best_fit.hpp>
#include <boost/interprocess/sync/mutex_family.hpp>
#include <boost/interprocess/indexes/iset_index.hpp>


namespace boost {
namespace interprocess {

template
<
class CharType,
class AllocationAlgorithm,
template<class IndexConfig> class IndexType
>
class basic_managed_external_buffer
: public ipcdetail::basic_managed_memory_impl <CharType, AllocationAlgorithm, IndexType>
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef ipcdetail::basic_managed_memory_impl
<CharType, AllocationAlgorithm, IndexType>    base_t;
BOOST_MOVABLE_BUT_NOT_COPYABLE(basic_managed_external_buffer)
#endif   

public:
typedef typename base_t::size_type              size_type;

basic_managed_external_buffer()
{}

basic_managed_external_buffer
(create_only_t, void *addr, size_type size)
{
BOOST_ASSERT((0 == (((std::size_t)addr) & (AllocationAlgorithm::Alignment - size_type(1u)))));
if(!base_t::create_impl(addr, size)){
throw interprocess_exception("Could not initialize buffer in basic_managed_external_buffer constructor");
}
}

basic_managed_external_buffer
(open_only_t, void *addr, size_type size)
{
BOOST_ASSERT((0 == (((std::size_t)addr) & (AllocationAlgorithm::Alignment - size_type(1u)))));
if(!base_t::open_impl(addr, size)){
throw interprocess_exception("Could not initialize buffer in basic_managed_external_buffer constructor");
}
}

basic_managed_external_buffer(BOOST_RV_REF(basic_managed_external_buffer) moved)
{
this->swap(moved);
}

basic_managed_external_buffer &operator=(BOOST_RV_REF(basic_managed_external_buffer) moved)
{
basic_managed_external_buffer tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

void grow(size_type extra_bytes)
{  base_t::grow(extra_bytes);   }

void swap(basic_managed_external_buffer &other)
{  base_t::swap(other); }
};

#ifdef BOOST_INTERPROCESS_DOXYGEN_INVOKED

typedef basic_managed_external_buffer
<char
,rbtree_best_fit<null_mutex_family>
,iset_index>
managed_external_buffer;

typedef basic_managed_external_buffer
<wchar_t
,rbtree_best_fit<null_mutex_family>
,iset_index>
wmanaged_external_buffer;

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

