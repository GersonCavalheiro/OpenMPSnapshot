
#ifndef BOOST_INTERPROCESS_MANAGED_HEAP_MEMORY_HPP
#define BOOST_INTERPROCESS_MANAGED_HEAP_MEMORY_HPP

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
#include <boost/move/utility_core.hpp>
#include <vector>
#include <boost/interprocess/detail/managed_memory_impl.hpp>
#include <boost/core/no_exceptions_support.hpp>
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
class basic_managed_heap_memory
: public ipcdetail::basic_managed_memory_impl <CharType, AllocationAlgorithm, IndexType>
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:

typedef ipcdetail::basic_managed_memory_impl
<CharType, AllocationAlgorithm, IndexType>             base_t;
BOOST_MOVABLE_BUT_NOT_COPYABLE(basic_managed_heap_memory)
#endif   

public: 
typedef typename base_t::size_type              size_type;

basic_managed_heap_memory(){}

~basic_managed_heap_memory()
{  this->priv_close();  }

basic_managed_heap_memory(size_type size)
:  m_heapmem(size, char(0))
{
if(!base_t::create_impl(&m_heapmem[0], size)){
this->priv_close();
throw interprocess_exception("Could not initialize heap in basic_managed_heap_memory constructor");
}
}

basic_managed_heap_memory(BOOST_RV_REF(basic_managed_heap_memory) moved)
{  this->swap(moved);   }

basic_managed_heap_memory &operator=(BOOST_RV_REF(basic_managed_heap_memory) moved)
{
basic_managed_heap_memory tmp(boost::move(moved));
this->swap(tmp);
return *this;
}

bool grow(size_type extra_bytes)
{
BOOST_TRY{
m_heapmem.resize(m_heapmem.size()+extra_bytes);
}
BOOST_CATCH(...){
return false;
}
BOOST_CATCH_END

base_t::close_impl();
base_t::open_impl(&m_heapmem[0], m_heapmem.size());
base_t::grow(extra_bytes);
return true;
}

void swap(basic_managed_heap_memory &other)
{
base_t::swap(other);
m_heapmem.swap(other.m_heapmem);
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
void priv_close()
{
base_t::destroy_impl();
std::vector<char>().swap(m_heapmem);
}

std::vector<char>  m_heapmem;
#endif   
};

#ifdef BOOST_INTERPROCESS_DOXYGEN_INVOKED

typedef basic_managed_heap_memory
<char
,rbtree_best_fit<null_mutex_family>
,iset_index>
managed_heap_memory;

typedef basic_managed_heap_memory
<wchar_t
,rbtree_best_fit<null_mutex_family>
,iset_index>
wmanaged_heap_memory;

#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

