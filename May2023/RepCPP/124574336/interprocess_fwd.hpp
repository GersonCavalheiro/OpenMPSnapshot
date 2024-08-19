
#ifndef BOOST_INTERPROCESS_FWD_HPP
#define BOOST_INTERPROCESS_FWD_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#ifndef BOOST_CSTDINT_HPP
#  include <boost/cstdint.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/std_fwd.hpp>


#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#include <cstddef>


namespace boost{  namespace intrusive{ }  }
namespace boost{  namespace interprocess{ namespace bi = boost::intrusive; }  }

namespace boost { namespace interprocess {


class permissions;


class shared_memory_object;

#if defined (BOOST_INTERPROCESS_WINDOWS)
class windows_shared_memory;
#endif   

#if defined(BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS)
class xsi_shared_memory;
#endif   


class file_mapping;
class mapped_region;


class null_mutex;

class interprocess_mutex;
class interprocess_recursive_mutex;

class named_mutex;
class named_recursive_mutex;

class interprocess_semaphore;
class named_semaphore;


struct mutex_family;
struct null_mutex_family;


class interprocess_sharable_mutex;
class interprocess_condition;


template <class Mutex>
class scoped_lock;

template <class SharableMutex>
class sharable_lock;

template <class UpgradableMutex>
class upgradable_lock;


template<class T, class SegmentManager>
class allocator;

template<class T, class SegmentManager, std::size_t NodesPerBlock = 64>
class node_allocator;

template<class T, class SegmentManager, std::size_t NodesPerBlock = 64>
class private_node_allocator;

template<class T, class SegmentManager, std::size_t NodesPerBlock = 64>
class cached_node_allocator;

template< class T, class SegmentManager, std::size_t NodesPerBlock = 64
, std::size_t MaxFreeBlocks = 2, unsigned char OverheadPercent = 5 >
class adaptive_pool;

template< class T, class SegmentManager, std::size_t NodesPerBlock = 64
, std::size_t MaxFreeBlocks = 2, unsigned char OverheadPercent = 5 >
class private_adaptive_pool;

template< class T, class SegmentManager, std::size_t NodesPerBlock = 64
, std::size_t MaxFreeBlocks = 2, unsigned char OverheadPercent = 5 >
class cached_adaptive_pool;



static const std::size_t offset_type_alignment = 0;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
#  ifdef BOOST_HAS_INTPTR_T
using ::boost::uintptr_t;
#  else
typedef std::size_t uintptr_t;
#  endif
#endif

template < class T, class DifferenceType = std::ptrdiff_t
, class OffsetType = uintptr_t, std::size_t Alignment = offset_type_alignment>
class offset_ptr;


template<class MutexFamily, class VoidMutex = offset_ptr<void> >
class simple_seq_fit;

template<class MutexFamily, class VoidMutex = offset_ptr<void>, std::size_t MemAlignment = 0>
class rbtree_best_fit;


template<class IndexConfig> class flat_map_index;
template<class IndexConfig> class iset_index;
template<class IndexConfig> class iunordered_set_index;
template<class IndexConfig> class map_index;
template<class IndexConfig> class null_index;
template<class IndexConfig> class unordered_map_index;


template <class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class segment_manager;


template <class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class basic_managed_external_buffer;

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


template <class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class basic_managed_shared_memory;

typedef basic_managed_shared_memory
<char
,rbtree_best_fit<mutex_family>
,iset_index>
managed_shared_memory;

typedef basic_managed_shared_memory
<wchar_t
,rbtree_best_fit<mutex_family>
,iset_index>
wmanaged_shared_memory;



#if defined (BOOST_INTERPROCESS_WINDOWS)

template <class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class basic_managed_windows_shared_memory;

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

#if defined(BOOST_INTERPROCESS_XSI_SHARED_MEMORY_OBJECTS)

template <class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class basic_managed_xsi_shared_memory;

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


typedef basic_managed_shared_memory
<char
,rbtree_best_fit<mutex_family, void*>
,iset_index>
fixed_managed_shared_memory;

typedef basic_managed_shared_memory
<wchar_t
,rbtree_best_fit<mutex_family, void*>
,iset_index>
wfixed_managed_shared_memory;


template
<class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class basic_managed_heap_memory;

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


template
<class CharType
,class MemoryAlgorithm
,template<class IndexConfig> class IndexType>
class basic_managed_mapped_file;

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


class interprocess_exception;
class lock_exception;
class bad_alloc;


template <class CharT
,class CharTraits = std::char_traits<CharT> >
class basic_bufferbuf;

template <class CharT
,class CharTraits = std::char_traits<CharT> >
class basic_ibufferstream;

template <class CharT
,class CharTraits = std::char_traits<CharT> >
class basic_obufferstream;

template <class CharT
,class CharTraits = std::char_traits<CharT> >
class basic_bufferstream;


template <class CharVector
,class CharTraits = std::char_traits<typename CharVector::value_type> >
class basic_vectorbuf;

template <class CharVector
,class CharTraits = std::char_traits<typename CharVector::value_type> >
class basic_ivectorstream;

template <class CharVector
,class CharTraits = std::char_traits<typename CharVector::value_type> >
class basic_ovectorstream;

template <class CharVector
,class CharTraits = std::char_traits<typename CharVector::value_type> >
class basic_vectorstream;


template<class T, class Deleter>
class scoped_ptr;

template<class T, class VoidPointer>
class intrusive_ptr;

template<class T, class VoidAllocator, class Deleter>
class shared_ptr;

template<class T, class VoidAllocator, class Deleter>
class weak_ptr;


template<class VoidPointer>
class message_queue_t;

typedef message_queue_t<offset_ptr<void> > message_queue;

}}  

#endif   

#include <boost/interprocess/detail/config_end.hpp>

#endif 
