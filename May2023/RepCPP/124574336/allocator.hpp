
#ifndef BOOST_INTERPROCESS_ALLOCATOR_HPP
#define BOOST_INTERPROCESS_ALLOCATOR_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/intrusive/pointer_traits.hpp>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/containers/allocation_type.hpp>
#include <boost/container/detail/multiallocation_chain.hpp>
#include <boost/interprocess/allocators/detail/allocator_common.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/containers/version_type.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/assert.hpp>
#include <boost/utility/addressof.hpp>
#include <boost/interprocess/detail/type_traits.hpp>
#include <boost/container/detail/placement_new.hpp>

#include <cstddef>
#include <stdexcept>


namespace boost {
namespace interprocess {


template<class T, class SegmentManager>
class allocator
{
public:
typedef SegmentManager                                segment_manager;
typedef typename SegmentManager::void_pointer         void_pointer;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:

typedef allocator<T, SegmentManager>   self_t;

typedef typename segment_manager::void_pointer  aux_pointer_t;

typedef typename boost::intrusive::
pointer_traits<aux_pointer_t>::template
rebind_pointer<const void>::type          cvoid_ptr;

typedef typename boost::intrusive::
pointer_traits<cvoid_ptr>::template
rebind_pointer<segment_manager>::type          alloc_ptr_t;

template<class T2, class SegmentManager2>
allocator& operator=(const allocator<T2, SegmentManager2>&);

allocator& operator=(const allocator&);

alloc_ptr_t mp_mngr;
#endif   

public:
typedef T                                    value_type;
typedef typename boost::intrusive::
pointer_traits<cvoid_ptr>::template
rebind_pointer<T>::type                pointer;
typedef typename boost::intrusive::
pointer_traits<pointer>::template
rebind_pointer<const T>::type          const_pointer;
typedef typename ipcdetail::add_reference
<value_type>::type         reference;
typedef typename ipcdetail::add_reference
<const value_type>::type   const_reference;
typedef typename segment_manager::size_type               size_type;
typedef typename segment_manager::difference_type         difference_type;

typedef boost::interprocess::version_type<allocator, 2>   version;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

typedef boost::container::dtl::transform_multiallocation_chain
<typename SegmentManager::multiallocation_chain, T>multiallocation_chain;
#endif   

template<class T2>
struct rebind
{
typedef allocator<T2, SegmentManager>     other;
};

segment_manager* get_segment_manager()const
{  return ipcdetail::to_raw_pointer(mp_mngr);   }

allocator(segment_manager *segment_mngr)
: mp_mngr(segment_mngr) { }

allocator(const allocator &other)
: mp_mngr(other.get_segment_manager()){ }

template<class T2>
allocator(const allocator<T2, SegmentManager> &other)
: mp_mngr(other.get_segment_manager()){}

pointer allocate(size_type count, cvoid_ptr hint = 0)
{
(void)hint;
if(size_overflows<sizeof(T)>(count)){
throw bad_alloc();
}
return pointer(static_cast<value_type*>(mp_mngr->allocate(count*sizeof(T))));
}

void deallocate(const pointer &ptr, size_type)
{  mp_mngr->deallocate((void*)ipcdetail::to_raw_pointer(ptr));  }

size_type max_size() const
{  return mp_mngr->get_size()/sizeof(T);   }

friend void swap(self_t &alloc1, self_t &alloc2)
{  boost::adl_move_swap(alloc1.mp_mngr, alloc2.mp_mngr);   }

size_type size(const pointer &p) const
{
return (size_type)mp_mngr->size(ipcdetail::to_raw_pointer(p))/sizeof(T);
}

pointer allocation_command(boost::interprocess::allocation_type command,
size_type limit_size, size_type &prefer_in_recvd_out_size, pointer &reuse)
{
value_type *reuse_raw = ipcdetail::to_raw_pointer(reuse);
pointer const p = mp_mngr->allocation_command(command, limit_size, prefer_in_recvd_out_size, reuse_raw);
reuse = reuse_raw;
return p;
}

void allocate_many(size_type elem_size, size_type num_elements, multiallocation_chain &chain)
{
if(size_overflows<sizeof(T)>(elem_size)){
throw bad_alloc();
}
mp_mngr->allocate_many(elem_size*sizeof(T), num_elements, chain);
}

void allocate_many(const size_type *elem_sizes, size_type n_elements, multiallocation_chain &chain)
{
mp_mngr->allocate_many(elem_sizes, n_elements, sizeof(T), chain);
}

void deallocate_many(multiallocation_chain &chain)
{  mp_mngr->deallocate_many(chain); }

pointer allocate_one()
{  return this->allocate(1);  }

void allocate_individual(size_type num_elements, multiallocation_chain &chain)
{  this->allocate_many(1, num_elements, chain); }

void deallocate_one(const pointer &p)
{  return this->deallocate(p, 1);  }

void deallocate_individual(multiallocation_chain &chain)
{  this->deallocate_many(chain); }

pointer address(reference value) const
{  return pointer(boost::addressof(value));  }

const_pointer address(const_reference value) const
{  return const_pointer(boost::addressof(value));  }

template<class P>
void construct(const pointer &ptr, BOOST_FWD_REF(P) p)
{  ::new((void*)ipcdetail::to_raw_pointer(ptr), boost_container_new_t()) value_type(::boost::forward<P>(p));  }

void destroy(const pointer &ptr)
{  BOOST_ASSERT(ptr != 0); (*ptr).~value_type();  }

};

template<class T, class SegmentManager> inline
bool operator==(const allocator<T , SegmentManager>  &alloc1,
const allocator<T, SegmentManager>  &alloc2)
{  return alloc1.get_segment_manager() == alloc2.get_segment_manager(); }

template<class T, class SegmentManager> inline
bool operator!=(const allocator<T, SegmentManager>  &alloc1,
const allocator<T, SegmentManager>  &alloc2)
{  return alloc1.get_segment_manager() != alloc2.get_segment_manager(); }

}  

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template<class T>
struct has_trivial_destructor;

template<class T, class SegmentManager>
struct has_trivial_destructor
<boost::interprocess::allocator <T, SegmentManager> >
{
static const bool value = true;
};
#endif   

}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

