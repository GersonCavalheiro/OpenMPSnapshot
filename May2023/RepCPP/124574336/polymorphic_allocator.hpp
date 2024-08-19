
#ifndef BOOST_CONTAINER_PMR_POLYMORPHIC_ALLOCATOR_HPP
#define BOOST_CONTAINER_PMR_POLYMORPHIC_ALLOCATOR_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/config.hpp>
#include <boost/move/detail/type_traits.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/container/detail/dispatch_uses_allocator.hpp>
#include <boost/container/new_allocator.hpp>
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/global_resource.hpp>

#include <cstddef>

namespace boost {
namespace container {
namespace pmr {

template <class T>
class polymorphic_allocator
{
public:
typedef T value_type;

polymorphic_allocator() BOOST_NOEXCEPT
: m_resource(::boost::container::pmr::get_default_resource())
{}

polymorphic_allocator(memory_resource* r)
: m_resource(r)
{  BOOST_ASSERT(r != 0);  }

polymorphic_allocator(const polymorphic_allocator& other)
: m_resource(other.m_resource)
{}

template <class U>
polymorphic_allocator(const polymorphic_allocator<U>& other) BOOST_NOEXCEPT
: m_resource(other.resource())
{}

polymorphic_allocator& operator=(const polymorphic_allocator& other)
{  m_resource = other.m_resource;   return *this;  }

T* allocate(size_t n)
{  return static_cast<T*>(m_resource->allocate(n*sizeof(T), ::boost::move_detail::alignment_of<T>::value));  }

void deallocate(T* p, size_t n)
{  m_resource->deallocate(p, n*sizeof(T), ::boost::move_detail::alignment_of<T>::value);  }

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
template < typename U, class ...Args>
void construct(U* p, BOOST_FWD_REF(Args)...args)
{
new_allocator<U> na;
dtl::dispatch_uses_allocator
(na, *this, p, ::boost::forward<Args>(args)...);
}

#else 

#define BOOST_CONTAINER_PMR_POLYMORPHIC_ALLOCATOR_CONSTRUCT_CODE(N) \
template < typename U BOOST_MOVE_I##N BOOST_MOVE_CLASSQ##N >\
void construct(U* p BOOST_MOVE_I##N BOOST_MOVE_UREFQ##N)\
{\
new_allocator<U> na;\
dtl::dispatch_uses_allocator\
(na, *this, p BOOST_MOVE_I##N BOOST_MOVE_FWDQ##N);\
}\
BOOST_MOVE_ITERATE_0TO9(BOOST_CONTAINER_PMR_POLYMORPHIC_ALLOCATOR_CONSTRUCT_CODE)
#undef BOOST_CONTAINER_PMR_POLYMORPHIC_ALLOCATOR_CONSTRUCT_CODE

#endif   

template <class U>
void destroy(U* p)
{  (void)p; p->~U(); }

polymorphic_allocator select_on_container_copy_construction() const
{  return polymorphic_allocator();  }

memory_resource* resource() const
{  return m_resource;  }

private:
memory_resource* m_resource;
};

template <class T1, class T2>
bool operator==(const polymorphic_allocator<T1>& a, const polymorphic_allocator<T2>& b) BOOST_NOEXCEPT
{  return *a.resource() == *b.resource();  }


template <class T1, class T2>
bool operator!=(const polymorphic_allocator<T1>& a, const polymorphic_allocator<T2>& b) BOOST_NOEXCEPT
{  return *a.resource() != *b.resource();  }

}  
}  
}  

#endif   
