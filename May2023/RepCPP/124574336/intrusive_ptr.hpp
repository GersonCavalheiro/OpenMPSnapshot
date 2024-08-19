
#ifndef BOOST_INTERPROCESS_INTRUSIVE_PTR_HPP_INCLUDED
#define BOOST_INTERPROCESS_INTRUSIVE_PTR_HPP_INCLUDED

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif


#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/assert.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/move/adl_move_swap.hpp>
#include <boost/move/core.hpp>

#include <iosfwd>               

#include <boost/intrusive/detail/minimal_less_equal_header.hpp>   

namespace boost {
namespace interprocess {

template<class T, class VoidPointer>
class intrusive_ptr
{
public:
typedef typename boost::intrusive::
pointer_traits<VoidPointer>::template
rebind_pointer<T>::type                pointer;
typedef T element_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef VoidPointer VP;
typedef intrusive_ptr this_type;
typedef pointer this_type::*unspecified_bool_type;
#endif   

BOOST_COPYABLE_AND_MOVABLE(intrusive_ptr)

public:
intrusive_ptr() BOOST_NOEXCEPT
: m_ptr(0)
{}

intrusive_ptr(const pointer &p, bool add_ref = true) BOOST_NOEXCEPT
: m_ptr(p)
{
if(m_ptr != 0 && add_ref) intrusive_ptr_add_ref(ipcdetail::to_raw_pointer(m_ptr));
}

intrusive_ptr(intrusive_ptr const & rhs) BOOST_NOEXCEPT
:  m_ptr(rhs.m_ptr)
{
if(m_ptr != 0) intrusive_ptr_add_ref(ipcdetail::to_raw_pointer(m_ptr));
}

intrusive_ptr(BOOST_RV_REF(intrusive_ptr) rhs) BOOST_NOEXCEPT
: m_ptr(rhs.m_ptr) 
{
rhs.m_ptr = 0;
}

template<class U> intrusive_ptr(intrusive_ptr<U, VP> const & rhs) BOOST_NOEXCEPT
:  m_ptr(rhs.get())
{
if(m_ptr != 0) intrusive_ptr_add_ref(ipcdetail::to_raw_pointer(m_ptr));
}

~intrusive_ptr()
{
reset();
}

intrusive_ptr & operator=(BOOST_COPY_ASSIGN_REF(intrusive_ptr) rhs) BOOST_NOEXCEPT
{
this_type(rhs).swap(*this);
return *this;
}

intrusive_ptr & operator=(BOOST_RV_REF(intrusive_ptr) rhs) BOOST_NOEXCEPT 
{
rhs.swap(*this);
rhs.reset();
return *this;
}

template<class U> intrusive_ptr & operator=(intrusive_ptr<U, VP> const & rhs) BOOST_NOEXCEPT
{
this_type(rhs).swap(*this);
return *this;
}

intrusive_ptr & operator=(pointer rhs) BOOST_NOEXCEPT
{
this_type(rhs).swap(*this);
return *this;
}

void reset() BOOST_NOEXCEPT {
if(m_ptr != 0) {
pointer ptr = m_ptr;
m_ptr = 0;
intrusive_ptr_release(ipcdetail::to_raw_pointer(ptr));
}
}

pointer &get() BOOST_NOEXCEPT
{  return m_ptr;  }

const pointer &get() const BOOST_NOEXCEPT
{  return m_ptr;  }

T & operator*() const BOOST_NOEXCEPT
{  return *m_ptr; }

const pointer &operator->() const BOOST_NOEXCEPT
{  return m_ptr;  }

pointer &operator->() BOOST_NOEXCEPT
{  return m_ptr;  }

operator unspecified_bool_type () const BOOST_NOEXCEPT
{  return m_ptr == 0? 0: &this_type::m_ptr;  }

bool operator! () const BOOST_NOEXCEPT
{  return m_ptr == 0;   }

void swap(intrusive_ptr & rhs) BOOST_NOEXCEPT
{  ::boost::adl_move_swap(m_ptr, rhs.m_ptr);  }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
pointer m_ptr;
#endif   
};

template<class T, class U, class VP> inline
bool operator==(intrusive_ptr<T, VP> const & a,
intrusive_ptr<U, VP> const & b) BOOST_NOEXCEPT
{  return a.get() == b.get(); }

template<class T, class U, class VP> inline
bool operator!=(intrusive_ptr<T, VP> const & a,
intrusive_ptr<U, VP> const & b) BOOST_NOEXCEPT
{  return a.get() != b.get(); }

template<class T, class VP> inline
bool operator==(intrusive_ptr<T, VP> const & a,
const typename intrusive_ptr<T, VP>::pointer &b) BOOST_NOEXCEPT
{  return a.get() == b; }

template<class T, class VP> inline
bool operator!=(intrusive_ptr<T, VP> const & a,
const typename intrusive_ptr<T, VP>::pointer &b) BOOST_NOEXCEPT
{  return a.get() != b; }

template<class T, class VP> inline
bool operator==(const typename intrusive_ptr<T, VP>::pointer &a,
intrusive_ptr<T, VP> const & b) BOOST_NOEXCEPT
{  return a == b.get(); }

template<class T, class VP> inline
bool operator!=(const typename intrusive_ptr<T, VP>::pointer &a,
intrusive_ptr<T, VP> const & b) BOOST_NOEXCEPT
{  return a != b.get(); }

template<class T, class VP> inline
bool operator<(intrusive_ptr<T, VP> const & a,
intrusive_ptr<T, VP> const & b) BOOST_NOEXCEPT
{
return std::less<typename intrusive_ptr<T, VP>::pointer>()
(a.get(), b.get());
}

template<class T, class VP> inline
void swap(intrusive_ptr<T, VP> & lhs,
intrusive_ptr<T, VP> & rhs) BOOST_NOEXCEPT
{  lhs.swap(rhs); }

template<class E, class T, class Y, class VP>
inline std::basic_ostream<E, T> & operator<<
(std::basic_ostream<E, T> & os, intrusive_ptr<Y, VP> const & p) BOOST_NOEXCEPT
{  os << p.get(); return os;  }

template<class T, class VP>
inline typename boost::interprocess::intrusive_ptr<T, VP>::pointer
to_raw_pointer(intrusive_ptr<T, VP> p) BOOST_NOEXCEPT
{  return p.get();   }












} 

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#if defined(_MSC_VER) && (_MSC_VER < 1400)
template<class T, class VP>
inline T *to_raw_pointer(boost::interprocess::intrusive_ptr<T, VP> p) BOOST_NOEXCEPT
{  return p.get();   }
#endif

#endif   

} 

#include <boost/interprocess/detail/config_end.hpp>

#endif  
