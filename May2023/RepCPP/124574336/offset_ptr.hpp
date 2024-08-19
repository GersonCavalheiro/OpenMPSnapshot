
#ifndef BOOST_INTERPROCESS_OFFSET_PTR_HPP
#define BOOST_INTERPROCESS_OFFSET_PTR_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_constructible.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_unsigned.hpp>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/detail/cast_tags.hpp>
#include <boost/interprocess/detail/mpl.hpp>
#include <boost/container/detail/type_traits.hpp>  
#include <boost/assert.hpp>
#include <iosfwd>


namespace boost {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <class T>
struct has_trivial_destructor;

#endif   

namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
namespace ipcdetail {

template<class OffsetType, std::size_t OffsetAlignment>
union offset_ptr_internal
{
BOOST_STATIC_ASSERT(sizeof(OffsetType) >= sizeof(uintptr_t));
BOOST_STATIC_ASSERT(boost::is_integral<OffsetType>::value && boost::is_unsigned<OffsetType>::value);

explicit offset_ptr_internal(OffsetType off)
: m_offset(off)
{}

OffsetType m_offset; 

typename ::boost::container::dtl::aligned_storage
< sizeof(OffsetType)
, (OffsetAlignment == offset_type_alignment) ? 1u : OffsetAlignment
>::type alignment_helper;
};



#define BOOST_INTERPROCESS_OFFSET_PTR_BRANCHLESS_TO_PTR
template <class OffsetType>
BOOST_INTERPROCESS_FORCEINLINE void * offset_ptr_to_raw_pointer(const volatile void *this_ptr, OffsetType offset)
{
typedef pointer_offset_caster<void*, OffsetType> caster_t;
#ifndef BOOST_INTERPROCESS_OFFSET_PTR_BRANCHLESS_TO_PTR
if(offset == 1){
return 0;
}
else{
return caster_t(caster_t(this_ptr).offset() + offset).pointer();
}
#else
OffsetType mask = offset == 1;
--mask;
OffsetType target_offset = caster_t(this_ptr).offset() + offset;
target_offset &= mask;
return caster_t(target_offset).pointer();
#endif
}

#define BOOST_INTERPROCESS_OFFSET_PTR_BRANCHLESS_TO_OFF
template<class OffsetType>
BOOST_INTERPROCESS_FORCEINLINE OffsetType offset_ptr_to_offset(const volatile void *ptr, const volatile void *this_ptr)
{
typedef pointer_offset_caster<void*, OffsetType> caster_t;
#ifndef BOOST_INTERPROCESS_OFFSET_PTR_BRANCHLESS_TO_OFF
if(!ptr){
return 1;
}
else{
OffsetType offset = caster_t(ptr).offset()- caster_t(this_ptr).offset();
BOOST_ASSERT(offset != 1);
return offset;
}
#else
OffsetType offset = caster_t(ptr).offset() - caster_t(this_ptr).offset();
--offset;
OffsetType mask = ptr == 0;
--mask;
offset &= mask;
return ++offset;
#endif
}

#define BOOST_INTERPROCESS_OFFSET_PTR_BRANCHLESS_TO_OFF_FROM_OTHER
template<class OffsetType>
BOOST_INTERPROCESS_FORCEINLINE OffsetType offset_ptr_to_offset_from_other
(const volatile void *this_ptr, const volatile void *other_ptr, OffsetType other_offset)
{
typedef pointer_offset_caster<void*, OffsetType> caster_t;
#ifndef BOOST_INTERPROCESS_OFFSET_PTR_BRANCHLESS_TO_OFF_FROM_OTHER
if(other_offset == 1){
return 1;
}
else{
OffsetType offset = caster_t(other_ptr).offset() - caster_t(this_ptr).offset() + other_offset;
BOOST_ASSERT(offset != 1);
return offset;
}
#else
OffsetType mask = other_offset == 1;
--mask;
OffsetType offset = caster_t(other_ptr).offset() - caster_t(this_ptr).offset();
offset &= mask;
return offset + other_offset;

#endif
}

template<class From, class To>
struct offset_ptr_maintains_address
{
static const bool value =    ipcdetail::is_cv_same<From, To>::value
|| ipcdetail::is_cv_same<void, To>::value
|| ipcdetail::is_cv_same<char, To>::value
;
};

template<class From, class To, class Ret = void>
struct enable_if_convertible_equal_address
: enable_if_c< ::boost::is_convertible<From*, To*>::value
&& offset_ptr_maintains_address<From, To>::value
, Ret>
{};

template<class From, class To, class Ret = void>
struct enable_if_convertible_unequal_address
: enable_if_c< ::boost::is_convertible<From*, To*>::value
&& !offset_ptr_maintains_address<From, To>::value
, Ret>
{};

}  
#endif   

template <class PointedType, class DifferenceType, class OffsetType, std::size_t OffsetAlignment>
class offset_ptr
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef offset_ptr<PointedType, DifferenceType, OffsetType, OffsetAlignment>   self_t;
void unspecified_bool_type_func() const {}
typedef void (self_t::*unspecified_bool_type)() const;
#endif   

public:
typedef PointedType                       element_type;
typedef PointedType *                     pointer;
typedef typename ipcdetail::
add_reference<PointedType>::type       reference;
typedef typename ipcdetail::
remove_volatile<typename ipcdetail::
remove_const<PointedType>::type
>::type                          value_type;
typedef DifferenceType                    difference_type;
typedef std::random_access_iterator_tag   iterator_category;
typedef OffsetType                        offset_type;

public:   

BOOST_INTERPROCESS_FORCEINLINE offset_ptr() BOOST_NOEXCEPT
: internal(1)
{}

BOOST_INTERPROCESS_FORCEINLINE offset_ptr(pointer ptr) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(ptr, this))
{}

template <class T>
BOOST_INTERPROCESS_FORCEINLINE offset_ptr( T *ptr
, typename ipcdetail::enable_if< ::boost::is_convertible<T*, PointedType*> >::type * = 0) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(static_cast<PointedType*>(ptr), this))
{}

BOOST_INTERPROCESS_FORCEINLINE offset_ptr(const offset_ptr& ptr) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset_from_other(this, &ptr, ptr.internal.m_offset))
{}

template<class T2>
BOOST_INTERPROCESS_FORCEINLINE offset_ptr( const offset_ptr<T2, DifferenceType, OffsetType, OffsetAlignment> &ptr
#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED
, typename ipcdetail::enable_if_convertible_equal_address<T2, PointedType>::type* = 0
#endif
) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset_from_other(this, &ptr, ptr.get_offset()))
{}

#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED

template<class T2>
BOOST_INTERPROCESS_FORCEINLINE offset_ptr( const offset_ptr<T2, DifferenceType, OffsetType, OffsetAlignment> &ptr
, typename ipcdetail::enable_if_convertible_unequal_address<T2, PointedType>::type* = 0) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(static_cast<PointedType*>(ptr.get()), this))
{}

#endif

template<class T2>
BOOST_INTERPROCESS_FORCEINLINE explicit offset_ptr(const offset_ptr<T2, DifferenceType, OffsetType, OffsetAlignment> &ptr
#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED
, typename ipcdetail::enable_if_c<
!::boost::is_convertible<T2*, PointedType*>::value && ::boost::is_constructible<T2*, PointedType*>::value
>::type * = 0
#endif
) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(static_cast<PointedType*>(ptr.get()), this))
{}

template<class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE offset_ptr(const offset_ptr<T2, P2, O2, A2> & r, ipcdetail::static_cast_tag) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(static_cast<PointedType*>(r.get()), this))
{}

template<class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE offset_ptr(const offset_ptr<T2, P2, O2, A2> & r, ipcdetail::const_cast_tag) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(const_cast<PointedType*>(r.get()), this))
{}

template<class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE offset_ptr(const offset_ptr<T2, P2, O2, A2> & r, ipcdetail::dynamic_cast_tag) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(dynamic_cast<PointedType*>(r.get()), this))
{}

template<class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE offset_ptr(const offset_ptr<T2, P2, O2, A2> & r, ipcdetail::reinterpret_cast_tag) BOOST_NOEXCEPT
: internal(ipcdetail::offset_ptr_to_offset<OffsetType>(reinterpret_cast<PointedType*>(r.get()), this))
{}

BOOST_INTERPROCESS_FORCEINLINE pointer get() const BOOST_NOEXCEPT
{  return static_cast<pointer>(ipcdetail::offset_ptr_to_raw_pointer(this, this->internal.m_offset));   }

BOOST_INTERPROCESS_FORCEINLINE offset_type get_offset() const BOOST_NOEXCEPT
{  return this->internal.m_offset;  }

BOOST_INTERPROCESS_FORCEINLINE pointer operator->() const BOOST_NOEXCEPT
{  return this->get(); }

BOOST_INTERPROCESS_FORCEINLINE reference operator* () const BOOST_NOEXCEPT
{
pointer p = this->get();
reference r = *p;
return r;
}

BOOST_INTERPROCESS_FORCEINLINE reference operator[](difference_type idx) const BOOST_NOEXCEPT
{  return this->get()[idx];  }

BOOST_INTERPROCESS_FORCEINLINE offset_ptr& operator= (pointer from) BOOST_NOEXCEPT
{
this->internal.m_offset = ipcdetail::offset_ptr_to_offset<OffsetType>(from, this);
return *this;
}

BOOST_INTERPROCESS_FORCEINLINE offset_ptr& operator= (const offset_ptr & ptr) BOOST_NOEXCEPT
{
this->internal.m_offset = ipcdetail::offset_ptr_to_offset_from_other(this, &ptr, ptr.internal.m_offset);
return *this;
}

template<class T2> BOOST_INTERPROCESS_FORCEINLINE 
#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED
typename ipcdetail::enable_if_c
< ::boost::is_convertible<T2*, PointedType*>::value, offset_ptr&>::type
#else
offset_ptr&
#endif
operator= (const offset_ptr<T2, DifferenceType, OffsetType, OffsetAlignment> &ptr) BOOST_NOEXCEPT
{
this->assign(ptr, ipcdetail::bool_<ipcdetail::offset_ptr_maintains_address<T2, PointedType>::value>());
return *this;
}

public:

BOOST_INTERPROCESS_FORCEINLINE offset_ptr &operator+= (difference_type offset) BOOST_NOEXCEPT
{  this->inc_offset(offset * sizeof (PointedType));   return *this;  }

BOOST_INTERPROCESS_FORCEINLINE offset_ptr &operator-= (difference_type offset) BOOST_NOEXCEPT
{  this->dec_offset(offset * sizeof (PointedType));   return *this;  }

BOOST_INTERPROCESS_FORCEINLINE offset_ptr& operator++ (void) BOOST_NOEXCEPT
{  this->inc_offset(sizeof (PointedType));   return *this;  }

BOOST_INTERPROCESS_FORCEINLINE offset_ptr operator++ (int) BOOST_NOEXCEPT
{
offset_ptr tmp(*this);
this->inc_offset(sizeof (PointedType));
return tmp;
}

BOOST_INTERPROCESS_FORCEINLINE offset_ptr& operator-- (void) BOOST_NOEXCEPT
{  this->dec_offset(sizeof (PointedType));   return *this;  }

BOOST_INTERPROCESS_FORCEINLINE offset_ptr operator-- (int) BOOST_NOEXCEPT
{
offset_ptr tmp(*this);
this->dec_offset(sizeof (PointedType));
return tmp;
}

#if defined(BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS)
BOOST_INTERPROCESS_FORCEINLINE operator unspecified_bool_type() const BOOST_NOEXCEPT
{  return this->internal.m_offset != 1? &self_t::unspecified_bool_type_func : 0;   }
#else
explicit operator bool() const BOOST_NOEXCEPT
{  return this->internal.m_offset != 1;  }
#endif

BOOST_INTERPROCESS_FORCEINLINE bool operator! () const BOOST_NOEXCEPT
{  return this->internal.m_offset == 1;   }

#if defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
template <class U>
struct rebind
{  typedef offset_ptr<U, DifferenceType, OffsetType, OffsetAlignment> other;  };
#else
template <class U>
using rebind = offset_ptr<U, DifferenceType, OffsetType, OffsetAlignment>;
#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED
typedef offset_ptr<PointedType, DifferenceType, OffsetType, OffsetAlignment> other;
#endif 
#endif

BOOST_INTERPROCESS_FORCEINLINE static offset_ptr pointer_to(reference r) BOOST_NOEXCEPT
{ return offset_ptr(&r); }

BOOST_INTERPROCESS_FORCEINLINE friend offset_ptr operator+(difference_type diff, offset_ptr right) BOOST_NOEXCEPT
{  right += diff;  return right;  }

BOOST_INTERPROCESS_FORCEINLINE friend offset_ptr operator+(offset_ptr left, difference_type diff) BOOST_NOEXCEPT
{  left += diff;  return left; }

BOOST_INTERPROCESS_FORCEINLINE friend offset_ptr operator-(offset_ptr left, difference_type diff) BOOST_NOEXCEPT
{  left -= diff;  return left; }

BOOST_INTERPROCESS_FORCEINLINE friend offset_ptr operator-(difference_type diff, offset_ptr right) BOOST_NOEXCEPT
{  right -= diff; return right; }

BOOST_INTERPROCESS_FORCEINLINE friend difference_type operator-(const offset_ptr &pt, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return difference_type(pt.get()- pt2.get());   }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator== (const offset_ptr &pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1.get() == pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator!= (const offset_ptr &pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1.get() != pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator<(const offset_ptr &pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1.get() < pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator<=(const offset_ptr &pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1.get() <= pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator>(const offset_ptr &pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1.get() > pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator>=(const offset_ptr &pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1.get() >= pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator== (pointer pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1 == pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator!= (pointer pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1 != pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator<(pointer pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1 < pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator<=(pointer pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1 <= pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator>(pointer pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1 > pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator>=(pointer pt1, const offset_ptr &pt2) BOOST_NOEXCEPT
{  return pt1 >= pt2.get();  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator== (const offset_ptr &pt1, pointer pt2) BOOST_NOEXCEPT
{  return pt1.get() == pt2;  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator!= (const offset_ptr &pt1, pointer pt2) BOOST_NOEXCEPT
{  return pt1.get() != pt2;  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator<(const offset_ptr &pt1, pointer pt2) BOOST_NOEXCEPT
{  return pt1.get() < pt2;  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator<=(const offset_ptr &pt1, pointer pt2) BOOST_NOEXCEPT
{  return pt1.get() <= pt2;  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator>(const offset_ptr &pt1, pointer pt2) BOOST_NOEXCEPT
{  return pt1.get() > pt2;  }

BOOST_INTERPROCESS_FORCEINLINE friend bool operator>=(const offset_ptr &pt1, pointer pt2) BOOST_NOEXCEPT
{  return pt1.get() >= pt2;  }

BOOST_INTERPROCESS_FORCEINLINE friend void swap(offset_ptr &left, offset_ptr &right) BOOST_NOEXCEPT
{
pointer ptr = right.get();
right = left;
left = ptr;
}

private:
template<class T2>
BOOST_INTERPROCESS_FORCEINLINE void assign(const offset_ptr<T2, DifferenceType, OffsetType, OffsetAlignment> &ptr, ipcdetail::bool_<true>) BOOST_NOEXCEPT
{  
this->internal.m_offset = ipcdetail::offset_ptr_to_offset_from_other<OffsetType>(this, &ptr, ptr.get_offset());
}

template<class T2>
BOOST_INTERPROCESS_FORCEINLINE void assign(const offset_ptr<T2, DifferenceType, OffsetType, OffsetAlignment> &ptr, ipcdetail::bool_<false>) BOOST_NOEXCEPT
{  
this->internal.m_offset = ipcdetail::offset_ptr_to_offset<OffsetType>(static_cast<PointedType*>(ptr.get()), this);
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
BOOST_INTERPROCESS_FORCEINLINE void inc_offset(DifferenceType bytes) BOOST_NOEXCEPT
{  internal.m_offset += bytes;   }

BOOST_INTERPROCESS_FORCEINLINE void dec_offset(DifferenceType bytes) BOOST_NOEXCEPT
{  internal.m_offset -= bytes;   }

ipcdetail::offset_ptr_internal<OffsetType, OffsetAlignment> internal;

public:
BOOST_INTERPROCESS_FORCEINLINE const OffsetType &priv_offset() const BOOST_NOEXCEPT
{  return internal.m_offset;   }

BOOST_INTERPROCESS_FORCEINLINE       OffsetType &priv_offset() BOOST_NOEXCEPT
{  return internal.m_offset;   }

#endif   
};

template<class E, class T, class W, class X, class Y, std::size_t Z>
inline std::basic_ostream<E, T> & operator<<
(std::basic_ostream<E, T> & os, offset_ptr<W, X, Y, Z> const & p)
{  return os << p.get_offset();   }

template<class E, class T, class W, class X, class Y, std::size_t Z>
inline std::basic_istream<E, T> & operator>>
(std::basic_istream<E, T> & is, offset_ptr<W, X, Y, Z> & p)
{  return is >> p.get_offset();  }

template<class T1, class P1, class O1, std::size_t A1, class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE boost::interprocess::offset_ptr<T1, P1, O1, A1>
static_pointer_cast(const boost::interprocess::offset_ptr<T2, P2, O2, A2> & r) BOOST_NOEXCEPT
{
return boost::interprocess::offset_ptr<T1, P1, O1, A1>
(r, boost::interprocess::ipcdetail::static_cast_tag());
}

template<class T1, class P1, class O1, std::size_t A1, class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE boost::interprocess::offset_ptr<T1, P1, O1, A1>
const_pointer_cast(const boost::interprocess::offset_ptr<T2, P2, O2, A2> & r) BOOST_NOEXCEPT
{
return boost::interprocess::offset_ptr<T1, P1, O1, A1>
(r, boost::interprocess::ipcdetail::const_cast_tag());
}

template<class T1, class P1, class O1, std::size_t A1, class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE boost::interprocess::offset_ptr<T1, P1, O1, A1>
dynamic_pointer_cast(const boost::interprocess::offset_ptr<T2, P2, O2, A2> & r) BOOST_NOEXCEPT
{
return boost::interprocess::offset_ptr<T1, P1, O1, A1>
(r, boost::interprocess::ipcdetail::dynamic_cast_tag());
}

template<class T1, class P1, class O1, std::size_t A1, class T2, class P2, class O2, std::size_t A2>
BOOST_INTERPROCESS_FORCEINLINE boost::interprocess::offset_ptr<T1, P1, O1, A1>
reinterpret_pointer_cast(const boost::interprocess::offset_ptr<T2, P2, O2, A2> & r) BOOST_NOEXCEPT
{
return boost::interprocess::offset_ptr<T1, P1, O1, A1>
(r, boost::interprocess::ipcdetail::reinterpret_cast_tag());
}

}  

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <class T, class P, class O, std::size_t A>
struct has_trivial_destructor< ::boost::interprocess::offset_ptr<T, P, O, A> >
{
static const bool value = true;
};

namespace move_detail {

template <class T, class P, class O, std::size_t A>
struct is_trivially_destructible< ::boost::interprocess::offset_ptr<T, P, O, A> >
{
static const bool value = true;
};

}  

namespace interprocess {

template <class T, class P, class O, std::size_t A>
BOOST_INTERPROCESS_FORCEINLINE T * to_raw_pointer(boost::interprocess::offset_ptr<T, P, O, A> const & p) BOOST_NOEXCEPT
{  return ipcdetail::to_raw_pointer(p);   }

}  


#endif   
}  

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace boost{

namespace intrusive {

template<class VoidPointer, std::size_t N>
struct max_pointer_plus_bits;

template<std::size_t OffsetAlignment, class P, class O, std::size_t A>
struct max_pointer_plus_bits<boost::interprocess::offset_ptr<void, P, O, A>, OffsetAlignment>
{
static const std::size_t value = ::boost::interprocess::ipcdetail::ls_zeros<OffsetAlignment>::value - 1;
};

template<class Pointer, std::size_t NumBits>
struct pointer_plus_bits;

template<class T, class P, class O, std::size_t A, std::size_t NumBits>
struct pointer_plus_bits<boost::interprocess::offset_ptr<T, P, O, A>, NumBits>
{
typedef boost::interprocess::offset_ptr<T, P, O, A>      pointer;
static const O Mask = ((static_cast<O>(1) << NumBits) - static_cast<O>(1)) << 1;
BOOST_STATIC_ASSERT(0 ==(Mask&1));


BOOST_INTERPROCESS_FORCEINLINE static pointer get_pointer(const pointer &n) BOOST_NOEXCEPT
{
pointer p;
O const tmp_off = n.priv_offset() & ~Mask;
p.priv_offset() = boost::interprocess::ipcdetail::offset_ptr_to_offset_from_other(&p, &n, tmp_off);
return p;
}

BOOST_INTERPROCESS_FORCEINLINE static void set_pointer(pointer &n, const pointer &p) BOOST_NOEXCEPT
{
BOOST_ASSERT(0 == (get_bits)(p));
O const stored_bits = n.priv_offset() & Mask;
n = p;
n.priv_offset() |= stored_bits;
}

BOOST_INTERPROCESS_FORCEINLINE static std::size_t get_bits(const pointer &n) BOOST_NOEXCEPT
{
return std::size_t((n.priv_offset() & Mask) >> 1u);
}

BOOST_INTERPROCESS_FORCEINLINE static void set_bits(pointer &n, std::size_t const b) BOOST_NOEXCEPT
{
BOOST_ASSERT(b < (std::size_t(1) << NumBits));
O tmp = n.priv_offset();
tmp &= ~Mask;
tmp |= O(b << 1u);
n.priv_offset() = tmp;
}
};

}  

template<class T, class U>
struct pointer_to_other;

template <class PointedType, class DifferenceType, class OffsetType, std::size_t OffsetAlignment, class U>
struct pointer_to_other
< ::boost::interprocess::offset_ptr<PointedType, DifferenceType, OffsetType, OffsetAlignment>, U >
{
typedef ::boost::interprocess::offset_ptr<U, DifferenceType, OffsetType, OffsetAlignment> type;
};

}  
#endif   

#include <boost/interprocess/detail/config_end.hpp>

#endif 
