#ifndef BOOST_SHARED_PTR_132_HPP_INCLUDED
#define BOOST_SHARED_PTR_132_HPP_INCLUDED


#include <boost/config.hpp>   

#if defined(BOOST_NO_MEMBER_TEMPLATES) && !defined(BOOST_MSVC6_MEMBER_TEMPLATES)
#include <boost/serialization/detail/shared_ptr_nmt_132.hpp>
#else

#include <boost/assert.hpp>
#include <boost/checked_delete.hpp>
#include <boost/serialization/throw_exception.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/detail/shared_count_132.hpp>

#include <memory>               
#include <algorithm>            
#include <functional>           
#include <typeinfo>             
#include <iosfwd>               

#ifdef BOOST_MSVC  
# pragma warning(push)
# pragma warning(disable:4284) 
#endif

namespace boost_132 {

template<class T> class weak_ptr;
template<class T> class enable_shared_from_this;

namespace detail
{

struct static_cast_tag {};
struct const_cast_tag {};
struct dynamic_cast_tag {};
struct polymorphic_cast_tag {};

template<class T> struct shared_ptr_traits
{
typedef T & reference;
};

template<> struct shared_ptr_traits<void>
{
typedef void reference;
};

#if !defined(BOOST_NO_CV_VOID_SPECIALIZATIONS)

template<> struct shared_ptr_traits<void const>
{
typedef void reference;
};

template<> struct shared_ptr_traits<void volatile>
{
typedef void reference;
};

template<> struct shared_ptr_traits<void const volatile>
{
typedef void reference;
};

#endif


template<class T, class Y> void sp_enable_shared_from_this( shared_count const & pn, enable_shared_from_this< T > const * pe, Y const * px )
{
if(pe != 0) pe->_internal_weak_this._internal_assign(const_cast<Y*>(px), pn);
}

inline void sp_enable_shared_from_this( shared_count const & , ... )
{
}

} 



template<class T> class shared_ptr
{
private:
typedef shared_ptr< T > this_type;

public:

typedef T element_type;
typedef T value_type;
typedef T * pointer;
typedef typename detail::shared_ptr_traits< T >::reference reference;

shared_ptr(): px(0), pn() 
{
}

template<class Y>
explicit shared_ptr(Y * p): px(p), pn(p, boost::checked_deleter<Y>()) 
{
detail::sp_enable_shared_from_this( pn, p, p );
}


template<class Y, class D> shared_ptr(Y * p, D d): px(p), pn(p, d)
{
detail::sp_enable_shared_from_this( pn, p, p );
}


#if defined(__GNUC__)
shared_ptr & operator=(shared_ptr const & r) 
{
px = r.px;
pn = r.pn; 
return *this;
}
#endif

template<class Y>
explicit shared_ptr(weak_ptr<Y> const & r): pn(r.pn) 
{
px = r.px;
}

template<class Y>
shared_ptr(shared_ptr<Y> const & r): px(r.px), pn(r.pn) 
{
}

template<class Y>
shared_ptr(shared_ptr<Y> const & r, detail::static_cast_tag): px(static_cast<element_type *>(r.px)), pn(r.pn)
{
}

template<class Y>
shared_ptr(shared_ptr<Y> const & r, detail::const_cast_tag): px(const_cast<element_type *>(r.px)), pn(r.pn)
{
}

template<class Y>
shared_ptr(shared_ptr<Y> const & r, detail::dynamic_cast_tag): px(dynamic_cast<element_type *>(r.px)), pn(r.pn)
{
if(px == 0) 
{
pn = detail::shared_count();
}
}

template<class Y>
shared_ptr(shared_ptr<Y> const & r, detail::polymorphic_cast_tag): px(dynamic_cast<element_type *>(r.px)), pn(r.pn)
{
if(px == 0)
{
boost::serialization::throw_exception(std::bad_cast());
}
}

#ifndef BOOST_NO_AUTO_PTR

template<class Y>
explicit shared_ptr(std::auto_ptr<Y> & r): px(r.get()), pn()
{
Y * tmp = r.get();
pn = detail::shared_count(r);
detail::sp_enable_shared_from_this( pn, tmp, tmp );
}

#endif

#if !defined(BOOST_MSVC) || (BOOST_MSVC > 1200)

template<class Y>
shared_ptr & operator=(shared_ptr<Y> const & r) 
{
px = r.px;
pn = r.pn; 
return *this;
}

#endif

#ifndef BOOST_NO_AUTO_PTR

template<class Y>
shared_ptr & operator=(std::auto_ptr<Y> & r)
{
this_type(r).swap(*this);
return *this;
}

#endif

void reset() 
{
this_type().swap(*this);
}

template<class Y> void reset(Y * p) 
{
BOOST_ASSERT(p == 0 || p != px); 
this_type(p).swap(*this);
}

template<class Y, class D> void reset(Y * p, D d)
{
this_type(p, d).swap(*this);
}

reference operator* () const 
{
BOOST_ASSERT(px != 0);
return *px;
}

T * operator-> () const 
{
BOOST_ASSERT(px != 0);
return px;
}

T * get() const 
{
return px;
}


#if defined(__SUNPRO_CC) && BOOST_WORKAROUND(__SUNPRO_CC, <= 0x530)

operator bool () const
{
return px != 0;
}

#elif defined(__MWERKS__) && BOOST_WORKAROUND(__MWERKS__, BOOST_TESTED_AT(0x3003))
typedef T * (this_type::*unspecified_bool_type)() const;

operator unspecified_bool_type() const 
{
return px == 0? 0: &this_type::get;
}

#else

typedef T * this_type::*unspecified_bool_type;

operator unspecified_bool_type() const 
{
return px == 0? 0: &this_type::px;
}

#endif


bool operator! () const 
{
return px == 0;
}

bool unique() const 
{
return pn.unique();
}

long use_count() const 
{
return pn.use_count();
}

void swap(shared_ptr< T > & other) 
{
std::swap(px, other.px);
pn.swap(other.pn);
}

template<class Y> bool _internal_less(shared_ptr<Y> const & rhs) const
{
return pn < rhs.pn;
}

void * _internal_get_deleter(std::type_info const & ti) const
{
return pn.get_deleter(ti);
}


#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS

private:

template<class Y> friend class shared_ptr;
template<class Y> friend class weak_ptr;


#endif
public: 
T * px;                     
detail::shared_count pn;    

};  

template<class T, class U> inline bool operator==(shared_ptr< T > const & a, shared_ptr<U> const & b)
{
return a.get() == b.get();
}

template<class T, class U> inline bool operator!=(shared_ptr< T > const & a, shared_ptr<U> const & b)
{
return a.get() != b.get();
}

template<class T, class U> inline bool operator<(shared_ptr< T > const & a, shared_ptr<U> const & b)
{
return a._internal_less(b);
}

template<class T> inline void swap(shared_ptr< T > & a, shared_ptr< T > & b)
{
a.swap(b);
}

template<class T, class U> shared_ptr< T > static_pointer_cast(shared_ptr<U> const & r)
{
return shared_ptr< T >(r, detail::static_cast_tag());
}

template<class T, class U> shared_ptr< T > const_pointer_cast(shared_ptr<U> const & r)
{
return shared_ptr< T >(r, detail::const_cast_tag());
}

template<class T, class U> shared_ptr< T > dynamic_pointer_cast(shared_ptr<U> const & r)
{
return shared_ptr< T >(r, detail::dynamic_cast_tag());
}


template<class T, class U> shared_ptr< T > shared_static_cast(shared_ptr<U> const & r)
{
return shared_ptr< T >(r, detail::static_cast_tag());
}

template<class T, class U> shared_ptr< T > shared_dynamic_cast(shared_ptr<U> const & r)
{
return shared_ptr< T >(r, detail::dynamic_cast_tag());
}

template<class T, class U> shared_ptr< T > shared_polymorphic_cast(shared_ptr<U> const & r)
{
return shared_ptr< T >(r, detail::polymorphic_cast_tag());
}

template<class T, class U> shared_ptr< T > shared_polymorphic_downcast(shared_ptr<U> const & r)
{
BOOST_ASSERT(dynamic_cast<T *>(r.get()) == r.get());
return shared_static_cast< T >(r);
}


template<class T> inline T * get_pointer(shared_ptr< T > const & p)
{
return p.get();
}



template<class E, class T, class Y> std::basic_ostream<E, T> & operator<< (std::basic_ostream<E, T> & os, shared_ptr<Y> const & p)
{
os << p.get();
return os;
}


#if defined(__EDG_VERSION__) && (__EDG_VERSION__ <= 238)


template<class D, class T> D * get_deleter(shared_ptr< T > const & p)
{
void const * q = p._internal_get_deleter(typeid(D));
return const_cast<D *>(static_cast<D const *>(q));
}

#else

template<class D, class T> D * get_deleter(shared_ptr< T > const & p)
{
return static_cast<D *>(p._internal_get_deleter(typeid(D)));
}

#endif

} 

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif  

#endif  
