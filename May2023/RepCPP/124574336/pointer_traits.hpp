
#ifndef BOOST_INTRUSIVE_POINTER_TRAITS_HPP
#define BOOST_INTRUSIVE_POINTER_TRAITS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/pointer_rebind.hpp>
#include <boost/move/detail/pointer_element.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <cstddef>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {
namespace detail {

#if !defined(BOOST_MSVC) || (BOOST_MSVC > 1310)
BOOST_INTRUSIVE_HAS_STATIC_MEMBER_FUNC_SIGNATURE(has_member_function_callable_with_pointer_to, pointer_to)
BOOST_INTRUSIVE_HAS_STATIC_MEMBER_FUNC_SIGNATURE(has_member_function_callable_with_dynamic_cast_from, dynamic_cast_from)
BOOST_INTRUSIVE_HAS_STATIC_MEMBER_FUNC_SIGNATURE(has_member_function_callable_with_static_cast_from, static_cast_from)
BOOST_INTRUSIVE_HAS_STATIC_MEMBER_FUNC_SIGNATURE(has_member_function_callable_with_const_cast_from, const_cast_from)
#else
BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED_IGNORE_SIGNATURE(has_member_function_callable_with_pointer_to, pointer_to)
BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED_IGNORE_SIGNATURE(has_member_function_callable_with_dynamic_cast_from, dynamic_cast_from)
BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED_IGNORE_SIGNATURE(has_member_function_callable_with_static_cast_from, static_cast_from)
BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED_IGNORE_SIGNATURE(has_member_function_callable_with_const_cast_from, const_cast_from)
#endif

BOOST_INTRUSIVE_INSTANTIATE_EVAL_DEFAULT_TYPE_TMPLT(element_type)
BOOST_INTRUSIVE_INSTANTIATE_DEFAULT_TYPE_TMPLT(difference_type)
BOOST_INTRUSIVE_INSTANTIATE_DEFAULT_TYPE_TMPLT(reference)
BOOST_INTRUSIVE_INSTANTIATE_DEFAULT_TYPE_TMPLT(value_traits_ptr)

}  


template <typename Ptr>
struct pointer_traits
{
#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
typedef Ptr             pointer;

typedef unspecified_type element_type;

typedef unspecified_type difference_type;

template <class U> using rebind = unspecified;

typedef unspecified_type reference;
#else
typedef Ptr                                                             pointer;
typedef BOOST_INTRUSIVE_OBTAIN_TYPE_WITH_EVAL_DEFAULT
( boost::intrusive::detail::, Ptr, element_type
, boost::movelib::detail::first_param<Ptr>)                          element_type;
typedef BOOST_INTRUSIVE_OBTAIN_TYPE_WITH_DEFAULT
(boost::intrusive::detail::, Ptr, difference_type, std::ptrdiff_t)   difference_type;

typedef BOOST_INTRUSIVE_OBTAIN_TYPE_WITH_DEFAULT
(boost::intrusive::detail::, Ptr, reference, typename boost::intrusive::detail::unvoid_ref<element_type>::type)   reference;
template <class U> struct rebind_pointer
{
typedef typename boost::intrusive::pointer_rebind<Ptr, U>::type  type;
};

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
template <class U> using rebind = typename boost::intrusive::pointer_rebind<Ptr, U>::type;
#endif
#endif   

BOOST_INTRUSIVE_FORCEINLINE static pointer pointer_to(reference r)
{
const bool value = boost::intrusive::detail::
has_member_function_callable_with_pointer_to
<Ptr, Ptr (*)(reference)>::value;
boost::intrusive::detail::bool_<value> flag;
return pointer_traits::priv_pointer_to(flag, r);
}

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer static_cast_from(const UPtr &uptr)
{
typedef const UPtr &RefArg;
const bool value = boost::intrusive::detail::
has_member_function_callable_with_static_cast_from
<pointer, pointer(*)(RefArg)>::value
|| boost::intrusive::detail::
has_member_function_callable_with_static_cast_from
<pointer, pointer(*)(UPtr)>::value;
return pointer_traits::priv_static_cast_from(boost::intrusive::detail::bool_<value>(), uptr);
}

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer const_cast_from(const UPtr &uptr)
{
typedef const UPtr &RefArg;
const bool value = boost::intrusive::detail::
has_member_function_callable_with_const_cast_from
<pointer, pointer(*)(RefArg)>::value
|| boost::intrusive::detail::
has_member_function_callable_with_const_cast_from
<pointer, pointer(*)(UPtr)>::value;
return pointer_traits::priv_const_cast_from(boost::intrusive::detail::bool_<value>(), uptr);
}

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer dynamic_cast_from(const UPtr &uptr)
{
typedef const UPtr &RefArg;
const bool value = boost::intrusive::detail::
has_member_function_callable_with_dynamic_cast_from
<pointer, pointer(*)(RefArg)>::value
|| boost::intrusive::detail::
has_member_function_callable_with_dynamic_cast_from
<pointer, pointer(*)(UPtr)>::value;
return pointer_traits::priv_dynamic_cast_from(boost::intrusive::detail::bool_<value>(), uptr);
}

private:
template <class T>
BOOST_INTRUSIVE_FORCEINLINE static T* to_raw_pointer(T* p)
{  return p; }

template <class Pointer>
BOOST_INTRUSIVE_FORCEINLINE static typename pointer_traits<Pointer>::element_type*
to_raw_pointer(const Pointer &p)
{  return pointer_traits::to_raw_pointer(p.operator->());  }

BOOST_INTRUSIVE_FORCEINLINE static pointer priv_pointer_to(boost::intrusive::detail::true_, reference r)
{ return Ptr::pointer_to(r); }

BOOST_INTRUSIVE_FORCEINLINE static pointer priv_pointer_to(boost::intrusive::detail::false_, reference r)
{ return pointer(boost::intrusive::detail::addressof(r)); }

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer priv_static_cast_from(boost::intrusive::detail::true_, const UPtr &uptr)
{ return Ptr::static_cast_from(uptr); }

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer priv_static_cast_from(boost::intrusive::detail::false_, const UPtr &uptr)
{  return uptr ? pointer_to(*static_cast<element_type*>(to_raw_pointer(uptr))) : pointer();  }

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer priv_const_cast_from(boost::intrusive::detail::true_, const UPtr &uptr)
{ return Ptr::const_cast_from(uptr); }

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer priv_const_cast_from(boost::intrusive::detail::false_, const UPtr &uptr)
{  return uptr ? pointer_to(const_cast<element_type&>(*uptr)) : pointer();  }

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer priv_dynamic_cast_from(boost::intrusive::detail::true_, const UPtr &uptr)
{ return Ptr::dynamic_cast_from(uptr); }

template<class UPtr>
BOOST_INTRUSIVE_FORCEINLINE static pointer priv_dynamic_cast_from(boost::intrusive::detail::false_, const UPtr &uptr)
{  return uptr ? pointer_to(dynamic_cast<element_type&>(*uptr)) : pointer();  }
};


template <typename Ptr>
struct pointer_traits<const Ptr> : pointer_traits<Ptr> {};
template <typename Ptr>
struct pointer_traits<volatile Ptr> : pointer_traits<Ptr> { };
template <typename Ptr>
struct pointer_traits<const volatile Ptr> : pointer_traits<Ptr> { };
template <typename Ptr>
struct pointer_traits<Ptr&> : pointer_traits<Ptr> { };


template <typename T>
struct pointer_traits<T*>
{
typedef T            element_type;
typedef T*           pointer;
typedef std::ptrdiff_t difference_type;

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
typedef T &          reference;
template <class U> using rebind = U*;
#else
typedef typename boost::intrusive::detail::unvoid_ref<element_type>::type reference;
#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
template <class U> using rebind = U*;
#endif
#endif

template <class U> struct rebind_pointer
{  typedef U* type;  };

BOOST_INTRUSIVE_FORCEINLINE static pointer pointer_to(reference r)
{ return boost::intrusive::detail::addressof(r); }

template<class U>
BOOST_INTRUSIVE_FORCEINLINE static pointer static_cast_from(U *uptr)
{  return static_cast<pointer>(uptr);  }

template<class U>
BOOST_INTRUSIVE_FORCEINLINE static pointer const_cast_from(U *uptr)
{  return const_cast<pointer>(uptr);  }

template<class U>
BOOST_INTRUSIVE_FORCEINLINE static pointer dynamic_cast_from(U *uptr)
{  return dynamic_cast<pointer>(uptr);  }
};

}  
}  

#include <boost/intrusive/detail/config_end.hpp>

#endif 
