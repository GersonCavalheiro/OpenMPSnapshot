
#ifndef BOOST_MOVE_DEFAULT_DELETE_HPP_INCLUDED
#define BOOST_MOVE_DEFAULT_DELETE_HPP_INCLUDED

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/move/detail/config_begin.hpp>
#include <boost/move/detail/workaround.hpp>
#include <boost/move/detail/unique_ptr_meta_utils.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/static_assert.hpp>

#include <cstddef>   


namespace boost{
namespace move_upd {

namespace bmupmu = ::boost::move_upmu;


template<class U, class T>
struct def_del_compatible_cond
: bmupmu::is_convertible<U*, T*>
{};

template<class U, class T, std::size_t N>
struct def_del_compatible_cond<U[N], T[]>
: def_del_compatible_cond<U[], T[]>
{};

template<class U, class T, class Type = bmupmu::nat>
struct enable_def_del
: bmupmu::enable_if_c<def_del_compatible_cond<U, T>::value, Type>
{};



template<class U, class T, class Type = bmupmu::nat>
struct enable_defdel_call
: public enable_def_del<U, T, Type>
{};

template<class U, class T, class Type>
struct enable_defdel_call<U, T[], Type>
: public enable_def_del<U[], T[], Type>
{};

template<class U, class T, class Type, std::size_t N>
struct enable_defdel_call<U, T[N], Type>
: public enable_def_del<U[N], T[N], Type>
{};


struct bool_conversion {int for_bool; int for_arg(); };
typedef int bool_conversion::* explicit_bool_arg;

#if !defined(BOOST_NO_CXX11_NULLPTR) && !defined(BOOST_NO_CXX11_DECLTYPE)
typedef decltype(nullptr) nullptr_type;
#elif !defined(BOOST_NO_CXX11_NULLPTR)
typedef std::nullptr_t nullptr_type;
#else
typedef int (bool_conversion::*nullptr_type)();
#endif

template<bool B>
struct is_array_del
{};

template<class T>
void call_delete(T *p, is_array_del<true>)
{
delete [] p;
}

template<class T>
void call_delete(T *p, is_array_del<false>)
{
delete p;
}

template< class T, class U
, bool enable =  def_del_compatible_cond< U, T>::value &&
!move_upmu::is_array<T>::value &&
!move_upmu::is_same<typename move_upmu::remove_cv<T>::type, void>::value &&
!move_upmu::is_same<typename move_upmu::remove_cv<U>::type, typename move_upmu::remove_cv<T>::type>::value
>
struct missing_virtual_destructor_default_delete
{  static const bool value = !move_upmu::has_virtual_destructor<T>::value;  };

template<class T, class U>
struct missing_virtual_destructor_default_delete<T, U, false>
{  static const bool value = false;  };


template<class Deleter, class U>
struct missing_virtual_destructor
{  static const bool value = false;  };

template<class T, class U>
struct missing_virtual_destructor< ::boost::movelib::default_delete<T>, U >
: missing_virtual_destructor_default_delete<T, U>
{};


}  

namespace movelib {

namespace bmupd = boost::move_upd;
namespace bmupmu = ::boost::move_upmu;

template <class T>
struct default_delete
{
BOOST_CONSTEXPR default_delete()
#if !defined(BOOST_GCC) || (BOOST_GCC < 40600 && BOOST_GCC >= 40700) || defined(BOOST_MOVE_DOXYGEN_INVOKED)
BOOST_NOEXCEPT
#endif
#if !defined(BOOST_NO_CXX11_DEFAULTED_FUNCTIONS) || defined(BOOST_MOVE_DOXYGEN_INVOKED)
= default;
#else
{};
#endif

#if defined(BOOST_MOVE_DOXYGEN_INVOKED)
default_delete(const default_delete&) BOOST_NOEXCEPT = default;
default_delete &operator=(const default_delete&) BOOST_NOEXCEPT = default;
#else
typedef typename bmupmu::remove_extent<T>::type element_type;
#endif

template <class U>
default_delete(const default_delete<U>&
BOOST_MOVE_DOCIGN(BOOST_MOVE_I typename bmupd::enable_def_del<U BOOST_MOVE_I T>::type* =0)
) BOOST_NOEXCEPT
{
BOOST_STATIC_ASSERT(( !bmupd::missing_virtual_destructor<default_delete, U>::value ));
}

template <class U>
BOOST_MOVE_DOC1ST(default_delete&, 
typename bmupd::enable_def_del<U BOOST_MOVE_I T BOOST_MOVE_I default_delete &>::type)
operator=(const default_delete<U>&) BOOST_NOEXCEPT
{
BOOST_STATIC_ASSERT(( !bmupd::missing_virtual_destructor<default_delete, U>::value ));
return *this;
}

template <class U>
BOOST_MOVE_DOC1ST(void, typename bmupd::enable_defdel_call<U BOOST_MOVE_I T BOOST_MOVE_I void>::type)
operator()(U* ptr) const BOOST_NOEXCEPT
{
BOOST_STATIC_ASSERT(sizeof(U) > 0);
BOOST_STATIC_ASSERT(( !bmupd::missing_virtual_destructor<default_delete, U>::value ));
element_type * const p = static_cast<element_type*>(ptr);
move_upd::call_delete(p, move_upd::is_array_del<bmupmu::is_array<T>::value>());
}

void operator()(BOOST_MOVE_DOC0PTR(bmupd::nullptr_type)) const BOOST_NOEXCEPT
{  BOOST_STATIC_ASSERT(sizeof(element_type) > 0);  }
};

}  
}  

#include <boost/move/detail/config_end.hpp>

#endif   
