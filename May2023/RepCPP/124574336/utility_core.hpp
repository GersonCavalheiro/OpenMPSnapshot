

#ifndef BOOST_MOVE_MOVE_UTILITY_CORE_HPP
#define BOOST_MOVE_MOVE_UTILITY_CORE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/move/detail/config_begin.hpp>
#include <boost/move/detail/workaround.hpp>  
#include <boost/move/core.hpp>
#include <boost/move/detail/meta_utils.hpp>
#include <boost/static_assert.hpp>

#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !defined(BOOST_MOVE_DOXYGEN_INVOKED)

namespace boost {

template<class T>
struct enable_move_utility_emulation
{
static const bool value = true;
};


template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< T &
, enable_move_utility_emulation<T>
, has_move_emulation_disabled<T>
>::type
move(T& x) BOOST_NOEXCEPT
{
return x;
}

template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< rv<T>&
, enable_move_utility_emulation<T>
, has_move_emulation_enabled<T>
>::type
move(T& x) BOOST_NOEXCEPT
{
return *BOOST_MOVE_TO_RV_CAST(::boost::rv<T>*, ::boost::move_detail::addressof(x) );
}

template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< rv<T>&
, enable_move_utility_emulation<T>
, has_move_emulation_enabled<T>
>::type
move(rv<T>& x) BOOST_NOEXCEPT
{
return x;
}


template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< T &
, enable_move_utility_emulation<T>
, ::boost::move_detail::is_rv<T>
>::type
forward(const typename ::boost::move_detail::identity<T>::type &x) BOOST_NOEXCEPT
{
return const_cast<T&>(x);
}

template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< const T &
, enable_move_utility_emulation<T>
, ::boost::move_detail::is_not_rv<T>
>::type
forward(const typename ::boost::move_detail::identity<T>::type &x) BOOST_NOEXCEPT
{
return x;
}


template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< T &
, enable_move_utility_emulation<T>
, ::boost::move_detail::is_rv<T>
>::type
move_if_not_lvalue_reference(const typename ::boost::move_detail::identity<T>::type &x) BOOST_NOEXCEPT
{
return const_cast<T&>(x);
}

template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< typename ::boost::move_detail::add_lvalue_reference<T>::type
, enable_move_utility_emulation<T>
, ::boost::move_detail::is_not_rv<T>
, ::boost::move_detail::or_
< ::boost::move_detail::is_lvalue_reference<T>
, has_move_emulation_disabled<T>
>
>::type
move_if_not_lvalue_reference(typename ::boost::move_detail::remove_reference<T>::type &x) BOOST_NOEXCEPT
{
return x;
}

template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_and
< rv<T>&
, enable_move_utility_emulation<T>
, ::boost::move_detail::is_not_rv<T>
, ::boost::move_detail::and_
< ::boost::move_detail::not_< ::boost::move_detail::is_lvalue_reference<T> >
, has_move_emulation_enabled<T>
>
>::type
move_if_not_lvalue_reference(typename ::boost::move_detail::remove_reference<T>::type &x) BOOST_NOEXCEPT
{
return move(x);
}

}  

#else    

#if defined(BOOST_MOVE_USE_STANDARD_LIBRARY_MOVE)
#include <utility>

namespace boost{

using ::std::move;
using ::std::forward;

}  

#else 

namespace boost {

template<class T>
struct enable_move_utility_emulation
{
static const bool value = false;
};


#if defined(BOOST_MOVE_DOXYGEN_INVOKED)
template <class T>
rvalue_reference move(input_reference) noexcept;

#elif defined(BOOST_MOVE_OLD_RVALUE_REF_BINDING_RULES)

template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::remove_reference<T>::type && move(T&& t) BOOST_NOEXCEPT
{  return t;   }

#else 

template <class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::remove_reference<T>::type && move(T&& t) BOOST_NOEXCEPT
{ return static_cast<typename ::boost::move_detail::remove_reference<T>::type &&>(t); }

#endif   



#if defined(BOOST_MOVE_DOXYGEN_INVOKED)
template <class T> output_reference forward(input_reference) noexcept;
#elif defined(BOOST_MOVE_OLD_RVALUE_REF_BINDING_RULES)


template <class T>
BOOST_MOVE_FORCEINLINE T&& forward(typename ::boost::move_detail::identity<T>::type&& t) BOOST_NOEXCEPT
{  return t;   }

#else 

template <class T>
BOOST_MOVE_FORCEINLINE T&& forward(typename ::boost::move_detail::remove_reference<T>::type& t) BOOST_NOEXCEPT
{  return static_cast<T&&>(t);   }

template <class T>
BOOST_MOVE_FORCEINLINE T&& forward(typename ::boost::move_detail::remove_reference<T>::type&& t) BOOST_NOEXCEPT
{
BOOST_STATIC_ASSERT(!boost::move_detail::is_lvalue_reference<T>::value);
return static_cast<T&&>(t);
}

#endif   



#if defined(BOOST_MOVE_DOXYGEN_INVOKED)
template <class T> output_reference move_if_not_lvalue_reference(input_reference) noexcept;
#elif defined(BOOST_MOVE_OLD_RVALUE_REF_BINDING_RULES)


template <class T>
BOOST_MOVE_FORCEINLINE T&& move_if_not_lvalue_reference(typename ::boost::move_detail::identity<T>::type&& t) BOOST_NOEXCEPT
{  return t;   }

#else 

template <class T>
BOOST_MOVE_FORCEINLINE T&& move_if_not_lvalue_reference(typename ::boost::move_detail::remove_reference<T>::type& t) BOOST_NOEXCEPT
{  return static_cast<T&&>(t);   }

template <class T>
BOOST_MOVE_FORCEINLINE T&& move_if_not_lvalue_reference(typename ::boost::move_detail::remove_reference<T>::type&& t) BOOST_NOEXCEPT
{
BOOST_STATIC_ASSERT(!boost::move_detail::is_lvalue_reference<T>::value);
return static_cast<T&&>(t);
}

#endif   

}  

#endif   

#endif   

#if !defined(BOOST_MOVE_DOXYGEN_INVOKED)

namespace boost{
namespace move_detail{

template <typename T>
typename boost::move_detail::add_rvalue_reference<T>::type declval();

}  
}  

#endif   


#include <boost/move/detail/config_end.hpp>

#endif 
