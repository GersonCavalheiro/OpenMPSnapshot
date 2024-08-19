

#ifndef BOOST_MOVE_CORE_HPP
#define BOOST_MOVE_CORE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/move/detail/config_begin.hpp>
#include <boost/move/detail/workaround.hpp>


#if defined(BOOST_NO_CXX11_DELETED_FUNCTIONS) || defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#define BOOST_MOVE_IMPL_NO_COPY_CTOR_OR_ASSIGN(TYPE) \
private:\
TYPE(TYPE &);\
TYPE& operator=(TYPE &);\
public:\
typedef int boost_move_no_copy_constructor_or_assign; \
private:\
#else
#define BOOST_MOVE_IMPL_NO_COPY_CTOR_OR_ASSIGN(TYPE) \
public:\
TYPE(TYPE const &) = delete;\
TYPE& operator=(TYPE const &) = delete;\
public:\
typedef int boost_move_no_copy_constructor_or_assign; \
private:\
#endif   


#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !defined(BOOST_MOVE_DOXYGEN_INVOKED)

#include <boost/move/detail/type_traits.hpp>

#define BOOST_MOVE_TO_RV_CAST(RV_TYPE, ARG) reinterpret_cast<RV_TYPE>(ARG)

#if defined(BOOST_GCC) && (BOOST_GCC >= 40400) && (BOOST_GCC < 40500)
#define BOOST_RV_ATTRIBUTE_MAY_ALIAS BOOST_MAY_ALIAS
#else
#define BOOST_RV_ATTRIBUTE_MAY_ALIAS 
#endif

namespace boost {

template <class T>
class BOOST_RV_ATTRIBUTE_MAY_ALIAS rv
: public ::boost::move_detail::if_c
< ::boost::move_detail::is_class<T>::value
, T
, ::boost::move_detail::nat
>::type
{
rv();
~rv() throw();
rv(rv const&);
void operator=(rv const&);
};



namespace move_detail {

template <class T>
struct is_rv
: integral_constant<bool, ::boost::move_detail::is_rv_impl<T>::value >
{};

template <class T>
struct is_not_rv
{
static const bool value = !is_rv<T>::value;
};

}  

template<class T>
struct has_move_emulation_enabled
: ::boost::move_detail::has_move_emulation_enabled_impl<T>
{};

template<class T>
struct has_move_emulation_disabled
{
static const bool value = !::boost::move_detail::has_move_emulation_enabled_impl<T>::value;
};

}  

#define BOOST_RV_REF(TYPE)\
::boost::rv< TYPE >& \

#define BOOST_RV_REF_2_TEMPL_ARGS(TYPE, ARG1, ARG2)\
::boost::rv< TYPE<ARG1, ARG2> >& \

#define BOOST_RV_REF_3_TEMPL_ARGS(TYPE, ARG1, ARG2, ARG3)\
::boost::rv< TYPE<ARG1, ARG2, ARG3> >& \

#define BOOST_RV_REF_BEG\
::boost::rv<   \

#define BOOST_RV_REF_END\
>& \

#define BOOST_RV_REF_BEG_IF_CXX11 \
\

#define BOOST_RV_REF_END_IF_CXX11 \
\

#define BOOST_FWD_REF(TYPE)\
const TYPE & \

#define BOOST_COPY_ASSIGN_REF(TYPE)\
const ::boost::rv< TYPE >& \

#define BOOST_COPY_ASSIGN_REF_BEG \
const ::boost::rv<  \

#define BOOST_COPY_ASSIGN_REF_END \
>& \

#define BOOST_COPY_ASSIGN_REF_2_TEMPL_ARGS(TYPE, ARG1, ARG2)\
const ::boost::rv< TYPE<ARG1, ARG2> >& \

#define BOOST_COPY_ASSIGN_REF_3_TEMPL_ARGS(TYPE, ARG1, ARG2, ARG3)\
const ::boost::rv< TYPE<ARG1, ARG2, ARG3> >& \

#define BOOST_CATCH_CONST_RLVALUE(TYPE)\
const ::boost::rv< TYPE >& \

namespace boost {
namespace move_detail {

template <class Ret, class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_c
<  ::boost::move_detail::is_lvalue_reference<Ret>::value ||
!::boost::has_move_emulation_enabled<T>::value
, T&>::type
move_return(T& x) BOOST_NOEXCEPT
{
return x;
}

template <class Ret, class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_c
< !::boost::move_detail::is_lvalue_reference<Ret>::value &&
::boost::has_move_emulation_enabled<T>::value
, ::boost::rv<T>&>::type
move_return(T& x) BOOST_NOEXCEPT
{
return *BOOST_MOVE_TO_RV_CAST(::boost::rv<T>*, ::boost::move_detail::addressof(x));
}

template <class Ret, class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_c
< !::boost::move_detail::is_lvalue_reference<Ret>::value &&
::boost::has_move_emulation_enabled<T>::value
, ::boost::rv<T>&>::type
move_return(::boost::rv<T>& x) BOOST_NOEXCEPT
{
return x;
}

}  
}  

#define BOOST_MOVE_RET(RET_TYPE, REF)\
boost::move_detail::move_return< RET_TYPE >(REF)

#define BOOST_MOVE_BASE(BASE_TYPE, ARG) \
::boost::move((BASE_TYPE&)(ARG))

#define BOOST_MOVABLE_BUT_NOT_COPYABLE(TYPE)\
BOOST_MOVE_IMPL_NO_COPY_CTOR_OR_ASSIGN(TYPE)\
public:\
BOOST_MOVE_FORCEINLINE operator ::boost::rv<TYPE>&() \
{  return *BOOST_MOVE_TO_RV_CAST(::boost::rv<TYPE>*, this);  }\
BOOST_MOVE_FORCEINLINE operator const ::boost::rv<TYPE>&() const \
{  return *BOOST_MOVE_TO_RV_CAST(const ::boost::rv<TYPE>*, this);  }\
private:\


#define BOOST_COPYABLE_AND_MOVABLE(TYPE)\
public:\
BOOST_MOVE_FORCEINLINE TYPE& operator=(TYPE &t)\
{  this->operator=(const_cast<const TYPE&>(t)); return *this;}\
public:\
BOOST_MOVE_FORCEINLINE operator ::boost::rv<TYPE>&() \
{  return *BOOST_MOVE_TO_RV_CAST(::boost::rv<TYPE>*, this);  }\
BOOST_MOVE_FORCEINLINE operator const ::boost::rv<TYPE>&() const \
{  return *BOOST_MOVE_TO_RV_CAST(const ::boost::rv<TYPE>*, this);  }\
private:\

#define BOOST_COPYABLE_AND_MOVABLE_ALT(TYPE)\
public:\
BOOST_MOVE_FORCEINLINE operator ::boost::rv<TYPE>&() \
{  return *BOOST_MOVE_TO_RV_CAST(::boost::rv<TYPE>*, this);  }\
BOOST_MOVE_FORCEINLINE operator const ::boost::rv<TYPE>&() const \
{  return *BOOST_MOVE_TO_RV_CAST(const ::boost::rv<TYPE>*, this);  }\
private:\

namespace boost{
namespace move_detail{

template< class T>
struct forward_type
{ typedef const T &type; };

template< class T>
struct forward_type< boost::rv<T> >
{ typedef T type; };

}}

#else    

#define BOOST_MOVABLE_BUT_NOT_COPYABLE(TYPE)\
BOOST_MOVE_IMPL_NO_COPY_CTOR_OR_ASSIGN(TYPE)\
public:\
typedef int boost_move_emulation_t;\
private:\

#define BOOST_COPYABLE_AND_MOVABLE(TYPE)\

#if !defined(BOOST_MOVE_DOXYGEN_INVOKED)
#define BOOST_COPYABLE_AND_MOVABLE_ALT(TYPE)\
#endif   

namespace boost {

template<class T>
struct has_move_emulation_enabled
{
static const bool value = false;
};

template<class T>
struct has_move_emulation_disabled
{
static const bool value = true;
};

}  

#define BOOST_RV_REF(TYPE)\
TYPE && \

#define BOOST_RV_REF_BEG\
\

#define BOOST_RV_REF_END\
&& \

#define BOOST_RV_REF_BEG_IF_CXX11 \
BOOST_RV_REF_BEG \

#define BOOST_RV_REF_END_IF_CXX11 \
BOOST_RV_REF_END \

#define BOOST_COPY_ASSIGN_REF(TYPE)\
const TYPE & \

#define BOOST_FWD_REF(TYPE)\
TYPE && \

#if !defined(BOOST_MOVE_DOXYGEN_INVOKED)

#define BOOST_RV_REF_2_TEMPL_ARGS(TYPE, ARG1, ARG2)\
TYPE<ARG1, ARG2> && \

#define BOOST_RV_REF_3_TEMPL_ARGS(TYPE, ARG1, ARG2, ARG3)\
TYPE<ARG1, ARG2, ARG3> && \

#define BOOST_COPY_ASSIGN_REF_BEG \
const \

#define BOOST_COPY_ASSIGN_REF_END \
& \

#define BOOST_COPY_ASSIGN_REF_2_TEMPL_ARGS(TYPE, ARG1, ARG2)\
const TYPE<ARG1, ARG2> & \

#define BOOST_COPY_ASSIGN_REF_3_TEMPL_ARGS(TYPE, ARG1, ARG2, ARG3)\
const TYPE<ARG1, ARG2, ARG3>& \

#define BOOST_CATCH_CONST_RLVALUE(TYPE)\
const TYPE & \

#endif   

#if !defined(BOOST_MOVE_MSVC_AUTO_MOVE_RETURN_BUG) || defined(BOOST_MOVE_DOXYGEN_INVOKED)

#define BOOST_MOVE_RET(RET_TYPE, REF)\
REF

#else 

#include <boost/move/detail/meta_utils.hpp>

namespace boost {
namespace move_detail {

template <class Ret, class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_c
<  ::boost::move_detail::is_lvalue_reference<Ret>::value
, T&>::type
move_return(T& x) BOOST_NOEXCEPT
{
return x;
}

template <class Ret, class T>
BOOST_MOVE_FORCEINLINE typename ::boost::move_detail::enable_if_c
< !::boost::move_detail::is_lvalue_reference<Ret>::value
, Ret && >::type
move_return(T&& t) BOOST_NOEXCEPT
{
return static_cast< Ret&& >(t);
}

}  
}  

#define BOOST_MOVE_RET(RET_TYPE, REF)\
boost::move_detail::move_return< RET_TYPE >(REF)

#endif   

#define BOOST_MOVE_BASE(BASE_TYPE, ARG) \
::boost::move((BASE_TYPE&)(ARG))

namespace boost {
namespace move_detail {

template< class T> struct forward_type { typedef T type; };

}}

#endif   

#include <boost/move/detail/config_end.hpp>

#endif 
