

#ifndef BOOST_MULTI_INDEX_GLOBAL_FUN_HPP
#define BOOST_MULTI_INDEX_GLOBAL_FUN_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/enable_if.hpp>

#if !defined(BOOST_NO_SFINAE)
#include <boost/type_traits/is_convertible.hpp>
#endif

namespace boost{

template<class T> class reference_wrapper; 

namespace multi_index{

namespace detail{



template<class Value,typename Type,Type (*PtrToFunction)(Value)>
struct const_ref_global_fun_base
{
typedef typename remove_reference<Type>::type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,Value>,Type>::type
#else
Type
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type operator()(Value x)const
{
return PtrToFunction(x);
}

Type operator()(
const reference_wrapper<
typename remove_reference<Value>::type>& x)const
{ 
return operator()(x.get());
}

Type operator()(
const reference_wrapper<
typename remove_const<
typename remove_reference<Value>::type>::type>& x

#if BOOST_WORKAROUND(BOOST_MSVC,==1310)

,int=0
#endif

)const
{ 
return operator()(x.get());
}
};

template<class Value,typename Type,Type (*PtrToFunction)(Value)>
struct non_const_ref_global_fun_base
{
typedef typename remove_reference<Type>::type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<ChainedPtr&,Value>,Type>::type
#else
Type
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type operator()(Value x)const
{
return PtrToFunction(x);
}

Type operator()(
const reference_wrapper<
typename remove_reference<Value>::type>& x)const
{ 
return operator()(x.get());
}
};

template<class Value,typename Type,Type (*PtrToFunction)(Value)>
struct non_ref_global_fun_base
{
typedef typename remove_reference<Type>::type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Value&>,Type>::type
#else
Type
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type operator()(const Value& x)const
{
return PtrToFunction(x);
}

Type operator()(const reference_wrapper<const Value>& x)const
{ 
return operator()(x.get());
}

Type operator()(
const reference_wrapper<typename remove_const<Value>::type>& x)const
{ 
return operator()(x.get());
}
};

} 

template<class Value,typename Type,Type (*PtrToFunction)(Value)>
struct global_fun:
mpl::if_c<
is_reference<Value>::value,
typename mpl::if_c<
is_const<typename remove_reference<Value>::type>::value,
detail::const_ref_global_fun_base<Value,Type,PtrToFunction>,
detail::non_const_ref_global_fun_base<Value,Type,PtrToFunction>
>::type,
detail::non_ref_global_fun_base<Value,Type,PtrToFunction>
>::type
{
};

} 

} 

#endif
