

#ifndef BOOST_MULTI_INDEX_IDENTITY_HPP
#define BOOST_MULTI_INDEX_IDENTITY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/mpl/if.hpp>
#include <boost/multi_index/identity_fwd.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

#if !defined(BOOST_NO_SFINAE)
#include <boost/type_traits/is_convertible.hpp>
#endif

namespace boost{

template<class Type> class reference_wrapper; 

namespace multi_index{

namespace detail{



template<typename Type>
struct const_identity_base
{
typedef Type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<is_convertible<const ChainedPtr&,Type&>,Type&>::type
#else
Type&
#endif 

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type& operator()(Type& x)const
{
return x;
}

Type& operator()(const reference_wrapper<Type>& x)const
{ 
return x.get();
}

Type& operator()(
const reference_wrapper<typename remove_const<Type>::type>& x

#if BOOST_WORKAROUND(BOOST_MSVC,==1310)

,int=0
#endif

)const
{ 
return x.get();
}
};

template<typename Type>
struct non_const_identity_base
{
typedef Type result_type;



template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Type&>,Type&>::type
#else
Type&
#endif 

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

const Type& operator()(const Type& x)const
{
return x;
}

Type& operator()(Type& x)const
{
return x;
}

const Type& operator()(const reference_wrapper<const Type>& x)const
{ 
return x.get();
}

Type& operator()(const reference_wrapper<Type>& x)const
{ 
return x.get();
}
};

} 

template<class Type>
struct identity:
mpl::if_c<
is_const<Type>::value,
detail::const_identity_base<Type>,detail::non_const_identity_base<Type>
>::type
{
};

} 

} 

#endif
