

#ifndef BOOST_MULTI_INDEX_MEMBER_HPP
#define BOOST_MULTI_INDEX_MEMBER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/utility/enable_if.hpp>
#include <cstddef>

#if !defined(BOOST_NO_SFINAE)
#include <boost/type_traits/is_convertible.hpp>
#endif

namespace boost{

template<class T> class reference_wrapper; 

namespace multi_index{

namespace detail{



template<class Class,typename Type,Type Class::*PtrToMember>
struct const_member_base
{
typedef Type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Class&>,Type&>::type
#else
Type&
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type& operator()(const Class& x)const
{
return x.*PtrToMember;
}

Type& operator()(const reference_wrapper<const Class>& x)const
{
return operator()(x.get());
}

Type& operator()(const reference_wrapper<Class>& x)const
{ 
return operator()(x.get());
}
};

template<class Class,typename Type,Type Class::*PtrToMember>
struct non_const_member_base
{
typedef Type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Class&>,Type&>::type
#else
Type&
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

const Type& operator()(const Class& x)const
{
return x.*PtrToMember;
}

Type& operator()(Class& x)const
{ 
return x.*PtrToMember;
}

const Type& operator()(const reference_wrapper<const Class>& x)const
{
return operator()(x.get());
}

Type& operator()(const reference_wrapper<Class>& x)const
{ 
return operator()(x.get());
}
};

} 

template<class Class,typename Type,Type Class::*PtrToMember>
struct member:
mpl::if_c<
is_const<Type>::value,
detail::const_member_base<Class,Type,PtrToMember>,
detail::non_const_member_base<Class,Type,PtrToMember>
>::type
{
};

namespace detail{



template<class Class,typename Type,std::size_t OffsetOfMember>
struct const_member_offset_base
{
typedef Type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Class&>,Type&>::type
#else
Type&
#endif 

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type& operator()(const Class& x)const
{
return *static_cast<const Type*>(
static_cast<const void*>(
static_cast<const char*>(
static_cast<const void *>(&x))+OffsetOfMember));
}

Type& operator()(const reference_wrapper<const Class>& x)const
{
return operator()(x.get());
}

Type& operator()(const reference_wrapper<Class>& x)const
{
return operator()(x.get());
}
};

template<class Class,typename Type,std::size_t OffsetOfMember>
struct non_const_member_offset_base
{
typedef Type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Class&>,Type&>::type
#else
Type&
#endif 

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

const Type& operator()(const Class& x)const
{
return *static_cast<const Type*>(
static_cast<const void*>(
static_cast<const char*>(
static_cast<const void *>(&x))+OffsetOfMember));
}

Type& operator()(Class& x)const
{ 
return *static_cast<Type*>(
static_cast<void*>(
static_cast<char*>(static_cast<void *>(&x))+OffsetOfMember));
}

const Type& operator()(const reference_wrapper<const Class>& x)const
{
return operator()(x.get());
}

Type& operator()(const reference_wrapper<Class>& x)const
{
return operator()(x.get());
}
};

} 

template<class Class,typename Type,std::size_t OffsetOfMember>
struct member_offset:
mpl::if_c<
is_const<Type>::value,
detail::const_member_offset_base<Class,Type,OffsetOfMember>,
detail::non_const_member_offset_base<Class,Type,OffsetOfMember>
>::type
{
};



#if defined(BOOST_NO_POINTER_TO_MEMBER_TEMPLATE_PARAMETERS)
#define BOOST_MULTI_INDEX_MEMBER(Class,Type,MemberName) \
::boost::multi_index::member_offset< Class,Type,offsetof(Class,MemberName) >
#else
#define BOOST_MULTI_INDEX_MEMBER(Class,Type,MemberName) \
::boost::multi_index::member< Class,Type,&Class::MemberName >
#endif

} 

} 

#endif
