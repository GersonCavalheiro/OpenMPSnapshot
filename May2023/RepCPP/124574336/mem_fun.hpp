

#ifndef BOOST_MULTI_INDEX_MEM_FUN_HPP
#define BOOST_MULTI_INDEX_MEM_FUN_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/if.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/enable_if.hpp>

#if !defined(BOOST_NO_SFINAE)
#include <boost/type_traits/is_convertible.hpp>
#endif

namespace boost{

template<class T> class reference_wrapper; 

namespace multi_index{



namespace detail{

template<
class Class,typename Type,
typename PtrToMemberFunctionType,PtrToMemberFunctionType PtrToMemberFunction
>
struct const_mem_fun_impl
{
typedef typename remove_reference<Type>::type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Class&>,Type>::type
#else
Type
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type operator()(const Class& x)const
{
return (x.*PtrToMemberFunction)();
}

Type operator()(const reference_wrapper<const Class>& x)const
{ 
return operator()(x.get());
}

Type operator()(const reference_wrapper<Class>& x)const
{ 
return operator()(x.get());
}
};

template<
class Class,typename Type,
typename PtrToMemberFunctionType,PtrToMemberFunctionType PtrToMemberFunction
>
struct mem_fun_impl
{
typedef typename remove_reference<Type>::type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<ChainedPtr&,Class&>,Type>::type
#else
Type
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type operator()(Class& x)const
{
return (x.*PtrToMemberFunction)();
}

Type operator()(const reference_wrapper<Class>& x)const
{ 
return operator()(x.get());
}
};

} 

template<class Class,typename Type,Type (Class::*PtrToMemberFunction)()const>
struct const_mem_fun:detail::const_mem_fun_impl<
Class,Type,Type (Class::*)()const,PtrToMemberFunction
>{};

template<
class Class,typename Type,
Type (Class::*PtrToMemberFunction)()const volatile
>
struct cv_mem_fun:detail::const_mem_fun_impl<
Class,Type,Type (Class::*)()const volatile,PtrToMemberFunction
>{};

template<class Class,typename Type,Type (Class::*PtrToMemberFunction)()>
struct mem_fun:
detail::mem_fun_impl<Class,Type,Type (Class::*)(),PtrToMemberFunction>{};

template<
class Class,typename Type,Type (Class::*PtrToMemberFunction)()volatile
>
struct volatile_mem_fun:detail::mem_fun_impl<
Class,Type,Type (Class::*)()volatile,PtrToMemberFunction
>{};

#if !defined(BOOST_NO_CXX11_REF_QUALIFIERS)

template<
class Class,typename Type,Type (Class::*PtrToMemberFunction)()const&
>
struct cref_mem_fun:detail::const_mem_fun_impl<
Class,Type,Type (Class::*)()const&,PtrToMemberFunction
>{};

template<
class Class,typename Type,
Type (Class::*PtrToMemberFunction)()const volatile&
>
struct cvref_mem_fun:detail::const_mem_fun_impl<
Class,Type,Type (Class::*)()const volatile&,PtrToMemberFunction
>{};

template<class Class,typename Type,Type (Class::*PtrToMemberFunction)()&>
struct ref_mem_fun:
detail::mem_fun_impl<Class,Type,Type (Class::*)()&,PtrToMemberFunction>{};

template<
class Class,typename Type,Type (Class::*PtrToMemberFunction)()volatile&
>
struct vref_mem_fun:detail::mem_fun_impl<
Class,Type,Type (Class::*)()volatile&,PtrToMemberFunction
>{};

#endif



template<
class Class,typename Type,
typename PtrToMemberFunctionType,PtrToMemberFunctionType PtrToMemberFunction
>
struct const_mem_fun_explicit
{
typedef typename remove_reference<Type>::type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<const ChainedPtr&,const Class&>,Type>::type
#else
Type
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type operator()(const Class& x)const
{
return (x.*PtrToMemberFunction)();
}

Type operator()(const reference_wrapper<const Class>& x)const
{ 
return operator()(x.get());
}

Type operator()(const reference_wrapper<Class>& x)const
{ 
return operator()(x.get());
}
};

template<
class Class,typename Type,
typename PtrToMemberFunctionType,PtrToMemberFunctionType PtrToMemberFunction
>
struct mem_fun_explicit
{
typedef typename remove_reference<Type>::type result_type;

template<typename ChainedPtr>

#if !defined(BOOST_NO_SFINAE)
typename disable_if<
is_convertible<ChainedPtr&,Class&>,Type>::type
#else
Type
#endif

operator()(const ChainedPtr& x)const
{
return operator()(*x);
}

Type operator()(Class& x)const
{
return (x.*PtrToMemberFunction)();
}

Type operator()(const reference_wrapper<Class>& x)const
{ 
return operator()(x.get());
}
};



#define BOOST_MULTI_INDEX_CONST_MEM_FUN(Class,Type,MemberFunName) \
::boost::multi_index::const_mem_fun< Class,Type,&Class::MemberFunName >
#define BOOST_MULTI_INDEX_MEM_FUN(Class,Type,MemberFunName) \
::boost::multi_index::mem_fun< Class,Type,&Class::MemberFunName >

} 

} 

#endif
