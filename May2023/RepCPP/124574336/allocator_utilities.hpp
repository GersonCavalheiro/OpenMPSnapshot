

#ifndef BOOST_DETAIL_ALLOCATOR_UTILITIES_HPP
#define BOOST_DETAIL_ALLOCATOR_UTILITIES_HPP

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/detail/select_type.hpp>
#include <boost/type_traits/is_same.hpp>
#include <cstddef>
#include <memory>
#include <new>

namespace boost{

namespace detail{



namespace allocator{



template<typename Type>
class partial_std_allocator_wrapper:public std::allocator<Type>
{
public:


typedef Type value_type;

partial_std_allocator_wrapper(){}

template<typename Other>
partial_std_allocator_wrapper(const partial_std_allocator_wrapper<Other>&){}

partial_std_allocator_wrapper(const std::allocator<Type>& x):
std::allocator<Type>(x)
{
}

#if defined(BOOST_DINKUMWARE_STDLIB)


Type* allocate(std::size_t n,const void* hint=0)
{
std::allocator<Type>& a=*this;
return a.allocate(n,hint);
}
#endif

};



#if defined(BOOST_NO_STD_ALLOCATOR)&&\
(defined(BOOST_HAS_PARTIAL_STD_ALLOCATOR)||defined(BOOST_DINKUMWARE_STDLIB))

template<typename Allocator>
struct is_partial_std_allocator
{
BOOST_STATIC_CONSTANT(bool,
value=
(is_same<
std::allocator<BOOST_DEDUCED_TYPENAME Allocator::value_type>,
Allocator
>::value)||
(is_same<
partial_std_allocator_wrapper<
BOOST_DEDUCED_TYPENAME Allocator::value_type>,
Allocator
>::value));
};

#else

template<typename Allocator>
struct is_partial_std_allocator
{
BOOST_STATIC_CONSTANT(bool,value=false);
};

#endif



template<typename Allocator,typename Type>
struct partial_std_allocator_rebind_to
{
typedef partial_std_allocator_wrapper<Type> type;
};



template<typename Allocator>
struct rebinder
{
template<typename Type>
struct result
{
#ifdef BOOST_NO_CXX11_ALLOCATOR
typedef typename Allocator::BOOST_NESTED_TEMPLATE
rebind<Type>::other other;
#else
typedef typename std::allocator_traits<Allocator>::BOOST_NESTED_TEMPLATE
rebind_alloc<Type> other;
#endif
};
};

template<typename Allocator,typename Type>
struct compliant_allocator_rebind_to
{
typedef typename rebinder<Allocator>::
BOOST_NESTED_TEMPLATE result<Type>::other type;
};



template<typename Allocator,typename Type>
struct rebind_to:
boost::detail::if_true<
is_partial_std_allocator<Allocator>::value
>::template then<
partial_std_allocator_rebind_to<Allocator,Type>,
compliant_allocator_rebind_to<Allocator,Type>
>::type
{
};



template<typename Type>
void construct(void* p,const Type& t)
{
new (p) Type(t);
}

#if BOOST_WORKAROUND(BOOST_MSVC,BOOST_TESTED_AT(1500))


#pragma warning(push)
#pragma warning(disable:4100)
#endif

template<typename Type>
void destroy(const Type* p)
{

#if BOOST_WORKAROUND(__SUNPRO_CC,BOOST_TESTED_AT(0x590))
const_cast<Type*>(p)->~Type();
#else
p->~Type();
#endif

}

#if BOOST_WORKAROUND(BOOST_MSVC,BOOST_TESTED_AT(1500))
#pragma warning(pop)
#endif

} 

} 

} 

#endif
