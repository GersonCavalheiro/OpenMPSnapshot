#ifndef BOOST_CONTAINER_DETAIL_CONTAINER_REBIND_HPP
#define BOOST_CONTAINER_DETAIL_CONTAINER_REBIND_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/allocator_traits.hpp>
#include <boost/container/container_fwd.hpp>


namespace boost {
namespace container {
namespace dtl {

template <class Cont, class U>
struct container_rebind;

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

template <template <class, class, class...> class Cont, typename V, typename A, class... An, class U>
struct container_rebind<Cont<V, A, An...>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, An...> type;
};

template <template <class, class> class Cont, typename V, typename A, class U>
struct container_rebind<Cont<V, A>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type> type;
};

template <template <class> class Cont, typename V, class U>
struct container_rebind<Cont<V>, U>
{
typedef Cont<U> type;
};

#else 

template <template <class> class Cont  
, typename V
, class U>
struct container_rebind<Cont<V>, U>
{
typedef Cont<U> type;
};

template <template <class, class> class Cont  
, typename V, typename A
, class U>
struct container_rebind<Cont<V, A>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type> type;
};

template <template <class, class, class> class Cont  
, typename V, typename A, class P0
, class U>
struct container_rebind<Cont<V, A, P0>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0> type;
};

template <template <class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1
, class U>
struct container_rebind<Cont<V, A, P0, P1>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1> type;
};

template <template <class, class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1, class P2
, class U>
struct container_rebind<Cont<V, A, P0, P1, P2>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1, P2> type;
};

template <template <class, class, class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1, class P2, class P3
, class U>
struct container_rebind<Cont<V, A, P0, P1, P2, P3>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1, P2, P3> type;
};

template <template <class, class, class, class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1, class P2, class P3, class P4
, class U>
struct container_rebind<Cont<V, A, P0, P1, P2, P3, P4>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1, P2, P3, P4> type;
};

template <template <class, class, class, class, class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1, class P2, class P3, class P4, class P5
, class U>
struct container_rebind<Cont<V, A, P0, P1, P2, P3, P4, P5>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1, P2, P3, P4, P5> type;
};

template <template <class, class, class, class, class, class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1, class P2, class P3, class P4, class P5, class P6
, class U>
struct container_rebind<Cont<V, A, P0, P1, P2, P3, P4, P5, P6>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1, P2, P3, P4, P5, P6> type;
};

template <template <class, class, class, class, class, class, class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1, class P2, class P3, class P4, class P5, class P6, class P7
, class U>
struct container_rebind<Cont<V, A, P0, P1, P2, P3, P4, P5, P6, P7>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1, P2, P3, P4, P5, P6, P7> type;
};

template <template <class, class, class, class, class, class, class, class, class, class, class> class Cont  
, typename V, typename A, class P0, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8
, class U>
struct container_rebind<Cont<V, A, P0, P1, P2, P3, P4, P5, P6, P7, P8>, U>
{
typedef Cont<U, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, P0, P1, P2, P3, P4, P5, P6, P7, P8> type;
};

#endif   


template <typename V, std::size_t N, typename A, typename O, class U>
struct container_rebind<small_vector<V, N, A, O>, U>
{
typedef small_vector<U, N, typename allocator_traits<typename real_allocator<V, A>::type>::template portable_rebind_alloc<U>::type, O> type;
};

template <typename V, std::size_t N, typename O, class U>
struct container_rebind<static_vector<V, N, O>, U>
{
typedef static_vector<U, N, O> type;
};

}  
}  
}  

#endif   
