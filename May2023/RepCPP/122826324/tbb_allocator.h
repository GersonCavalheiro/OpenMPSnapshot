

#ifndef __TBB_tbb_allocator_H
#define __TBB_tbb_allocator_H

#include "tbb_stddef.h"
#include <new>
#if __TBB_CPP11_RVALUE_REF_PRESENT && !__TBB_CPP11_STD_FORWARD_BROKEN
#include <utility> 
#endif

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <cstring>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

namespace tbb {

namespace internal {


void __TBB_EXPORTED_FUNC deallocate_via_handler_v3( void *p );


void* __TBB_EXPORTED_FUNC allocate_via_handler_v3( size_t n );

bool __TBB_EXPORTED_FUNC is_malloc_used_v3();
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4100)
#endif


template<typename T>
class tbb_allocator {
public:
typedef typename internal::allocator_type<T>::value_type value_type;
typedef value_type* pointer;
typedef const value_type* const_pointer;
typedef value_type& reference;
typedef const value_type& const_reference;
typedef size_t size_type;
typedef ptrdiff_t difference_type;
template<typename U> struct rebind {
typedef tbb_allocator<U> other;
};

enum malloc_type {
scalable, 
standard
};

tbb_allocator() throw() {}
tbb_allocator( const tbb_allocator& ) throw() {}
template<typename U> tbb_allocator(const tbb_allocator<U>&) throw() {}

pointer address(reference x) const {return &x;}
const_pointer address(const_reference x) const {return &x;}

pointer allocate( size_type n, const void*  = 0) {
return pointer(internal::allocate_via_handler_v3( n * sizeof(value_type) ));
}

void deallocate( pointer p, size_type ) {
internal::deallocate_via_handler_v3(p);        
}

size_type max_size() const throw() {
size_type max = static_cast<size_type>(-1) / sizeof (value_type);
return (max > 0 ? max : 1);
}

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT
template<typename... Args>
void construct(pointer p, Args&&... args)
#if __TBB_CPP11_STD_FORWARD_BROKEN
{ ::new((void *)p) T((args)...); }
#else
{ ::new((void *)p) T(std::forward<Args>(args)...); }
#endif
#else 
void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
#endif 

void destroy( pointer p ) {p->~value_type();}

static malloc_type allocator_type() {
return internal::is_malloc_used_v3() ? standard : scalable;
}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif 


template<> 
class tbb_allocator<void> {
public:
typedef void* pointer;
typedef const void* const_pointer;
typedef void value_type;
template<typename U> struct rebind {
typedef tbb_allocator<U> other;
};
};

template<typename T, typename U>
inline bool operator==( const tbb_allocator<T>&, const tbb_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const tbb_allocator<T>&, const tbb_allocator<U>& ) {return false;}


template <typename T, template<typename X> class Allocator = tbb_allocator>
class zero_allocator : public Allocator<T>
{
public:
typedef Allocator<T> base_allocator_type;
typedef typename base_allocator_type::value_type value_type;
typedef typename base_allocator_type::pointer pointer;
typedef typename base_allocator_type::const_pointer const_pointer;
typedef typename base_allocator_type::reference reference;
typedef typename base_allocator_type::const_reference const_reference;
typedef typename base_allocator_type::size_type size_type;
typedef typename base_allocator_type::difference_type difference_type;
template<typename U> struct rebind {
typedef zero_allocator<U, Allocator> other;
};

zero_allocator() throw() { }
zero_allocator(const zero_allocator &a) throw() : base_allocator_type( a ) { }
template<typename U>
zero_allocator(const zero_allocator<U> &a) throw() : base_allocator_type( Allocator<U>( a ) ) { }

pointer allocate(const size_type n, const void *hint = 0 ) {
pointer ptr = base_allocator_type::allocate( n, hint );
std::memset( ptr, 0, n * sizeof(value_type) );
return ptr;
}
};


template<template<typename T> class Allocator> 
class zero_allocator<void, Allocator> : public Allocator<void> {
public:
typedef Allocator<void> base_allocator_type;
typedef typename base_allocator_type::value_type value_type;
typedef typename base_allocator_type::pointer pointer;
typedef typename base_allocator_type::const_pointer const_pointer;
template<typename U> struct rebind {
typedef zero_allocator<U, Allocator> other;
};
};

template<typename T1, template<typename X1> class B1, typename T2, template<typename X2> class B2>
inline bool operator==( const zero_allocator<T1,B1> &a, const zero_allocator<T2,B2> &b) {
return static_cast< B1<T1> >(a) == static_cast< B2<T2> >(b);
}
template<typename T1, template<typename X1> class B1, typename T2, template<typename X2> class B2>
inline bool operator!=( const zero_allocator<T1,B1> &a, const zero_allocator<T2,B2> &b) {
return static_cast< B1<T1> >(a) != static_cast< B2<T2> >(b);
}

} 

#endif 
