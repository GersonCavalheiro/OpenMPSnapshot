

#ifndef __TBB_memory_pool_H
#define __TBB_memory_pool_H

#if !TBB_PREVIEW_MEMORY_POOL
#error Set TBB_PREVIEW_MEMORY_POOL to include memory_pool.h
#endif


#include "scalable_allocator.h"
#include <new> 
#include <stdexcept> 
#include <string>
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#include <utility> 
#endif

#if __TBB_EXTRA_DEBUG
#define __TBBMALLOC_ASSERT ASSERT
#else
#define __TBBMALLOC_ASSERT(a,b) ((void)0)
#endif

namespace tbb {
namespace interface6 {
namespace internal {

class pool_base : tbb::internal::no_copy {
public:
void recycle() { rml::pool_reset(my_pool); }

void *malloc(size_t size) { return rml::pool_malloc(my_pool, size); }

void free(void* ptr) { rml::pool_free(my_pool, ptr); }

void *realloc(void* ptr, size_t size) {
return rml::pool_realloc(my_pool, ptr, size);
}

protected:
void destroy() { rml::pool_destroy(my_pool); }

rml::MemoryPool *my_pool;
};

} 

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4100)
#endif


template<typename T, typename P = internal::pool_base>
class memory_pool_allocator {
protected:
typedef P pool_type;
pool_type *my_pool;
template<typename U, typename R>
friend class memory_pool_allocator;
template<typename V, typename U, typename R>
friend bool operator==( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
template<typename V, typename U, typename R>
friend bool operator!=( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
public:
typedef typename tbb::internal::allocator_type<T>::value_type value_type;
typedef value_type* pointer;
typedef const value_type* const_pointer;
typedef value_type& reference;
typedef const value_type& const_reference;
typedef size_t size_type;
typedef ptrdiff_t difference_type;
template<typename U> struct rebind {
typedef memory_pool_allocator<U, P> other;
};

explicit memory_pool_allocator(pool_type &pool) throw() : my_pool(&pool) {}
memory_pool_allocator(const memory_pool_allocator& src) throw() : my_pool(src.my_pool) {}
template<typename U>
memory_pool_allocator(const memory_pool_allocator<U,P>& src) throw() : my_pool(src.my_pool) {}

pointer address(reference x) const { return &x; }
const_pointer address(const_reference x) const { return &x; }

pointer allocate( size_type n, const void*  = 0) {
pointer p = static_cast<pointer>( my_pool->malloc( n*sizeof(value_type) ) );
if (!p)
tbb::internal::throw_exception(std::bad_alloc());
return p;
}
void deallocate( pointer p, size_type ) {
my_pool->free(p);
}
size_type max_size() const throw() {
size_type max = static_cast<size_type>(-1) / sizeof (value_type);
return (max > 0 ? max : 1);
}
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
template<typename U, typename... Args>
void construct(U *p, Args&&... args)
{ ::new((void *)p) U(std::forward<Args>(args)...); }
#else 
#if __TBB_CPP11_RVALUE_REF_PRESENT
void construct( pointer p, value_type&& value ) {::new((void*)(p)) value_type(std::move(value));}
#endif
void construct( pointer p, const value_type& value ) { ::new((void*)(p)) value_type(value); }
#endif 

void destroy( pointer p ) { p->~value_type(); }

};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif 


template<typename P>
class memory_pool_allocator<void, P> {
public:
typedef P pool_type;
typedef void* pointer;
typedef const void* const_pointer;
typedef void value_type;
template<typename U> struct rebind {
typedef memory_pool_allocator<U, P> other;
};

explicit memory_pool_allocator( pool_type &pool) throw() : my_pool(&pool) {}
memory_pool_allocator( const memory_pool_allocator& src) throw() : my_pool(src.my_pool) {}
template<typename U>
memory_pool_allocator(const memory_pool_allocator<U,P>& src) throw() : my_pool(src.my_pool) {}

protected:
pool_type *my_pool;
template<typename U, typename R>
friend class memory_pool_allocator;
template<typename V, typename U, typename R>
friend bool operator==( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
template<typename V, typename U, typename R>
friend bool operator!=( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
};

template<typename T, typename U, typename P>
inline bool operator==( const memory_pool_allocator<T,P>& a, const memory_pool_allocator<U,P>& b) {return a.my_pool==b.my_pool;}

template<typename T, typename U, typename P>
inline bool operator!=( const memory_pool_allocator<T,P>& a, const memory_pool_allocator<U,P>& b) {return a.my_pool!=b.my_pool;}


template <typename Alloc>
class memory_pool : public internal::pool_base {
Alloc my_alloc; 
static void *allocate_request(intptr_t pool_id, size_t & bytes);
static int deallocate_request(intptr_t pool_id, void*, size_t raw_bytes);

public:
explicit memory_pool(const Alloc &src = Alloc());

~memory_pool() { destroy(); } 

};

class fixed_pool : public internal::pool_base {
void *my_buffer;
size_t my_size;
inline static void *allocate_request(intptr_t pool_id, size_t & bytes);

public:
inline fixed_pool(void *buf, size_t size);
~fixed_pool() { destroy(); }
};


template <typename Alloc>
memory_pool<Alloc>::memory_pool(const Alloc &src) : my_alloc(src) {
rml::MemPoolPolicy args(allocate_request, deallocate_request,
sizeof(typename Alloc::value_type));
rml::MemPoolError res = rml::pool_create_v1(intptr_t(this), &args, &my_pool);
if (res!=rml::POOL_OK)
tbb::internal::throw_exception(std::runtime_error("Can't create pool"));
}
template <typename Alloc>
void *memory_pool<Alloc>::allocate_request(intptr_t pool_id, size_t & bytes) {
memory_pool<Alloc> &self = *reinterpret_cast<memory_pool<Alloc>*>(pool_id);
const size_t unit_size = sizeof(typename Alloc::value_type);
__TBBMALLOC_ASSERT( 0 == bytes%unit_size, NULL);
void *ptr;
__TBB_TRY { ptr = self.my_alloc.allocate( bytes/unit_size ); }
__TBB_CATCH(...) { return 0; }
return ptr;
}
#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
#pragma warning (push)
#pragma warning (disable: 4702)
#endif
template <typename Alloc>
int memory_pool<Alloc>::deallocate_request(intptr_t pool_id, void* raw_ptr, size_t raw_bytes) {
memory_pool<Alloc> &self = *reinterpret_cast<memory_pool<Alloc>*>(pool_id);
const size_t unit_size = sizeof(typename Alloc::value_type);
__TBBMALLOC_ASSERT( 0 == raw_bytes%unit_size, NULL);
self.my_alloc.deallocate( static_cast<typename Alloc::value_type*>(raw_ptr), raw_bytes/unit_size );
return 0;
}
#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
#pragma warning (pop)
#endif
inline fixed_pool::fixed_pool(void *buf, size_t size) : my_buffer(buf), my_size(size) {
if (!buf || !size)
tbb::internal::throw_exception(std::invalid_argument("Zero in parameter is invalid"));
rml::MemPoolPolicy args(allocate_request, 0, size, true);
rml::MemPoolError res = rml::pool_create_v1(intptr_t(this), &args, &my_pool);
if (res!=rml::POOL_OK)
tbb::internal::throw_exception(std::runtime_error("Can't create pool"));
}
inline void *fixed_pool::allocate_request(intptr_t pool_id, size_t & bytes) {
fixed_pool &self = *reinterpret_cast<fixed_pool*>(pool_id);
__TBBMALLOC_ASSERT(0 != self.my_size, "The buffer must not be used twice.");
bytes = self.my_size;
self.my_size = 0; 
return self.my_buffer;
}

} 
using interface6::memory_pool_allocator;
using interface6::memory_pool;
using interface6::fixed_pool;
} 

#undef __TBBMALLOC_ASSERT
#endif
