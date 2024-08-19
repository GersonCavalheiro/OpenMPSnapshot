

#ifndef __TBB_scalable_allocator_H
#define __TBB_scalable_allocator_H


#include <stddef.h> 
#if !_MSC_VER
#include <stdint.h> 
#endif

#if !defined(__cplusplus) && __ICC==1100
#pragma warning (push)
#pragma warning (disable: 991)
#endif

#ifdef __cplusplus
extern "C" {
#endif 

#if _MSC_VER >= 1400
#define __TBB_EXPORTED_FUNC   __cdecl
#else
#define __TBB_EXPORTED_FUNC
#endif


void * __TBB_EXPORTED_FUNC scalable_malloc (size_t size);


void   __TBB_EXPORTED_FUNC scalable_free (void* ptr);


void * __TBB_EXPORTED_FUNC scalable_realloc (void* ptr, size_t size);


void * __TBB_EXPORTED_FUNC scalable_calloc (size_t nobj, size_t size);


int __TBB_EXPORTED_FUNC scalable_posix_memalign (void** memptr, size_t alignment, size_t size);


void * __TBB_EXPORTED_FUNC scalable_aligned_malloc (size_t size, size_t alignment);


void * __TBB_EXPORTED_FUNC scalable_aligned_realloc (void* ptr, size_t size, size_t alignment);


void __TBB_EXPORTED_FUNC scalable_aligned_free (void* ptr);


size_t __TBB_EXPORTED_FUNC scalable_msize (void* ptr);


typedef enum {
TBBMALLOC_OK,
TBBMALLOC_INVALID_PARAM,
TBBMALLOC_UNSUPPORTED,
TBBMALLOC_NO_MEMORY,
TBBMALLOC_NO_EFFECT
} ScalableAllocationResult;


typedef enum {
TBBMALLOC_USE_HUGE_PAGES,  

USE_HUGE_PAGES = TBBMALLOC_USE_HUGE_PAGES,

TBBMALLOC_SET_SOFT_HEAP_LIMIT,

TBBMALLOC_SET_HUGE_SIZE_THRESHOLD
} AllocationModeParam;


int __TBB_EXPORTED_FUNC scalable_allocation_mode(int param, intptr_t value);

typedef enum {

TBBMALLOC_CLEAN_ALL_BUFFERS,

TBBMALLOC_CLEAN_THREAD_BUFFERS
} ScalableAllocationCmd;


int __TBB_EXPORTED_FUNC scalable_allocation_command(int cmd, void *param);

#ifdef __cplusplus
} 
#endif 

#ifdef __cplusplus

namespace rml {
class MemoryPool;

typedef void *(*rawAllocType)(intptr_t pool_id, size_t &bytes);
typedef int   (*rawFreeType)(intptr_t pool_id, void* raw_ptr, size_t raw_bytes);



struct MemPoolPolicy {
enum {
TBBMALLOC_POOL_VERSION = 1
};

rawAllocType pAlloc;
rawFreeType  pFree;
size_t       granularity;
int          version;
unsigned     fixedPool : 1,
keepAllMemory : 1,
reserved : 30;

MemPoolPolicy(rawAllocType pAlloc_, rawFreeType pFree_,
size_t granularity_ = 0, bool fixedPool_ = false,
bool keepAllMemory_ = false) :
pAlloc(pAlloc_), pFree(pFree_), granularity(granularity_), version(TBBMALLOC_POOL_VERSION),
fixedPool(fixedPool_), keepAllMemory(keepAllMemory_),
reserved(0) {}
};

enum MemPoolError {
POOL_OK = TBBMALLOC_OK,
INVALID_POLICY = TBBMALLOC_INVALID_PARAM,
UNSUPPORTED_POLICY = TBBMALLOC_UNSUPPORTED,
NO_MEMORY = TBBMALLOC_NO_MEMORY,
NO_EFFECT = TBBMALLOC_NO_EFFECT
};

MemPoolError pool_create_v1(intptr_t pool_id, const MemPoolPolicy *policy,
rml::MemoryPool **pool);

bool  pool_destroy(MemoryPool* memPool);
void *pool_malloc(MemoryPool* memPool, size_t size);
void *pool_realloc(MemoryPool* memPool, void *object, size_t size);
void *pool_aligned_malloc(MemoryPool* mPool, size_t size, size_t alignment);
void *pool_aligned_realloc(MemoryPool* mPool, void *ptr, size_t size, size_t alignment);
bool  pool_reset(MemoryPool* memPool);
bool  pool_free(MemoryPool *memPool, void *object);
MemoryPool *pool_identify(void *object);
size_t pool_msize(MemoryPool *memPool, void *object);

} 

#include <new>      


#ifndef __TBB_NO_IMPLICIT_LINKAGE
#define __TBB_NO_IMPLICIT_LINKAGE 1
#include "tbb_stddef.h"
#undef  __TBB_NO_IMPLICIT_LINKAGE
#else
#include "tbb_stddef.h"
#endif

#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#include <utility> 
#endif

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
#include <memory_resource>
#endif

namespace tbb {

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4100)
#endif

namespace internal {

#if TBB_USE_EXCEPTIONS
template<typename E> __TBB_NOINLINE( void throw_exception(const E &e) );
#endif

template<typename E>
void throw_exception(const E &e) {
__TBB_THROW(e);
}

} 


template<typename T>
class scalable_allocator {
public:
typedef typename internal::allocator_type<T>::value_type value_type;
typedef value_type* pointer;
typedef const value_type* const_pointer;
typedef value_type& reference;
typedef const value_type& const_reference;
typedef size_t size_type;
typedef ptrdiff_t difference_type;
template<class U> struct rebind {
typedef scalable_allocator<U> other;
};

scalable_allocator() throw() {}
scalable_allocator( const scalable_allocator& ) throw() {}
template<typename U> scalable_allocator(const scalable_allocator<U>&) throw() {}

pointer address(reference x) const {return &x;}
const_pointer address(const_reference x) const {return &x;}

pointer allocate( size_type n, const void*  =0 ) {
pointer p = static_cast<pointer>( scalable_malloc( n * sizeof(value_type) ) );
if (!p)
internal::throw_exception(std::bad_alloc());
return p;
}

void deallocate( pointer p, size_type ) {
scalable_free( p );
}

size_type max_size() const throw() {
size_type absolutemax = static_cast<size_type>(-1) / sizeof (value_type);
return (absolutemax > 0 ? absolutemax : 1);
}
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
template<typename U, typename... Args>
void construct(U *p, Args&&... args)
{ ::new((void *)p) U(std::forward<Args>(args)...); }
#else 
#if __TBB_CPP11_RVALUE_REF_PRESENT
void construct( pointer p, value_type&& value ) { ::new((void*)(p)) value_type( std::move( value ) ); }
#endif
void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
#endif 
void destroy( pointer p ) {p->~value_type();}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif 


template<>
class scalable_allocator<void> {
public:
typedef void* pointer;
typedef const void* const_pointer;
typedef void value_type;
template<class U> struct rebind {
typedef scalable_allocator<U> other;
};
};

template<typename T, typename U>
inline bool operator==( const scalable_allocator<T>&, const scalable_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const scalable_allocator<T>&, const scalable_allocator<U>& ) {return false;}

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT

namespace internal {

class scalable_resource_impl : public std::pmr::memory_resource {
private:
void* do_allocate(size_t bytes, size_t alignment) override {
void* ptr = scalable_aligned_malloc( bytes, alignment );
if (!ptr) {
throw_exception(std::bad_alloc());
}
return ptr;
}

void do_deallocate(void* ptr, size_t , size_t ) override {
scalable_free(ptr);
}

bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
return this == &other ||
#if __TBB_USE_OPTIONAL_RTTI
dynamic_cast<const scalable_resource_impl*>(&other) != NULL;
#else
false;
#endif
}
};

} 

inline std::pmr::memory_resource* scalable_memory_resource() noexcept {
static tbb::internal::scalable_resource_impl scalable_res;
return &scalable_res;
}

#endif 

} 

#if _MSC_VER
#if (__TBB_BUILD || __TBBMALLOC_BUILD) && !defined(__TBBMALLOC_NO_IMPLICIT_LINKAGE)
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
#endif

#if !__TBBMALLOC_NO_IMPLICIT_LINKAGE
#ifdef _DEBUG
#pragma comment(lib, "tbbmalloc_debug.lib")
#else
#pragma comment(lib, "tbbmalloc.lib")
#endif
#endif


#endif

#endif 

#if !defined(__cplusplus) && __ICC==1100
#pragma warning (pop)
#endif 

#endif 
