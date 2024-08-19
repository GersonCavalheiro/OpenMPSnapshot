

#include "tbb/tbb_config.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/tbb_allocator.h"
#include "tbb/tbb_exception.h"
#include "tbb_misc.h"
#include "dynamic_link.h"
#include <cstdlib>

#if _WIN32||_WIN64
#include "tbb/machine/windows_api.h"
#else
#include <dlfcn.h>
#endif 

#if __TBB_WEAK_SYMBOLS_PRESENT

#pragma weak scalable_malloc
#pragma weak scalable_free
#pragma weak scalable_aligned_malloc
#pragma weak scalable_aligned_free

extern "C" {
void* scalable_malloc( size_t );
void  scalable_free( void* );
void* scalable_aligned_malloc( size_t, size_t );
void  scalable_aligned_free( void* );
}

#endif 

namespace tbb {

namespace internal {

static void* DummyMalloc( size_t size );

static void DummyFree( void * ptr );

static void* (*MallocHandler)( size_t size ) = &DummyMalloc;

static void (*FreeHandler)( void* pointer ) = &DummyFree;

static void* dummy_padded_allocate( size_t bytes, size_t alignment );

static void dummy_padded_free( void * ptr );

static void* padded_allocate( size_t bytes, size_t alignment );

static void padded_free( void* p );

static void* (*padded_allocate_handler)( size_t bytes, size_t alignment ) = &dummy_padded_allocate;

static void (*padded_free_handler)( void* p ) = &dummy_padded_free;

static const dynamic_link_descriptor MallocLinkTable[] = {
DLD(scalable_malloc, MallocHandler),
DLD(scalable_free, FreeHandler),
DLD(scalable_aligned_malloc, padded_allocate_handler),
DLD(scalable_aligned_free, padded_free_handler),
};


#if TBB_USE_DEBUG
#define DEBUG_SUFFIX "_debug"
#else
#define DEBUG_SUFFIX
#endif 

#if _WIN32||_WIN64
#define MALLOCLIB_NAME "tbbmalloc" DEBUG_SUFFIX ".dll"
#elif __APPLE__
#define MALLOCLIB_NAME "libtbbmalloc" DEBUG_SUFFIX ".dylib"
#elif __FreeBSD__ || __NetBSD__ || __OpenBSD__ || __sun || _AIX || __ANDROID__
#define MALLOCLIB_NAME "libtbbmalloc" DEBUG_SUFFIX ".so"
#elif __linux__  
#define MALLOCLIB_NAME "libtbbmalloc" DEBUG_SUFFIX  __TBB_STRING(.so.TBB_COMPATIBLE_INTERFACE_VERSION)
#else
#error Unknown OS
#endif


void initialize_handler_pointers() {
__TBB_ASSERT( MallocHandler==&DummyMalloc, NULL );
bool success = dynamic_link( MALLOCLIB_NAME, MallocLinkTable, 4 );
if( !success ) {
FreeHandler = &std::free;
MallocHandler = &std::malloc;
padded_allocate_handler = &padded_allocate;
padded_free_handler = &padded_free;
}
#if !__TBB_RML_STATIC
PrintExtraVersionInfo( "ALLOCATOR", success?"scalable_malloc":"malloc" );
#endif
}

static tbb::atomic<do_once_state> initialization_state;
void initialize_cache_aligned_allocator() {
atomic_do_once( &initialize_handler_pointers, initialization_state );
}

static void* DummyMalloc( size_t size ) {
initialize_cache_aligned_allocator();
__TBB_ASSERT( MallocHandler!=&DummyMalloc, NULL );
return (*MallocHandler)( size );
}

static void DummyFree( void * ptr ) {
initialize_cache_aligned_allocator();
__TBB_ASSERT( FreeHandler!=&DummyFree, NULL );
(*FreeHandler)( ptr );
}

static void* dummy_padded_allocate( size_t bytes, size_t alignment ) {
initialize_cache_aligned_allocator();
__TBB_ASSERT( padded_allocate_handler!=&dummy_padded_allocate, NULL );
return (*padded_allocate_handler)(bytes, alignment);
}

static void dummy_padded_free( void * ptr ) {
initialize_cache_aligned_allocator();
__TBB_ASSERT( padded_free_handler!=&dummy_padded_free, NULL );
(*padded_free_handler)( ptr );
}

static size_t NFS_LineSize = 128;

size_t NFS_GetLineSize() {
return NFS_LineSize;
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( disable: 4146 4706 )
#endif

void* NFS_Allocate( size_t n, size_t element_size, void*  ) {
const size_t nfs_cache_line_size = NFS_LineSize;
__TBB_ASSERT( nfs_cache_line_size <= NFS_MaxLineSize, "illegal value for NFS_LineSize" );
__TBB_ASSERT( is_power_of_two(nfs_cache_line_size), "must be power of two" );
size_t bytes = n*element_size;

if (bytes<n || bytes+nfs_cache_line_size<bytes) {
throw_exception(eid_bad_alloc);
}
if (bytes==0) bytes = 1;

void* result = (*padded_allocate_handler)( bytes, nfs_cache_line_size );
if (!result)
throw_exception(eid_bad_alloc);

__TBB_ASSERT( is_aligned(result, nfs_cache_line_size), "The address returned isn't aligned to cache line size" );
return result;
}

void NFS_Free( void* p ) {
(*padded_free_handler)( p );
}

static void* padded_allocate( size_t bytes, size_t alignment ) {
unsigned char* result = NULL;
unsigned char* base = (unsigned char*)std::malloc(alignment+bytes);
if( base ) {
result = (unsigned char*)((uintptr_t)(base+alignment)&-alignment);
((uintptr_t*)result)[-1] = uintptr_t(base);
}
return result;
}

static void padded_free( void* p ) {
if( p ) {
__TBB_ASSERT( (uintptr_t)p>=0x4096, "attempt to free block not obtained from cache_aligned_allocator" );
unsigned char* base = ((unsigned char**)p)[-1];
__TBB_ASSERT( (void*)((uintptr_t)(base+NFS_LineSize)&-NFS_LineSize)==p, "not allocated by NFS_Allocate?" );
std::free(base);
}
}

void* __TBB_EXPORTED_FUNC allocate_via_handler_v3( size_t n ) {
void* result = (*MallocHandler) (n);
if (!result) {
throw_exception(eid_bad_alloc);
}
return result;
}

void __TBB_EXPORTED_FUNC deallocate_via_handler_v3( void *p ) {
if( p ) {
(*FreeHandler)( p );
}
}

bool __TBB_EXPORTED_FUNC is_malloc_used_v3() {
if (MallocHandler == &DummyMalloc) {
void* void_ptr = (*MallocHandler)(1);
(*FreeHandler)(void_ptr);
}
__TBB_ASSERT( MallocHandler!=&DummyMalloc && FreeHandler!=&DummyFree, NULL );
__TBB_ASSERT( !(((void*)MallocHandler==(void*)&std::malloc) ^ ((void*)FreeHandler==(void*)&std::free)),
"Both shim pointers must refer to routines from the same package (either TBB or CRT)" );
return (void*)MallocHandler == (void*)&std::malloc;
}

} 

} 
