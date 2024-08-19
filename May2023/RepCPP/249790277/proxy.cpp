

#include "proxy.h"
#include "tbb/tbb_config.h"

#if !defined(__EXCEPTIONS) && !defined(_CPPUNWIND) && !defined(__SUNPRO_CC)
#if TBB_USE_EXCEPTIONS
#error Compilation settings do not support exception handling. Please do not set TBB_USE_EXCEPTIONS macro or set it to 0.
#elif !defined(TBB_USE_EXCEPTIONS)
#define TBB_USE_EXCEPTIONS 0
#endif
#elif !defined(TBB_USE_EXCEPTIONS)
#define TBB_USE_EXCEPTIONS 1
#endif

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED

#ifndef __THROW
#define __THROW
#endif



#include <string.h> 
#include <unistd.h> 

static long memoryPageSize;

static inline void initPageSize()
{
memoryPageSize = sysconf(_SC_PAGESIZE);
}

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED
#include "Customize.h" 
#include <dlfcn.h>
#include <malloc.h>    


extern "C" void *__TBB_malloc_proxy(size_t) __attribute__ ((alias ("malloc")));

static void *orig_msize;

#elif MALLOC_ZONE_OVERLOAD_ENABLED

#include "proxy_overload_osx.h"

#endif 

static void *orig_free,
*orig_realloc;

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED
#define ZONE_ARG
#define PREFIX(name) name

static void *orig_libc_free,
*orig_libc_realloc;

static intptr_t origFuncSearched;

inline void InitOrigPointers()
{
if (!origFuncSearched) {
orig_free = dlsym(RTLD_NEXT, "free");
orig_realloc = dlsym(RTLD_NEXT, "realloc");
orig_msize = dlsym(RTLD_NEXT, "malloc_usable_size");
orig_libc_free = dlsym(RTLD_NEXT, "__libc_free");
orig_libc_realloc = dlsym(RTLD_NEXT, "__libc_realloc");

FencedStore(origFuncSearched, 1);
}
}


extern "C" {
#elif MALLOC_ZONE_OVERLOAD_ENABLED

#define ZONE_ARG struct _malloc_zone_t *,
#define PREFIX(name) impl_##name
inline void InitOrigPointers() {}

#endif 

void *PREFIX(malloc)(ZONE_ARG size_t size) __THROW
{
return scalable_malloc(size);
}

void *PREFIX(calloc)(ZONE_ARG size_t num, size_t size) __THROW
{
return scalable_calloc(num, size);
}

void PREFIX(free)(ZONE_ARG void *object) __THROW
{
InitOrigPointers();
__TBB_malloc_safer_free(object, (void (*)(void*))orig_free);
}

void *PREFIX(realloc)(ZONE_ARG void* ptr, size_t sz) __THROW
{
InitOrigPointers();
return __TBB_malloc_safer_realloc(ptr, sz, orig_realloc);
}


void *PREFIX(memalign)(ZONE_ARG size_t alignment, size_t size) __THROW
{
return scalable_aligned_malloc(size, alignment);
}


void *PREFIX(valloc)(ZONE_ARG size_t size) __THROW
{
if (! memoryPageSize) initPageSize();

return scalable_aligned_malloc(size, memoryPageSize);
}

#undef ZONE_ARG
#undef PREFIX

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED

#if __ANDROID__
size_t malloc_usable_size(const void *ptr) __THROW
#else
size_t malloc_usable_size(void *ptr) __THROW
#endif
{
InitOrigPointers();
return __TBB_malloc_safer_msize(const_cast<void*>(ptr), (size_t (*)(void*))orig_msize);
}

int posix_memalign(void **memptr, size_t alignment, size_t size) __THROW
{
return scalable_posix_memalign(memptr, alignment, size);
}


void *pvalloc(size_t size) __THROW
{
if (! memoryPageSize) initPageSize();
size = size? ((size-1) | (memoryPageSize-1)) + 1 : memoryPageSize;

return scalable_aligned_malloc(size, memoryPageSize);
}

int mallopt(int , int ) __THROW
{
return 1;
}

struct mallinfo mallinfo() __THROW
{
struct mallinfo m;
memset(&m, 0, sizeof(struct mallinfo));

return m;
}

#if __ANDROID__
size_t dlmalloc_usable_size(const void *ptr) __attribute__ ((alias ("malloc_usable_size")));
#else 
void *aligned_alloc(size_t alignment, size_t size) __attribute__ ((alias ("memalign")));
void *__libc_malloc(size_t size) __attribute__ ((alias ("malloc")));
void *__libc_calloc(size_t num, size_t size) __attribute__ ((alias ("calloc")));
void *__libc_memalign(size_t alignment, size_t size) __attribute__ ((alias ("memalign")));
void *__libc_pvalloc(size_t size) __attribute__ ((alias ("pvalloc")));
void *__libc_valloc(size_t size) __attribute__ ((alias ("valloc")));

void __libc_free(void *ptr)
{
InitOrigPointers();
__TBB_malloc_safer_free(ptr, (void (*)(void*))orig_libc_free);
}

void *__libc_realloc(void *ptr, size_t size)
{
InitOrigPointers();
return __TBB_malloc_safer_realloc(ptr, size, orig_libc_realloc);
}
#endif 

} 



#include <new>

void * operator new(size_t sz) throw (std::bad_alloc) {
void *res = scalable_malloc(sz);
#if TBB_USE_EXCEPTIONS
if (NULL == res)
throw std::bad_alloc();
#endif 
return res;
}
void* operator new[](size_t sz) throw (std::bad_alloc) {
void *res = scalable_malloc(sz);
#if TBB_USE_EXCEPTIONS
if (NULL == res)
throw std::bad_alloc();
#endif 
return res;
}
void operator delete(void* ptr) throw() {
InitOrigPointers();
__TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}
void operator delete[](void* ptr) throw() {
InitOrigPointers();
__TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}
void* operator new(size_t sz, const std::nothrow_t&) throw() {
return scalable_malloc(sz);
}
void* operator new[](std::size_t sz, const std::nothrow_t&) throw() {
return scalable_malloc(sz);
}
void operator delete(void* ptr, const std::nothrow_t&) throw() {
InitOrigPointers();
__TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}
void operator delete[](void* ptr, const std::nothrow_t&) throw() {
InitOrigPointers();
__TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}

#endif 
#endif 


#ifdef _WIN32
#include <windows.h>

#if !__TBB_WIN8UI_SUPPORT

#include <stdio.h>
#include "tbb_function_replacement.h"
#include "shared_utils.h"

void __TBB_malloc_safer_delete( void *ptr)
{
__TBB_malloc_safer_free( ptr, NULL );
}

void* safer_aligned_malloc( size_t size, size_t alignment )
{
return scalable_aligned_malloc( size, alignment>sizeof(size_t*)?alignment:sizeof(size_t*) );
}

void* safer_expand( void *, size_t )
{
return NULL;
}

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(CRTLIB)                                             \
void (*orig_free_##CRTLIB)(void*);                                                                   \
void __TBB_malloc_safer_free_##CRTLIB(void *ptr)                                                     \
{                                                                                                    \
__TBB_malloc_safer_free( ptr, orig_free_##CRTLIB );                                              \
}                                                                                                    \
\
void (*orig__aligned_free_##CRTLIB)(void*);                                                          \
void __TBB_malloc_safer__aligned_free_##CRTLIB(void *ptr)                                            \
{                                                                                                    \
__TBB_malloc_safer_free( ptr, orig__aligned_free_##CRTLIB );                                     \
}                                                                                                    \
\
size_t (*orig__msize_##CRTLIB)(void*);                                                               \
size_t __TBB_malloc_safer__msize_##CRTLIB(void *ptr)                                                 \
{                                                                                                    \
return __TBB_malloc_safer_msize( ptr, orig__msize_##CRTLIB );                                    \
}                                                                                                    \
\
size_t (*orig__aligned_msize_##CRTLIB)(void*, size_t, size_t);                                       \
size_t __TBB_malloc_safer__aligned_msize_##CRTLIB( void *ptr, size_t alignment, size_t offset)       \
{                                                                                                    \
return __TBB_malloc_safer_aligned_msize( ptr, alignment, offset, orig__aligned_msize_##CRTLIB ); \
}                                                                                                    \
\
void* __TBB_malloc_safer_realloc_##CRTLIB( void *ptr, size_t size )                                  \
{                                                                                                    \
orig_ptrs func_ptrs = {orig_free_##CRTLIB, orig__msize_##CRTLIB};                                \
return __TBB_malloc_safer_realloc( ptr, size, &func_ptrs );                                      \
}                                                                                                    \
\
void* __TBB_malloc_safer__aligned_realloc_##CRTLIB( void *ptr, size_t size, size_t aligment )        \
{                                                                                                    \
orig_aligned_ptrs func_ptrs = {orig__aligned_free_##CRTLIB, orig__aligned_msize_##CRTLIB};       \
return __TBB_malloc_safer_aligned_realloc( ptr, size, aligment, &func_ptrs );                    \
}

const char* known_bytecodes[] = {
#if _WIN64
"4883EC284885C974",       
"4883EC284885C975",       
"4885C974375348",         
"E907000000CCCC",         
"C7442410000000008B",     
"E90B000000CCCC",         
"48895C24085748",         
"48894C24084883EC28BA",   
"4C894424184889542410",   
"48894C24084883EC2848",   
#if __TBB_OVERLOAD_OLD_MSVCR
"48895C2408574883EC3049", 
"4883EC384885C975",       
"4C8BC1488B0DA6E4040033", 
#endif
#else 
"8BFF558BEC8B",           
"8BFF558BEC83",           
"8BFF558BECFF",           
"8BFF558BEC51",           
"558BEC8B450885C074",     
"558BEC837D08000F",       
"558BEC837D08007419FF",   
"558BEC8B450885C075",     
"558BEC6A018B",           
"558BEC8B451050",         
"558BEC8B450850",         
"8BFF558BEC6A",           
#if __TBB_OVERLOAD_OLD_MSVCR
"6A1868********E8",       
"6A1C68********E8",       
#endif
#endif 
NULL
};

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,function_name,dbgsuffix) \
ReplaceFunctionWithStore( #CRT_VER #dbgsuffix ".dll", #function_name, \
(FUNCPTR)__TBB_malloc_safer_##function_name##_##CRT_VER##dbgsuffix, \
known_bytecodes, (FUNCPTR*)&orig_##function_name##_##CRT_VER##dbgsuffix );

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_NO_FALLBACK(CRT_VER,function_name,dbgsuffix) \
ReplaceFunctionWithStore( #CRT_VER #dbgsuffix ".dll", #function_name, \
(FUNCPTR)__TBB_malloc_safer_##function_name##_##CRT_VER##dbgsuffix, 0, NULL );

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_REDIRECT(CRT_VER,function_name,dest_func,dbgsuffix) \
ReplaceFunctionWithStore( #CRT_VER #dbgsuffix ".dll", #function_name, \
(FUNCPTR)__TBB_malloc_safer_##dest_func##_##CRT_VER##dbgsuffix, 0, NULL );

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_IMPL(CRT_VER,dbgsuffix)                             \
if (BytecodesAreKnown(#CRT_VER #dbgsuffix ".dll")) {                                          \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,free,dbgsuffix)                         \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,_msize,dbgsuffix)                       \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_NO_FALLBACK(CRT_VER,realloc,dbgsuffix)          \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,_aligned_free,dbgsuffix)                \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,_aligned_msize,dbgsuffix)               \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_NO_FALLBACK(CRT_VER,_aligned_realloc,dbgsuffix) \
} else                                                                                        \
SkipReplacement(#CRT_VER #dbgsuffix ".dll");

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_RELEASE(CRT_VER) __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_IMPL(CRT_VER,)
#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_DEBUG(CRT_VER) __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_IMPL(CRT_VER,d)

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(CRT_VER)     \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_RELEASE(CRT_VER) \
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_DEBUG(CRT_VER)

#if __TBB_OVERLOAD_OLD_MSVCR
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr70d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr70);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr71d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr71);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr80d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr80);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr90d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr90);
#endif
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr100d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr100);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr110d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr110);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr120d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr120);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(ucrtbase);




#include <new>

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable : 4290 )
#endif

void * operator_new(size_t sz) throw (std::bad_alloc) {
void *res = scalable_malloc(sz);
if (NULL == res) throw std::bad_alloc();
return res;
}
void* operator_new_arr(size_t sz) throw (std::bad_alloc) {
void *res = scalable_malloc(sz);
if (NULL == res) throw std::bad_alloc();
return res;
}
void operator_delete(void* ptr) throw() {
__TBB_malloc_safer_delete(ptr);
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif

void operator_delete_arr(void* ptr) throw() {
__TBB_malloc_safer_delete(ptr);
}
void* operator_new_t(size_t sz, const std::nothrow_t&) throw() {
return scalable_malloc(sz);
}
void* operator_new_arr_t(std::size_t sz, const std::nothrow_t&) throw() {
return scalable_malloc(sz);
}
void operator_delete_t(void* ptr, const std::nothrow_t&) throw() {
__TBB_malloc_safer_delete(ptr);
}
void operator_delete_arr_t(void* ptr, const std::nothrow_t&) throw() {
__TBB_malloc_safer_delete(ptr);
}

struct Module {
const char *name;
bool        doFuncReplacement; 
};

Module modules_to_replace[] = {
{"msvcr100d.dll", true},
{"msvcr100.dll", true},
{"msvcr110d.dll", true},
{"msvcr110.dll", true},
{"msvcr120d.dll", true},
{"msvcr120.dll", true},
{"ucrtbase.dll", true},
#if __TBB_OVERLOAD_OLD_MSVCR
{"msvcr90d.dll", true},
{"msvcr90.dll", true},
{"msvcr80d.dll", true},
{"msvcr80.dll", true},
{"msvcr70d.dll", true},
{"msvcr70.dll", true},
{"msvcr71d.dll", true},
{"msvcr71.dll", true},
#endif
#if __TBB_TODO
{"msvcrtd.dll", true},
{"msvcrt.dll", true},
#endif
};



typedef struct FRData_t {
const char *_func;
FUNCPTR _fptr;
FRR_ON_ERROR _on_error;
} FRDATA;

FRDATA c_routines_to_replace[] = {
{ "malloc",  (FUNCPTR)scalable_malloc, FRR_FAIL },
{ "calloc",  (FUNCPTR)scalable_calloc, FRR_FAIL },
{ "_aligned_malloc",  (FUNCPTR)safer_aligned_malloc, FRR_FAIL },
{ "_expand",  (FUNCPTR)safer_expand, FRR_IGNORE },
};

FRDATA cxx_routines_to_replace[] = {
#if _WIN64
{ "??2@YAPEAX_K@Z", (FUNCPTR)operator_new, FRR_FAIL },
{ "??_U@YAPEAX_K@Z", (FUNCPTR)operator_new_arr, FRR_FAIL },
{ "??3@YAXPEAX@Z", (FUNCPTR)operator_delete, FRR_FAIL },
{ "??_V@YAXPEAX@Z", (FUNCPTR)operator_delete_arr, FRR_FAIL },
#else
{ "??2@YAPAXI@Z", (FUNCPTR)operator_new, FRR_FAIL },
{ "??_U@YAPAXI@Z", (FUNCPTR)operator_new_arr, FRR_FAIL },
{ "??3@YAXPAX@Z", (FUNCPTR)operator_delete, FRR_FAIL },
{ "??_V@YAXPAX@Z", (FUNCPTR)operator_delete_arr, FRR_FAIL },
#endif
{ "??2@YAPAXIABUnothrow_t@std@@@Z", (FUNCPTR)operator_new_t, FRR_IGNORE },
{ "??_U@YAPAXIABUnothrow_t@std@@@Z", (FUNCPTR)operator_new_arr_t, FRR_IGNORE }
};

#ifndef UNICODE
typedef char unicode_char_t;
#define WCHAR_SPEC "%s"
#else
typedef wchar_t unicode_char_t;
#define WCHAR_SPEC "%ls"
#endif

bool BytecodesAreKnown(const unicode_char_t *dllName)
{
const char *funcName[] = {"free", "_msize", "_aligned_free", "_aligned_msize", 0};
HMODULE module = GetModuleHandle(dllName);

if (!module)
return false;
for (int i=0; funcName[i]; i++)
if (! IsPrologueKnown(module, funcName[i], known_bytecodes)) {
fprintf(stderr, "TBBmalloc: skip allocation functions replacement in " WCHAR_SPEC
": unknown prologue for function " WCHAR_SPEC "\n", dllName, funcName[i]);
return false;
}
return true;
}

void SkipReplacement(const unicode_char_t *dllName)
{
#ifndef UNICODE
const char *dllStr = dllName;
#else
const size_t sz = 128; 

char buffer[sz];
size_t real_sz;
char *dllStr = buffer;

errno_t ret = wcstombs_s(&real_sz, dllStr, sz, dllName, sz-1);
__TBB_ASSERT(!ret, "Dll name conversion failed")
#endif

for (size_t i=0; i<arrayLength(modules_to_replace); i++)
if (!strcmp(modules_to_replace[i].name, dllStr)) {
modules_to_replace[i].doFuncReplacement = false;
break;
}
}

void ReplaceFunctionWithStore( const unicode_char_t *dllName, const char *funcName, FUNCPTR newFunc, const char ** opcodes, FUNCPTR* origFunc,  FRR_ON_ERROR on_error = FRR_FAIL )
{
FRR_TYPE res = ReplaceFunction( dllName, funcName, newFunc, opcodes, origFunc );

if (res == FRR_OK || res == FRR_NODLL || (res == FRR_NOFUNC && on_error == FRR_IGNORE))
return;

fprintf(stderr, "Failed to %s function %s in module %s\n",
res==FRR_NOFUNC? "find" : "replace", funcName, dllName);
exit(1);
}

void doMallocReplacement()
{
#if __TBB_OVERLOAD_OLD_MSVCR
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr70)
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr71)
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr80)
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr90)
#endif
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr100)
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr110)
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr120)
__TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_RELEASE(ucrtbase)

for (size_t j = 0; j < arrayLength(modules_to_replace); j++) {
if (!modules_to_replace[j].doFuncReplacement)
continue;
for (size_t i = 0; i < arrayLength(c_routines_to_replace); i++)
{
ReplaceFunctionWithStore( modules_to_replace[j].name, c_routines_to_replace[i]._func, c_routines_to_replace[i]._fptr, NULL, NULL,  c_routines_to_replace[i]._on_error );
}
if ( strcmp(modules_to_replace[j].name, "ucrtbase.dll") == 0 ){
continue;
}

for (size_t i = 0; i < arrayLength(cxx_routines_to_replace); i++)
{
#if !_WIN64
if ( ((strcmp(modules_to_replace[j].name, "msvcr110.dll") == 0) || (strcmp(modules_to_replace[j].name, "msvcr120.dll") == 0)) && (strcmp(cxx_routines_to_replace[i]._func, "??3@YAXPAX@Z") == 0) ) continue;
if ( (strcmp(modules_to_replace[j].name, "msvcr120.dll") == 0) && (strcmp(cxx_routines_to_replace[i]._func, "??_V@YAXPAX@Z") == 0) ) continue;
#endif
ReplaceFunctionWithStore( modules_to_replace[j].name, cxx_routines_to_replace[i]._func, cxx_routines_to_replace[i]._fptr, NULL, NULL,  cxx_routines_to_replace[i]._on_error );
}
}
}

#endif 

extern "C" BOOL WINAPI DllMain( HINSTANCE hInst, DWORD callReason, LPVOID reserved )
{

if ( callReason==DLL_PROCESS_ATTACH && reserved && hInst ) {
#if !__TBB_WIN8UI_SUPPORT
#if TBBMALLOC_USE_TBB_FOR_ALLOCATOR_ENV_CONTROLLED
char pinEnvVariable[50];
if( GetEnvironmentVariable("TBBMALLOC_USE_TBB_FOR_ALLOCATOR", pinEnvVariable, 50))
{
doMallocReplacement();
}
#else
doMallocReplacement();
#endif
#endif 
}

return TRUE;
}

extern "C" __declspec(dllexport) void __TBB_malloc_proxy()
{

}

#endif 
