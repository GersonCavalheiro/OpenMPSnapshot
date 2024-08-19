
#ifndef _ITTNOTIFY_CONFIG_H_
#define _ITTNOTIFY_CONFIG_H_


#ifndef ITT_OS_WIN
#  define ITT_OS_WIN   1
#endif 

#ifndef ITT_OS_LINUX
#  define ITT_OS_LINUX 2
#endif 

#ifndef ITT_OS_MAC
#  define ITT_OS_MAC   3
#endif 

#ifndef ITT_OS_FREEBSD
#  define ITT_OS_FREEBSD   4
#endif 

#ifndef ITT_OS
#  if defined WIN32 || defined _WIN32
#    define ITT_OS ITT_OS_WIN
#  elif defined( __APPLE__ ) && defined( __MACH__ )
#    define ITT_OS ITT_OS_MAC
#  elif defined( __FreeBSD__ )
#    define ITT_OS ITT_OS_FREEBSD
#  else
#    define ITT_OS ITT_OS_LINUX
#  endif
#endif 

#ifndef ITT_PLATFORM_WIN
#  define ITT_PLATFORM_WIN 1
#endif 

#ifndef ITT_PLATFORM_POSIX
#  define ITT_PLATFORM_POSIX 2
#endif 

#ifndef ITT_PLATFORM_MAC
#  define ITT_PLATFORM_MAC 3
#endif 

#ifndef ITT_PLATFORM_FREEBSD
#  define ITT_PLATFORM_FREEBSD 4
#endif 

#ifndef ITT_PLATFORM
#  if ITT_OS==ITT_OS_WIN
#    define ITT_PLATFORM ITT_PLATFORM_WIN
#  elif ITT_OS==ITT_OS_MAC
#    define ITT_PLATFORM ITT_PLATFORM_MAC
#  elif ITT_OS==ITT_OS_FREEBSD
#    define ITT_PLATFORM ITT_PLATFORM_FREEBSD
#  else
#    define ITT_PLATFORM ITT_PLATFORM_POSIX
#  endif
#endif 

#if defined(_UNICODE) && !defined(UNICODE)
#define UNICODE
#endif

#include <stddef.h>
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#include <tchar.h>
#else  
#include <stdint.h>
#if defined(UNICODE) || defined(_UNICODE)
#include <wchar.h>
#endif 
#endif 

#ifndef ITTAPI_CDECL
#  if ITT_PLATFORM==ITT_PLATFORM_WIN
#    define ITTAPI_CDECL __cdecl
#  else 
#    if defined _M_IX86 || defined __i386__
#      define ITTAPI_CDECL __attribute__ ((cdecl))
#    else  
#      define ITTAPI_CDECL 
#    endif 
#  endif 
#endif 

#ifndef STDCALL
#  if ITT_PLATFORM==ITT_PLATFORM_WIN
#    define STDCALL __stdcall
#  else 
#    if defined _M_IX86 || defined __i386__
#      define STDCALL __attribute__ ((stdcall))
#    else  
#      define STDCALL 
#    endif 
#  endif 
#endif 

#define ITTAPI    ITTAPI_CDECL
#define LIBITTAPI ITTAPI_CDECL


#define ITTAPI_CALL    ITTAPI_CDECL
#define LIBITTAPI_CALL ITTAPI_CDECL

#if ITT_PLATFORM==ITT_PLATFORM_WIN

#if defined(__MINGW32__) && !defined(__cplusplus)
#define ITT_INLINE           static __inline__ __attribute__((__always_inline__,__gnu_inline__))
#else
#define ITT_INLINE           static __forceinline
#endif 

#define ITT_INLINE_ATTRIBUTE 
#else  

#ifdef __STRICT_ANSI__
#define ITT_INLINE           static
#define ITT_INLINE_ATTRIBUTE __attribute__((unused))
#else  
#define ITT_INLINE           static inline
#define ITT_INLINE_ATTRIBUTE __attribute__((always_inline, unused))
#endif 
#endif 


#ifndef ITT_ARCH_IA32
#  define ITT_ARCH_IA32  1
#endif 

#ifndef ITT_ARCH_IA32E
#  define ITT_ARCH_IA32E 2
#endif 

#ifndef ITT_ARCH_IA64
#  define ITT_ARCH_IA64 3
#endif 

#ifndef ITT_ARCH_ARM
#  define ITT_ARCH_ARM  4
#endif 

#ifndef ITT_ARCH_PPC64
#  define ITT_ARCH_PPC64  5
#endif 

#ifndef ITT_ARCH_ARM64
#  define ITT_ARCH_ARM64  6
#endif 

#ifndef ITT_ARCH
#  if defined _M_IX86 || defined __i386__
#    define ITT_ARCH ITT_ARCH_IA32
#  elif defined _M_X64 || defined _M_AMD64 || defined __x86_64__
#    define ITT_ARCH ITT_ARCH_IA32E
#  elif defined _M_IA64 || defined __ia64__
#    define ITT_ARCH ITT_ARCH_IA64
#  elif defined _M_ARM || defined __arm__
#    define ITT_ARCH ITT_ARCH_ARM
#  elif defined __aarch64__
#    define ITT_ARCH ITT_ARCH_ARM64
#  elif defined __powerpc64__
#    define ITT_ARCH ITT_ARCH_PPC64
#  endif
#endif

#ifdef __cplusplus
#  define ITT_EXTERN_C extern "C"
#  define ITT_EXTERN_C_BEGIN extern "C" {
#  define ITT_EXTERN_C_END }
#else
#  define ITT_EXTERN_C 
#  define ITT_EXTERN_C_BEGIN 
#  define ITT_EXTERN_C_END 
#endif 

#define ITT_TO_STR_AUX(x) #x
#define ITT_TO_STR(x)     ITT_TO_STR_AUX(x)

#define __ITT_BUILD_ASSERT(expr, suffix) do { \
static char __itt_build_check_##suffix[(expr) ? 1 : -1]; \
__itt_build_check_##suffix[0] = 0; \
} while(0)
#define _ITT_BUILD_ASSERT(expr, suffix)  __ITT_BUILD_ASSERT((expr), suffix)
#define ITT_BUILD_ASSERT(expr)           _ITT_BUILD_ASSERT((expr), __LINE__)

#define ITT_MAGIC { 0xED, 0xAB, 0xAB, 0xEC, 0x0D, 0xEE, 0xDA, 0x30 }


#define API_VERSION_BUILD    20180723

#ifndef API_VERSION_NUM
#define API_VERSION_NUM 3.23.0
#endif 

#define API_VERSION "ITT-API-Version " ITT_TO_STR(API_VERSION_NUM) \
" (" ITT_TO_STR(API_VERSION_BUILD) ")"


#if ITT_PLATFORM==ITT_PLATFORM_WIN
#include <windows.h>
typedef HMODULE           lib_t;
typedef DWORD             TIDT;
typedef CRITICAL_SECTION  mutex_t;
#ifdef __cplusplus
#define MUTEX_INITIALIZER {}
#else
#define MUTEX_INITIALIZER { 0 }
#endif
#define strong_alias(name, aliasname) 
#else  
#include <dlfcn.h>
#if defined(UNICODE) || defined(_UNICODE)
#include <wchar.h>
#endif 
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1 
#endif 
#ifndef __USE_UNIX98
#define __USE_UNIX98 1 
#endif 
#include <pthread.h>
typedef void*             lib_t;
typedef pthread_t         TIDT;
typedef pthread_mutex_t   mutex_t;
#define MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define _strong_alias(name, aliasname) \
extern __typeof (name) aliasname __attribute__ ((alias (#name)));
#define strong_alias(name, aliasname) _strong_alias(name, aliasname)
#endif 

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_get_proc(lib, name) GetProcAddress(lib, name)
#define __itt_mutex_init(mutex)   InitializeCriticalSection(mutex)
#define __itt_mutex_lock(mutex)   EnterCriticalSection(mutex)
#define __itt_mutex_unlock(mutex) LeaveCriticalSection(mutex)
#define __itt_mutex_destroy(mutex) DeleteCriticalSection(mutex)
#define __itt_load_lib(name)      LoadLibraryA(name)
#define __itt_unload_lib(handle)  FreeLibrary(handle)
#define __itt_system_error()      (int)GetLastError()
#define __itt_fstrcmp(s1, s2)     lstrcmpA(s1, s2)
#define __itt_fstrnlen(s, l)      strnlen_s(s, l)
#define __itt_fstrcpyn(s1, b, s2, l) strncpy_s(s1, b, s2, l)
#define __itt_thread_id()         GetCurrentThreadId()
#define __itt_thread_yield()      SwitchToThread()
#ifndef ITT_SIMPLE_INIT
ITT_INLINE long
__itt_interlocked_increment(volatile long* ptr) ITT_INLINE_ATTRIBUTE;
ITT_INLINE long __itt_interlocked_increment(volatile long* ptr)
{
return InterlockedIncrement(ptr);
}
ITT_INLINE long
__itt_interlocked_compare_exchange(volatile long* ptr, long exchange, long comperand) ITT_INLINE_ATTRIBUTE;
ITT_INLINE long
__itt_interlocked_compare_exchange(volatile long* ptr, long exchange, long comperand)
{
return InterlockedCompareExchange(ptr, exchange, comperand);
}
#endif 

#define DL_SYMBOLS (1)
#define PTHREAD_SYMBOLS (1)

#else 
#define __itt_get_proc(lib, name) dlsym(lib, name)
#define __itt_mutex_init(mutex)   {\
pthread_mutexattr_t mutex_attr;                                         \
int error_code = pthread_mutexattr_init(&mutex_attr);                   \
if (error_code)                                                         \
__itt_report_error(__itt_error_system, "pthread_mutexattr_init",    \
error_code);                                     \
error_code = pthread_mutexattr_settype(&mutex_attr,                     \
PTHREAD_MUTEX_RECURSIVE);        \
if (error_code)                                                         \
__itt_report_error(__itt_error_system, "pthread_mutexattr_settype", \
error_code);                                     \
error_code = pthread_mutex_init(mutex, &mutex_attr);                    \
if (error_code)                                                         \
__itt_report_error(__itt_error_system, "pthread_mutex_init",        \
error_code);                                     \
error_code = pthread_mutexattr_destroy(&mutex_attr);                    \
if (error_code)                                                         \
__itt_report_error(__itt_error_system, "pthread_mutexattr_destroy", \
error_code);                                     \
}
#define __itt_mutex_lock(mutex)   pthread_mutex_lock(mutex)
#define __itt_mutex_unlock(mutex) pthread_mutex_unlock(mutex)
#define __itt_mutex_destroy(mutex) pthread_mutex_destroy(mutex)
#define __itt_load_lib(name)      dlopen(name, RTLD_LAZY)
#define __itt_unload_lib(handle)  dlclose(handle)
#define __itt_system_error()      errno
#define __itt_fstrcmp(s1, s2)     strcmp(s1, s2)


#ifdef SDL_STRNLEN_S
#define __itt_fstrnlen(s, l)      SDL_STRNLEN_S(s, l)
#else
#define __itt_fstrnlen(s, l)      strlen(s)
#endif 
#ifdef SDL_STRNCPY_S
#define __itt_fstrcpyn(s1, b, s2, l) SDL_STRNCPY_S(s1, b, s2, l)
#else
#define __itt_fstrcpyn(s1, b, s2, l) {                                      \
if (b > 0) {                                                            \
\
\
volatile size_t num_to_copy = (size_t)(b - 1) < (size_t)(l) ?       \
(size_t)(b - 1) : (size_t)(l);                              \
strncpy(s1, s2, num_to_copy);                                       \
s1[num_to_copy] = 0;                                                \
}                                                                       \
}
#endif 

#define __itt_thread_id()         pthread_self()
#define __itt_thread_yield()      sched_yield()
#if ITT_ARCH==ITT_ARCH_IA64
#ifdef __INTEL_COMPILER
#define __TBB_machine_fetchadd4(addr, val) __fetchadd4_acq((void *)addr, val)
#else  

#endif 
#elif ITT_ARCH==ITT_ARCH_IA32 || ITT_ARCH==ITT_ARCH_IA32E 
ITT_INLINE long
__TBB_machine_fetchadd4(volatile void* ptr, long addend) ITT_INLINE_ATTRIBUTE;
ITT_INLINE long __TBB_machine_fetchadd4(volatile void* ptr, long addend)
{
long result;
__asm__ __volatile__("lock\nxadd %0,%1"
: "=r"(result),"=m"(*(volatile int*)ptr)
: "0"(addend), "m"(*(volatile int*)ptr)
: "memory");
return result;
}
#else
#define __TBB_machine_fetchadd4(addr, val) __sync_fetch_and_add(addr, val)
#endif 
#ifndef ITT_SIMPLE_INIT
ITT_INLINE long
__itt_interlocked_increment(volatile long* ptr) ITT_INLINE_ATTRIBUTE;
ITT_INLINE long __itt_interlocked_increment(volatile long* ptr)
{
return __TBB_machine_fetchadd4(ptr, 1) + 1L;
}
ITT_INLINE long
__itt_interlocked_compare_exchange(volatile long* ptr, long exchange, long comperand) ITT_INLINE_ATTRIBUTE;
ITT_INLINE long
__itt_interlocked_compare_exchange(volatile long* ptr, long exchange, long comperand)
{
return __sync_val_compare_and_swap(ptr, exchange, comperand);
}
#endif 

void* dlopen(const char*, int) __attribute__((weak));
void* dlsym(void*, const char*) __attribute__((weak));
int dlclose(void*) __attribute__((weak));
#define DL_SYMBOLS (dlopen && dlsym && dlclose)

int pthread_mutex_init(pthread_mutex_t*, const pthread_mutexattr_t*) __attribute__((weak));
int pthread_mutex_lock(pthread_mutex_t*) __attribute__((weak));
int pthread_mutex_unlock(pthread_mutex_t*) __attribute__((weak));
int pthread_mutex_destroy(pthread_mutex_t*) __attribute__((weak));
int pthread_mutexattr_init(pthread_mutexattr_t*) __attribute__((weak));
int pthread_mutexattr_settype(pthread_mutexattr_t*, int) __attribute__((weak));
int pthread_mutexattr_destroy(pthread_mutexattr_t*) __attribute__((weak));
pthread_t pthread_self(void) __attribute__((weak));
#define PTHREAD_SYMBOLS (pthread_mutex_init && pthread_mutex_lock && pthread_mutex_unlock && pthread_mutex_destroy && pthread_mutexattr_init && pthread_mutexattr_settype && pthread_mutexattr_destroy && pthread_self)

#endif 


#define ITT_STRDUP_MAX_STRING_SIZE 4096
#define __itt_fstrdup(s, new_s) do {                                        \
if (s != NULL) {                                                        \
size_t s_len = __itt_fstrnlen(s, ITT_STRDUP_MAX_STRING_SIZE);       \
new_s = (char *)malloc(s_len + 1);                                  \
if (new_s != NULL) {                                                \
__itt_fstrcpyn(new_s, s_len + 1, s, s_len);                     \
}                                                                   \
}                                                                       \
} while(0)

typedef enum {
__itt_thread_normal  = 0,
__itt_thread_ignored = 1
} __itt_thread_state;

#pragma pack(push, 8)

typedef struct ___itt_thread_info
{
const char* nameA; 
#if defined(UNICODE) || defined(_UNICODE)
const wchar_t* nameW; 
#else  
void* nameW;
#endif 
TIDT               tid;
__itt_thread_state state;   
int                extra1;  
void*              extra2;  
struct ___itt_thread_info* next;
} __itt_thread_info;

#include "ittnotify_types.h" 

typedef struct ___itt_api_info_20101001
{
const char*    name;
void**         func_ptr;
void*          init_func;
__itt_group_id group;
}  __itt_api_info_20101001;

typedef struct ___itt_api_info
{
const char*    name;
void**         func_ptr;
void*          init_func;
void*          null_func;
__itt_group_id group;
}  __itt_api_info;

typedef struct __itt_counter_info
{
const char* nameA;  
#if defined(UNICODE) || defined(_UNICODE)
const wchar_t* nameW; 
#else  
void* nameW;
#endif 
const char* domainA;  
#if defined(UNICODE) || defined(_UNICODE)
const wchar_t* domainW; 
#else  
void* domainW;
#endif 
int type;
long index;
int   extra1; 
void* extra2; 
struct __itt_counter_info* next;
}  __itt_counter_info_t;

struct ___itt_domain;
struct ___itt_string_handle;
struct ___itt_histogram;

#include "ittnotify.h"

typedef struct ___itt_global
{
unsigned char          magic[8];
unsigned long          version_major;
unsigned long          version_minor;
unsigned long          version_build;
volatile long          api_initialized;
volatile long          mutex_initialized;
volatile long          atomic_counter;
mutex_t                mutex;
lib_t                  lib;
void*                  error_handler;
const char**           dll_path_ptr;
__itt_api_info*        api_list_ptr;
struct ___itt_global*  next;

__itt_thread_info*     thread_list;
struct ___itt_domain*  domain_list;
struct ___itt_string_handle* string_list;
__itt_collection_state state;
__itt_counter_info_t*  counter_list;
unsigned int           ipt_collect_events;
struct ___itt_histogram* histogram_list;
} __itt_global;

#pragma pack(pop)

#define NEW_THREAD_INFO_W(gptr,h,h_tail,t,s,n) { \
h = (__itt_thread_info*)malloc(sizeof(__itt_thread_info)); \
if (h != NULL) { \
h->tid    = t; \
h->nameA  = NULL; \
h->nameW  = n ? _wcsdup(n) : NULL; \
h->state  = s; \
h->extra1 = 0;     \
h->extra2 = NULL;  \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->thread_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_THREAD_INFO_A(gptr,h,h_tail,t,s,n) { \
h = (__itt_thread_info*)malloc(sizeof(__itt_thread_info)); \
if (h != NULL) { \
h->tid    = t; \
char *n_copy = NULL; \
__itt_fstrdup(n, n_copy); \
h->nameA  = n_copy; \
h->nameW  = NULL; \
h->state  = s; \
h->extra1 = 0;     \
h->extra2 = NULL;  \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->thread_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_DOMAIN_W(gptr,h,h_tail,name) { \
h = (__itt_domain*)malloc(sizeof(__itt_domain)); \
if (h != NULL) { \
h->flags  = 1;     \
h->nameA  = NULL; \
h->nameW  = name ? _wcsdup(name) : NULL; \
h->extra1 = 0;     \
h->extra2 = NULL;  \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->domain_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_DOMAIN_A(gptr,h,h_tail,name) { \
h = (__itt_domain*)malloc(sizeof(__itt_domain)); \
if (h != NULL) { \
h->flags  = 1;     \
char *name_copy = NULL; \
__itt_fstrdup(name, name_copy); \
h->nameA  = name_copy; \
h->nameW  = NULL; \
h->extra1 = 0;     \
h->extra2 = NULL;  \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->domain_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_STRING_HANDLE_W(gptr,h,h_tail,name) { \
h = (__itt_string_handle*)malloc(sizeof(__itt_string_handle)); \
if (h != NULL) { \
h->strA   = NULL; \
h->strW   = name ? _wcsdup(name) : NULL; \
h->extra1 = 0;     \
h->extra2 = NULL;  \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->string_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_STRING_HANDLE_A(gptr,h,h_tail,name) { \
h = (__itt_string_handle*)malloc(sizeof(__itt_string_handle)); \
if (h != NULL) { \
char *name_copy = NULL; \
__itt_fstrdup(name, name_copy); \
h->strA  = name_copy; \
h->strW   = NULL; \
h->extra1 = 0;     \
h->extra2 = NULL;  \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->string_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_COUNTER_W(gptr,h,h_tail,name,domain,type) { \
h = (__itt_counter_info_t*)malloc(sizeof(__itt_counter_info_t)); \
if (h != NULL) { \
h->nameA   = NULL; \
h->nameW   = name ? _wcsdup(name) : NULL; \
h->domainA   = NULL; \
h->domainW   = name ? _wcsdup(domain) : NULL; \
h->type = type; \
h->index = 0; \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->counter_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_COUNTER_A(gptr,h,h_tail,name,domain,type) { \
h = (__itt_counter_info_t*)malloc(sizeof(__itt_counter_info_t)); \
if (h != NULL) { \
char *name_copy = NULL; \
__itt_fstrdup(name, name_copy); \
h->nameA  = name_copy; \
h->nameW   = NULL; \
char *domain_copy = NULL; \
__itt_fstrdup(domain, domain_copy); \
h->domainA  = domain_copy; \
h->domainW   = NULL; \
h->type = type; \
h->index = 0; \
h->next   = NULL; \
if (h_tail == NULL) \
(gptr)->counter_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_HISTOGRAM_W(gptr,h,h_tail,domain,name,x_type,y_type) { \
h = (__itt_histogram*)malloc(sizeof(__itt_histogram)); \
if (h != NULL) { \
h->domain = domain; \
h->nameA  = NULL; \
h->nameW  = name ? _wcsdup(name) : NULL; \
h->x_type = x_type; \
h->y_type = y_type; \
h->extra1 = 0; \
h->extra2 = NULL; \
if (h_tail == NULL) \
(gptr)->histogram_list = h; \
else \
h_tail->next = h; \
} \
}

#define NEW_HISTOGRAM_A(gptr,h,h_tail,domain,name,x_type,y_type) { \
h = (__itt_histogram*)malloc(sizeof(__itt_histogram)); \
if (h != NULL) { \
h->domain = domain; \
char *name_copy = NULL; \
__itt_fstrdup(name, name_copy); \
h->nameA  = name_copy; \
h->nameW  = NULL; \
h->x_type = x_type; \
h->y_type = y_type; \
h->extra1 = 0; \
h->extra2 = NULL; \
if (h_tail == NULL) \
(gptr)->histogram_list = h; \
else \
h_tail->next = h; \
} \
}

#endif 
