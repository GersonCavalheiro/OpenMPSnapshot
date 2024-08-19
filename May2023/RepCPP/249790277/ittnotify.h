

#ifndef _ITTNOTIFY_H_
#define _ITTNOTIFY_H_




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

#ifndef CDECL
#  if ITT_PLATFORM==ITT_PLATFORM_WIN
#    define CDECL __cdecl
#  else 
#    if defined _M_IX86 || defined __i386__
#      define CDECL __attribute__ ((cdecl))
#    else  
#      define CDECL 
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

#define ITTAPI    CDECL
#define LIBITTAPI CDECL


#define ITTAPI_CALL    CDECL
#define LIBITTAPI_CALL CDECL

#if ITT_PLATFORM==ITT_PLATFORM_WIN

#define ITT_INLINE           __forceinline
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


#ifdef INTEL_ITTNOTIFY_ENABLE_LEGACY
#  if ITT_PLATFORM==ITT_PLATFORM_WIN
#    pragma message("WARNING!!! Deprecated API is used. Please undefine INTEL_ITTNOTIFY_ENABLE_LEGACY macro")
#  else 
#  endif 
#  include "legacy/ittnotify.h"
#endif 



#define ITT_JOIN_AUX(p,n) p##n
#define ITT_JOIN(p,n)     ITT_JOIN_AUX(p,n)

#ifdef ITT_MAJOR
#undef ITT_MAJOR
#endif
#ifdef ITT_MINOR
#undef ITT_MINOR
#endif
#define ITT_MAJOR     3
#define ITT_MINOR     0


#define ITT_VERSIONIZE(x)    \
ITT_JOIN(x,              \
ITT_JOIN(_,              \
ITT_JOIN(ITT_MAJOR,      \
ITT_JOIN(_, ITT_MINOR))))

#ifndef INTEL_ITTNOTIFY_PREFIX
#  define INTEL_ITTNOTIFY_PREFIX __itt_
#endif 
#ifndef INTEL_ITTNOTIFY_POSTFIX
#  define INTEL_ITTNOTIFY_POSTFIX _ptr_
#endif 

#define ITTNOTIFY_NAME_AUX(n) ITT_JOIN(INTEL_ITTNOTIFY_PREFIX,n)
#define ITTNOTIFY_NAME(n)     ITT_VERSIONIZE(ITTNOTIFY_NAME_AUX(ITT_JOIN(n,INTEL_ITTNOTIFY_POSTFIX)))

#define ITTNOTIFY_VOID(n) (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)
#define ITTNOTIFY_DATA(n) (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)

#define ITTNOTIFY_VOID_D0(n,d)       (!(d)->flags) ? (void)0 : (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)(d)
#define ITTNOTIFY_VOID_D1(n,d,x)     (!(d)->flags) ? (void)0 : (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)(d,x)
#define ITTNOTIFY_VOID_D2(n,d,x,y)   (!(d)->flags) ? (void)0 : (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)(d,x,y)
#define ITTNOTIFY_VOID_D3(n,d,x,y,z) (!(d)->flags) ? (void)0 : (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)(d,x,y,z)
#define ITTNOTIFY_VOID_D4(n,d,x,y,z,a)     (!(d)->flags) ? (void)0 : (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)(d,x,y,z,a)
#define ITTNOTIFY_VOID_D5(n,d,x,y,z,a,b)   (!(d)->flags) ? (void)0 : (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)(d,x,y,z,a,b)
#define ITTNOTIFY_VOID_D6(n,d,x,y,z,a,b,c) (!(d)->flags) ? (void)0 : (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)(d,x,y,z,a,b,c)
#define ITTNOTIFY_DATA_D0(n,d)       (!(d)->flags) ?       0 : (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)(d)
#define ITTNOTIFY_DATA_D1(n,d,x)     (!(d)->flags) ?       0 : (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)(d,x)
#define ITTNOTIFY_DATA_D2(n,d,x,y)   (!(d)->flags) ?       0 : (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)(d,x,y)
#define ITTNOTIFY_DATA_D3(n,d,x,y,z) (!(d)->flags) ?       0 : (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)(d,x,y,z)
#define ITTNOTIFY_DATA_D4(n,d,x,y,z,a)     (!(d)->flags) ? 0 : (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)(d,x,y,z,a)
#define ITTNOTIFY_DATA_D5(n,d,x,y,z,a,b)   (!(d)->flags) ? 0 : (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)(d,x,y,z,a,b)
#define ITTNOTIFY_DATA_D6(n,d,x,y,z,a,b,c) (!(d)->flags) ? 0 : (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)(d,x,y,z,a,b,c)

#ifdef ITT_STUB
#undef ITT_STUB
#endif
#ifdef ITT_STUBV
#undef ITT_STUBV
#endif
#define ITT_STUBV(api,type,name,args)                             \
typedef type (api* ITT_JOIN(ITTNOTIFY_NAME(name),_t)) args;   \
extern ITT_JOIN(ITTNOTIFY_NAME(name),_t) ITTNOTIFY_NAME(name);
#define ITT_STUB ITT_STUBV


#ifdef __cplusplus
extern "C" {
#endif 






void ITTAPI __itt_pause(void);

void ITTAPI __itt_resume(void);

void ITTAPI __itt_detach(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, pause,  (void))
ITT_STUBV(ITTAPI, void, resume, (void))
ITT_STUBV(ITTAPI, void, detach, (void))
#define __itt_pause      ITTNOTIFY_VOID(pause)
#define __itt_pause_ptr  ITTNOTIFY_NAME(pause)
#define __itt_resume     ITTNOTIFY_VOID(resume)
#define __itt_resume_ptr ITTNOTIFY_NAME(resume)
#define __itt_detach     ITTNOTIFY_VOID(detach)
#define __itt_detach_ptr ITTNOTIFY_NAME(detach)
#else  
#define __itt_pause()
#define __itt_pause_ptr  0
#define __itt_resume()
#define __itt_resume_ptr 0
#define __itt_detach()
#define __itt_detach_ptr 0
#endif 
#else  
#define __itt_pause_ptr  0
#define __itt_resume_ptr 0
#define __itt_detach_ptr 0
#endif 






#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_thread_set_nameA(const char    *name);
void ITTAPI __itt_thread_set_nameW(const wchar_t *name);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_thread_set_name     __itt_thread_set_nameW
#  define __itt_thread_set_name_ptr __itt_thread_set_nameW_ptr
#else 
#  define __itt_thread_set_name     __itt_thread_set_nameA
#  define __itt_thread_set_name_ptr __itt_thread_set_nameA_ptr
#endif 
#else  
void ITTAPI __itt_thread_set_name(const char *name);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, thread_set_nameA, (const char    *name))
ITT_STUBV(ITTAPI, void, thread_set_nameW, (const wchar_t *name))
#else  
ITT_STUBV(ITTAPI, void, thread_set_name,  (const char    *name))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_thread_set_nameA     ITTNOTIFY_VOID(thread_set_nameA)
#define __itt_thread_set_nameA_ptr ITTNOTIFY_NAME(thread_set_nameA)
#define __itt_thread_set_nameW     ITTNOTIFY_VOID(thread_set_nameW)
#define __itt_thread_set_nameW_ptr ITTNOTIFY_NAME(thread_set_nameW)
#else 
#define __itt_thread_set_name     ITTNOTIFY_VOID(thread_set_name)
#define __itt_thread_set_name_ptr ITTNOTIFY_NAME(thread_set_name)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_thread_set_nameA(name)
#define __itt_thread_set_nameA_ptr 0
#define __itt_thread_set_nameW(name)
#define __itt_thread_set_nameW_ptr 0
#else 
#define __itt_thread_set_name(name)
#define __itt_thread_set_name_ptr 0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_thread_set_nameA_ptr 0
#define __itt_thread_set_nameW_ptr 0
#else 
#define __itt_thread_set_name_ptr 0
#endif 
#endif 





void ITTAPI __itt_thread_ignore(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, thread_ignore, (void))
#define __itt_thread_ignore     ITTNOTIFY_VOID(thread_ignore)
#define __itt_thread_ignore_ptr ITTNOTIFY_NAME(thread_ignore)
#else  
#define __itt_thread_ignore()
#define __itt_thread_ignore_ptr 0
#endif 
#else  
#define __itt_thread_ignore_ptr 0
#endif 








#define __itt_suppress_all_errors 0x7fffffff


#define __itt_suppress_threading_errors 0x000000ff


#define __itt_suppress_memory_errors 0x0000ff00


void ITTAPI __itt_suppress_push(unsigned int mask);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, suppress_push, (unsigned int mask))
#define __itt_suppress_push     ITTNOTIFY_VOID(suppress_push)
#define __itt_suppress_push_ptr ITTNOTIFY_NAME(suppress_push)
#else  
#define __itt_suppress_push(mask)
#define __itt_suppress_push_ptr 0
#endif 
#else  
#define __itt_suppress_push_ptr 0
#endif 



void ITTAPI __itt_suppress_pop(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, suppress_pop, (void))
#define __itt_suppress_pop     ITTNOTIFY_VOID(suppress_pop)
#define __itt_suppress_pop_ptr ITTNOTIFY_NAME(suppress_pop)
#else  
#define __itt_suppress_pop()
#define __itt_suppress_pop_ptr 0
#endif 
#else  
#define __itt_suppress_pop_ptr 0
#endif 



typedef enum __itt_suppress_mode {
__itt_unsuppress_range,
__itt_suppress_range
} __itt_suppress_mode_t;


void ITTAPI __itt_suppress_mark_range(__itt_suppress_mode_t mode, unsigned int mask, void * address, size_t size);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, suppress_mark_range, (__itt_suppress_mode_t mode, unsigned int mask, void * address, size_t size))
#define __itt_suppress_mark_range     ITTNOTIFY_VOID(suppress_mark_range)
#define __itt_suppress_mark_range_ptr ITTNOTIFY_NAME(suppress_mark_range)
#else  
#define __itt_suppress_mark_range(mask)
#define __itt_suppress_mark_range_ptr 0
#endif 
#else  
#define __itt_suppress_mark_range_ptr 0
#endif 



void ITTAPI __itt_suppress_clear_range(__itt_suppress_mode_t mode, unsigned int mask, void * address, size_t size);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, suppress_clear_range, (__itt_suppress_mode_t mode, unsigned int mask, void * address, size_t size))
#define __itt_suppress_clear_range     ITTNOTIFY_VOID(suppress_clear_range)
#define __itt_suppress_clear_range_ptr ITTNOTIFY_NAME(suppress_clear_range)
#else  
#define __itt_suppress_clear_range(mask)
#define __itt_suppress_clear_range_ptr 0
#endif 
#else  
#define __itt_suppress_clear_range_ptr 0
#endif 






#define __itt_attr_barrier 1


#define __itt_attr_mutex   2



#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_sync_createA(void *addr, const char    *objtype, const char    *objname, int attribute);
void ITTAPI __itt_sync_createW(void *addr, const wchar_t *objtype, const wchar_t *objname, int attribute);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_sync_create     __itt_sync_createW
#  define __itt_sync_create_ptr __itt_sync_createW_ptr
#else 
#  define __itt_sync_create     __itt_sync_createA
#  define __itt_sync_create_ptr __itt_sync_createA_ptr
#endif 
#else 
void ITTAPI __itt_sync_create (void *addr, const char *objtype, const char *objname, int attribute);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, sync_createA, (void *addr, const char    *objtype, const char    *objname, int attribute))
ITT_STUBV(ITTAPI, void, sync_createW, (void *addr, const wchar_t *objtype, const wchar_t *objname, int attribute))
#else  
ITT_STUBV(ITTAPI, void, sync_create,  (void *addr, const char*    objtype, const char*    objname, int attribute))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_createA     ITTNOTIFY_VOID(sync_createA)
#define __itt_sync_createA_ptr ITTNOTIFY_NAME(sync_createA)
#define __itt_sync_createW     ITTNOTIFY_VOID(sync_createW)
#define __itt_sync_createW_ptr ITTNOTIFY_NAME(sync_createW)
#else 
#define __itt_sync_create     ITTNOTIFY_VOID(sync_create)
#define __itt_sync_create_ptr ITTNOTIFY_NAME(sync_create)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_createA(addr, objtype, objname, attribute)
#define __itt_sync_createA_ptr 0
#define __itt_sync_createW(addr, objtype, objname, attribute)
#define __itt_sync_createW_ptr 0
#else 
#define __itt_sync_create(addr, objtype, objname, attribute)
#define __itt_sync_create_ptr 0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_createA_ptr 0
#define __itt_sync_createW_ptr 0
#else 
#define __itt_sync_create_ptr 0
#endif 
#endif 



#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_sync_renameA(void *addr, const char    *name);
void ITTAPI __itt_sync_renameW(void *addr, const wchar_t *name);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_sync_rename     __itt_sync_renameW
#  define __itt_sync_rename_ptr __itt_sync_renameW_ptr
#else 
#  define __itt_sync_rename     __itt_sync_renameA
#  define __itt_sync_rename_ptr __itt_sync_renameA_ptr
#endif 
#else 
void ITTAPI __itt_sync_rename(void *addr, const char *name);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, sync_renameA, (void *addr, const char    *name))
ITT_STUBV(ITTAPI, void, sync_renameW, (void *addr, const wchar_t *name))
#else  
ITT_STUBV(ITTAPI, void, sync_rename,  (void *addr, const char    *name))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_renameA     ITTNOTIFY_VOID(sync_renameA)
#define __itt_sync_renameA_ptr ITTNOTIFY_NAME(sync_renameA)
#define __itt_sync_renameW     ITTNOTIFY_VOID(sync_renameW)
#define __itt_sync_renameW_ptr ITTNOTIFY_NAME(sync_renameW)
#else 
#define __itt_sync_rename     ITTNOTIFY_VOID(sync_rename)
#define __itt_sync_rename_ptr ITTNOTIFY_NAME(sync_rename)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_renameA(addr, name)
#define __itt_sync_renameA_ptr 0
#define __itt_sync_renameW(addr, name)
#define __itt_sync_renameW_ptr 0
#else 
#define __itt_sync_rename(addr, name)
#define __itt_sync_rename_ptr 0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_renameA_ptr 0
#define __itt_sync_renameW_ptr 0
#else 
#define __itt_sync_rename_ptr 0
#endif 
#endif 



void ITTAPI __itt_sync_destroy(void *addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_destroy, (void *addr))
#define __itt_sync_destroy     ITTNOTIFY_VOID(sync_destroy)
#define __itt_sync_destroy_ptr ITTNOTIFY_NAME(sync_destroy)
#else  
#define __itt_sync_destroy(addr)
#define __itt_sync_destroy_ptr 0
#endif 
#else  
#define __itt_sync_destroy_ptr 0
#endif 





void ITTAPI __itt_sync_prepare(void* addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_prepare, (void *addr))
#define __itt_sync_prepare     ITTNOTIFY_VOID(sync_prepare)
#define __itt_sync_prepare_ptr ITTNOTIFY_NAME(sync_prepare)
#else  
#define __itt_sync_prepare(addr)
#define __itt_sync_prepare_ptr 0
#endif 
#else  
#define __itt_sync_prepare_ptr 0
#endif 



void ITTAPI __itt_sync_cancel(void *addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_cancel, (void *addr))
#define __itt_sync_cancel     ITTNOTIFY_VOID(sync_cancel)
#define __itt_sync_cancel_ptr ITTNOTIFY_NAME(sync_cancel)
#else  
#define __itt_sync_cancel(addr)
#define __itt_sync_cancel_ptr 0
#endif 
#else  
#define __itt_sync_cancel_ptr 0
#endif 



void ITTAPI __itt_sync_acquired(void *addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_acquired, (void *addr))
#define __itt_sync_acquired     ITTNOTIFY_VOID(sync_acquired)
#define __itt_sync_acquired_ptr ITTNOTIFY_NAME(sync_acquired)
#else  
#define __itt_sync_acquired(addr)
#define __itt_sync_acquired_ptr 0
#endif 
#else  
#define __itt_sync_acquired_ptr 0
#endif 



void ITTAPI __itt_sync_releasing(void* addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_releasing, (void *addr))
#define __itt_sync_releasing     ITTNOTIFY_VOID(sync_releasing)
#define __itt_sync_releasing_ptr ITTNOTIFY_NAME(sync_releasing)
#else  
#define __itt_sync_releasing(addr)
#define __itt_sync_releasing_ptr 0
#endif 
#else  
#define __itt_sync_releasing_ptr 0
#endif 








void ITTAPI __itt_fsync_prepare(void* addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_prepare, (void *addr))
#define __itt_fsync_prepare     ITTNOTIFY_VOID(fsync_prepare)
#define __itt_fsync_prepare_ptr ITTNOTIFY_NAME(fsync_prepare)
#else  
#define __itt_fsync_prepare(addr)
#define __itt_fsync_prepare_ptr 0
#endif 
#else  
#define __itt_fsync_prepare_ptr 0
#endif 



void ITTAPI __itt_fsync_cancel(void *addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_cancel, (void *addr))
#define __itt_fsync_cancel     ITTNOTIFY_VOID(fsync_cancel)
#define __itt_fsync_cancel_ptr ITTNOTIFY_NAME(fsync_cancel)
#else  
#define __itt_fsync_cancel(addr)
#define __itt_fsync_cancel_ptr 0
#endif 
#else  
#define __itt_fsync_cancel_ptr 0
#endif 



void ITTAPI __itt_fsync_acquired(void *addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_acquired, (void *addr))
#define __itt_fsync_acquired     ITTNOTIFY_VOID(fsync_acquired)
#define __itt_fsync_acquired_ptr ITTNOTIFY_NAME(fsync_acquired)
#else  
#define __itt_fsync_acquired(addr)
#define __itt_fsync_acquired_ptr 0
#endif 
#else  
#define __itt_fsync_acquired_ptr 0
#endif 



void ITTAPI __itt_fsync_releasing(void* addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_releasing, (void *addr))
#define __itt_fsync_releasing     ITTNOTIFY_VOID(fsync_releasing)
#define __itt_fsync_releasing_ptr ITTNOTIFY_NAME(fsync_releasing)
#else  
#define __itt_fsync_releasing(addr)
#define __itt_fsync_releasing_ptr 0
#endif 
#else  
#define __itt_fsync_releasing_ptr 0
#endif 




#if !defined(_ADVISOR_ANNOTATE_H_) || defined(ANNOTATE_EXPAND_NULL)

typedef void* __itt_model_site;             
typedef void* __itt_model_site_instance;    
typedef void* __itt_model_task;             
typedef void* __itt_model_task_instance;    


typedef enum {
__itt_model_disable_observation,
__itt_model_disable_collection
} __itt_model_disable;

#endif 


void ITTAPI __itt_model_site_begin(__itt_model_site *site, __itt_model_site_instance *instance, const char *name);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_model_site_beginW(const wchar_t *name);
#endif
void ITTAPI __itt_model_site_beginA(const char *name);
void ITTAPI __itt_model_site_beginAL(const char *name, size_t siteNameLen);
void ITTAPI __itt_model_site_end  (__itt_model_site *site, __itt_model_site_instance *instance);
void ITTAPI __itt_model_site_end_2(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_site_begin,  (__itt_model_site *site, __itt_model_site_instance *instance, const char *name))
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, model_site_beginW,  (const wchar_t *name))
#endif
ITT_STUBV(ITTAPI, void, model_site_beginA,  (const char *name))
ITT_STUBV(ITTAPI, void, model_site_beginAL,  (const char *name, size_t siteNameLen))
ITT_STUBV(ITTAPI, void, model_site_end,    (__itt_model_site *site, __itt_model_site_instance *instance))
ITT_STUBV(ITTAPI, void, model_site_end_2,  (void))
#define __itt_model_site_begin      ITTNOTIFY_VOID(model_site_begin)
#define __itt_model_site_begin_ptr  ITTNOTIFY_NAME(model_site_begin)
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_model_site_beginW      ITTNOTIFY_VOID(model_site_beginW)
#define __itt_model_site_beginW_ptr  ITTNOTIFY_NAME(model_site_beginW)
#endif
#define __itt_model_site_beginA      ITTNOTIFY_VOID(model_site_beginA)
#define __itt_model_site_beginA_ptr  ITTNOTIFY_NAME(model_site_beginA)
#define __itt_model_site_beginAL      ITTNOTIFY_VOID(model_site_beginAL)
#define __itt_model_site_beginAL_ptr  ITTNOTIFY_NAME(model_site_beginAL)
#define __itt_model_site_end        ITTNOTIFY_VOID(model_site_end)
#define __itt_model_site_end_ptr    ITTNOTIFY_NAME(model_site_end)
#define __itt_model_site_end_2        ITTNOTIFY_VOID(model_site_end_2)
#define __itt_model_site_end_2_ptr    ITTNOTIFY_NAME(model_site_end_2)
#else  
#define __itt_model_site_begin(site, instance, name)
#define __itt_model_site_begin_ptr  0
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_model_site_beginW(name)
#define __itt_model_site_beginW_ptr  0
#endif
#define __itt_model_site_beginA(name)
#define __itt_model_site_beginA_ptr  0
#define __itt_model_site_beginAL(name, siteNameLen)
#define __itt_model_site_beginAL_ptr  0
#define __itt_model_site_end(site, instance)
#define __itt_model_site_end_ptr    0
#define __itt_model_site_end_2()
#define __itt_model_site_end_2_ptr    0
#endif 
#else  
#define __itt_model_site_begin_ptr  0
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_model_site_beginW_ptr  0
#endif
#define __itt_model_site_beginA_ptr  0
#define __itt_model_site_beginAL_ptr  0
#define __itt_model_site_end_ptr    0
#define __itt_model_site_end_2_ptr    0
#endif 



void ITTAPI __itt_model_task_begin(__itt_model_task *task, __itt_model_task_instance *instance, const char *name);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_model_task_beginW(const wchar_t *name);
void ITTAPI __itt_model_iteration_taskW(const wchar_t *name);
#endif
void ITTAPI __itt_model_task_beginA(const char *name);
void ITTAPI __itt_model_task_beginAL(const char *name, size_t taskNameLen);
void ITTAPI __itt_model_iteration_taskA(const char *name);
void ITTAPI __itt_model_iteration_taskAL(const char *name, size_t taskNameLen);
void ITTAPI __itt_model_task_end  (__itt_model_task *task, __itt_model_task_instance *instance);
void ITTAPI __itt_model_task_end_2(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_task_begin,  (__itt_model_task *task, __itt_model_task_instance *instance, const char *name))
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, model_task_beginW,  (const wchar_t *name))
ITT_STUBV(ITTAPI, void, model_iteration_taskW, (const wchar_t *name))
#endif
ITT_STUBV(ITTAPI, void, model_task_beginA,  (const char *name))
ITT_STUBV(ITTAPI, void, model_task_beginAL,  (const char *name, size_t taskNameLen))
ITT_STUBV(ITTAPI, void, model_iteration_taskA,  (const char *name))
ITT_STUBV(ITTAPI, void, model_iteration_taskAL,  (const char *name, size_t taskNameLen))
ITT_STUBV(ITTAPI, void, model_task_end,    (__itt_model_task *task, __itt_model_task_instance *instance))
ITT_STUBV(ITTAPI, void, model_task_end_2,  (void))
#define __itt_model_task_begin      ITTNOTIFY_VOID(model_task_begin)
#define __itt_model_task_begin_ptr  ITTNOTIFY_NAME(model_task_begin)
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_model_task_beginW     ITTNOTIFY_VOID(model_task_beginW)
#define __itt_model_task_beginW_ptr ITTNOTIFY_NAME(model_task_beginW)
#define __itt_model_iteration_taskW     ITTNOTIFY_VOID(model_iteration_taskW)
#define __itt_model_iteration_taskW_ptr ITTNOTIFY_NAME(model_iteration_taskW)
#endif
#define __itt_model_task_beginA    ITTNOTIFY_VOID(model_task_beginA)
#define __itt_model_task_beginA_ptr ITTNOTIFY_NAME(model_task_beginA)
#define __itt_model_task_beginAL    ITTNOTIFY_VOID(model_task_beginAL)
#define __itt_model_task_beginAL_ptr ITTNOTIFY_NAME(model_task_beginAL)
#define __itt_model_iteration_taskA    ITTNOTIFY_VOID(model_iteration_taskA)
#define __itt_model_iteration_taskA_ptr ITTNOTIFY_NAME(model_iteration_taskA)
#define __itt_model_iteration_taskAL    ITTNOTIFY_VOID(model_iteration_taskAL)
#define __itt_model_iteration_taskAL_ptr ITTNOTIFY_NAME(model_iteration_taskAL)
#define __itt_model_task_end        ITTNOTIFY_VOID(model_task_end)
#define __itt_model_task_end_ptr    ITTNOTIFY_NAME(model_task_end)
#define __itt_model_task_end_2        ITTNOTIFY_VOID(model_task_end_2)
#define __itt_model_task_end_2_ptr    ITTNOTIFY_NAME(model_task_end_2)
#else  
#define __itt_model_task_begin(task, instance, name)
#define __itt_model_task_begin_ptr  0
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_model_task_beginW(name)
#define __itt_model_task_beginW_ptr  0
#endif
#define __itt_model_task_beginA(name)
#define __itt_model_task_beginA_ptr  0
#define __itt_model_task_beginAL(name, siteNameLen)
#define __itt_model_task_beginAL_ptr  0
#define __itt_model_iteration_taskA(name)
#define __itt_model_iteration_taskA_ptr  0
#define __itt_model_iteration_taskAL(name, siteNameLen)
#define __itt_model_iteration_taskAL_ptr  0
#define __itt_model_task_end(task, instance)
#define __itt_model_task_end_ptr    0
#define __itt_model_task_end_2()
#define __itt_model_task_end_2_ptr    0
#endif 
#else  
#define __itt_model_task_begin_ptr  0
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_model_task_beginW_ptr 0
#endif
#define __itt_model_task_beginA_ptr  0
#define __itt_model_task_beginAL_ptr  0
#define __itt_model_iteration_taskA_ptr    0
#define __itt_model_iteration_taskAL_ptr    0
#define __itt_model_task_end_ptr    0
#define __itt_model_task_end_2_ptr    0
#endif 



void ITTAPI __itt_model_lock_acquire(void *lock);
void ITTAPI __itt_model_lock_acquire_2(void *lock);
void ITTAPI __itt_model_lock_release(void *lock);
void ITTAPI __itt_model_lock_release_2(void *lock);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_lock_acquire, (void *lock))
ITT_STUBV(ITTAPI, void, model_lock_acquire_2, (void *lock))
ITT_STUBV(ITTAPI, void, model_lock_release, (void *lock))
ITT_STUBV(ITTAPI, void, model_lock_release_2, (void *lock))
#define __itt_model_lock_acquire     ITTNOTIFY_VOID(model_lock_acquire)
#define __itt_model_lock_acquire_ptr ITTNOTIFY_NAME(model_lock_acquire)
#define __itt_model_lock_acquire_2     ITTNOTIFY_VOID(model_lock_acquire_2)
#define __itt_model_lock_acquire_2_ptr ITTNOTIFY_NAME(model_lock_acquire_2)
#define __itt_model_lock_release     ITTNOTIFY_VOID(model_lock_release)
#define __itt_model_lock_release_ptr ITTNOTIFY_NAME(model_lock_release)
#define __itt_model_lock_release_2     ITTNOTIFY_VOID(model_lock_release_2)
#define __itt_model_lock_release_2_ptr ITTNOTIFY_NAME(model_lock_release_2)
#else  
#define __itt_model_lock_acquire(lock)
#define __itt_model_lock_acquire_ptr 0
#define __itt_model_lock_acquire_2(lock)
#define __itt_model_lock_acquire_2_ptr 0
#define __itt_model_lock_release(lock)
#define __itt_model_lock_release_ptr 0
#define __itt_model_lock_release_2(lock)
#define __itt_model_lock_release_2_ptr 0
#endif 
#else  
#define __itt_model_lock_acquire_ptr 0
#define __itt_model_lock_acquire_2_ptr 0
#define __itt_model_lock_release_ptr 0
#define __itt_model_lock_release_2_ptr 0
#endif 



void ITTAPI __itt_model_record_allocation  (void *addr, size_t size);
void ITTAPI __itt_model_record_deallocation(void *addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_record_allocation,   (void *addr, size_t size))
ITT_STUBV(ITTAPI, void, model_record_deallocation, (void *addr))
#define __itt_model_record_allocation       ITTNOTIFY_VOID(model_record_allocation)
#define __itt_model_record_allocation_ptr   ITTNOTIFY_NAME(model_record_allocation)
#define __itt_model_record_deallocation     ITTNOTIFY_VOID(model_record_deallocation)
#define __itt_model_record_deallocation_ptr ITTNOTIFY_NAME(model_record_deallocation)
#else  
#define __itt_model_record_allocation(addr, size)
#define __itt_model_record_allocation_ptr   0
#define __itt_model_record_deallocation(addr)
#define __itt_model_record_deallocation_ptr 0
#endif 
#else  
#define __itt_model_record_allocation_ptr   0
#define __itt_model_record_deallocation_ptr 0
#endif 



void ITTAPI __itt_model_induction_uses(void* addr, size_t size);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_induction_uses, (void *addr, size_t size))
#define __itt_model_induction_uses     ITTNOTIFY_VOID(model_induction_uses)
#define __itt_model_induction_uses_ptr ITTNOTIFY_NAME(model_induction_uses)
#else  
#define __itt_model_induction_uses(addr, size)
#define __itt_model_induction_uses_ptr   0
#endif 
#else  
#define __itt_model_induction_uses_ptr   0
#endif 



void ITTAPI __itt_model_reduction_uses(void* addr, size_t size);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_reduction_uses, (void *addr, size_t size))
#define __itt_model_reduction_uses     ITTNOTIFY_VOID(model_reduction_uses)
#define __itt_model_reduction_uses_ptr ITTNOTIFY_NAME(model_reduction_uses)
#else  
#define __itt_model_reduction_uses(addr, size)
#define __itt_model_reduction_uses_ptr   0
#endif 
#else  
#define __itt_model_reduction_uses_ptr   0
#endif 



void ITTAPI __itt_model_observe_uses(void* addr, size_t size);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_observe_uses, (void *addr, size_t size))
#define __itt_model_observe_uses     ITTNOTIFY_VOID(model_observe_uses)
#define __itt_model_observe_uses_ptr ITTNOTIFY_NAME(model_observe_uses)
#else  
#define __itt_model_observe_uses(addr, size)
#define __itt_model_observe_uses_ptr   0
#endif 
#else  
#define __itt_model_observe_uses_ptr   0
#endif 



void ITTAPI __itt_model_clear_uses(void* addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_clear_uses, (void *addr))
#define __itt_model_clear_uses     ITTNOTIFY_VOID(model_clear_uses)
#define __itt_model_clear_uses_ptr ITTNOTIFY_NAME(model_clear_uses)
#else  
#define __itt_model_clear_uses(addr)
#define __itt_model_clear_uses_ptr 0
#endif 
#else  
#define __itt_model_clear_uses_ptr 0
#endif 



void ITTAPI __itt_model_disable_push(__itt_model_disable x);
void ITTAPI __itt_model_disable_pop(void);
void ITTAPI __itt_model_aggregate_task(size_t x);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_disable_push, (__itt_model_disable x))
ITT_STUBV(ITTAPI, void, model_disable_pop,  (void))
ITT_STUBV(ITTAPI, void, model_aggregate_task, (size_t x))
#define __itt_model_disable_push     ITTNOTIFY_VOID(model_disable_push)
#define __itt_model_disable_push_ptr ITTNOTIFY_NAME(model_disable_push)
#define __itt_model_disable_pop      ITTNOTIFY_VOID(model_disable_pop)
#define __itt_model_disable_pop_ptr  ITTNOTIFY_NAME(model_disable_pop)
#define __itt_model_aggregate_task      ITTNOTIFY_VOID(model_aggregate_task)
#define __itt_model_aggregate_task_ptr  ITTNOTIFY_NAME(model_aggregate_task)
#else  
#define __itt_model_disable_push(x)
#define __itt_model_disable_push_ptr 0
#define __itt_model_disable_pop()
#define __itt_model_disable_pop_ptr 0
#define __itt_model_aggregate_task(x)
#define __itt_model_aggregate_task_ptr 0
#endif 
#else  
#define __itt_model_disable_push_ptr 0
#define __itt_model_disable_pop_ptr 0
#define __itt_model_aggregate_task_ptr 0
#endif 





typedef void* __itt_heap_function;


#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_heap_function ITTAPI __itt_heap_function_createA(const char*    name, const char*    domain);
__itt_heap_function ITTAPI __itt_heap_function_createW(const wchar_t* name, const wchar_t* domain);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_heap_function_create     __itt_heap_function_createW
#  define __itt_heap_function_create_ptr __itt_heap_function_createW_ptr
#else
#  define __itt_heap_function_create     __itt_heap_function_createA
#  define __itt_heap_function_create_ptr __itt_heap_function_createA_ptr
#endif 
#else  
__itt_heap_function ITTAPI __itt_heap_function_create(const char* name, const char* domain);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_heap_function, heap_function_createA, (const char*    name, const char*    domain))
ITT_STUB(ITTAPI, __itt_heap_function, heap_function_createW, (const wchar_t* name, const wchar_t* domain))
#else  
ITT_STUB(ITTAPI, __itt_heap_function, heap_function_create,  (const char*    name, const char*    domain))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_heap_function_createA     ITTNOTIFY_DATA(heap_function_createA)
#define __itt_heap_function_createA_ptr ITTNOTIFY_NAME(heap_function_createA)
#define __itt_heap_function_createW     ITTNOTIFY_DATA(heap_function_createW)
#define __itt_heap_function_createW_ptr ITTNOTIFY_NAME(heap_function_createW)
#else 
#define __itt_heap_function_create      ITTNOTIFY_DATA(heap_function_create)
#define __itt_heap_function_create_ptr  ITTNOTIFY_NAME(heap_function_create)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_heap_function_createA(name, domain) (__itt_heap_function)0
#define __itt_heap_function_createA_ptr 0
#define __itt_heap_function_createW(name, domain) (__itt_heap_function)0
#define __itt_heap_function_createW_ptr 0
#else 
#define __itt_heap_function_create(name, domain)  (__itt_heap_function)0
#define __itt_heap_function_create_ptr  0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_heap_function_createA_ptr 0
#define __itt_heap_function_createW_ptr 0
#else 
#define __itt_heap_function_create_ptr  0
#endif 
#endif 



void ITTAPI __itt_heap_allocate_begin(__itt_heap_function h, size_t size, int initialized);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_allocate_begin, (__itt_heap_function h, size_t size, int initialized))
#define __itt_heap_allocate_begin     ITTNOTIFY_VOID(heap_allocate_begin)
#define __itt_heap_allocate_begin_ptr ITTNOTIFY_NAME(heap_allocate_begin)
#else  
#define __itt_heap_allocate_begin(h, size, initialized)
#define __itt_heap_allocate_begin_ptr   0
#endif 
#else  
#define __itt_heap_allocate_begin_ptr   0
#endif 



void ITTAPI __itt_heap_allocate_end(__itt_heap_function h, void** addr, size_t size, int initialized);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_allocate_end, (__itt_heap_function h, void** addr, size_t size, int initialized))
#define __itt_heap_allocate_end     ITTNOTIFY_VOID(heap_allocate_end)
#define __itt_heap_allocate_end_ptr ITTNOTIFY_NAME(heap_allocate_end)
#else  
#define __itt_heap_allocate_end(h, addr, size, initialized)
#define __itt_heap_allocate_end_ptr   0
#endif 
#else  
#define __itt_heap_allocate_end_ptr   0
#endif 



void ITTAPI __itt_heap_free_begin(__itt_heap_function h, void* addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_free_begin, (__itt_heap_function h, void* addr))
#define __itt_heap_free_begin     ITTNOTIFY_VOID(heap_free_begin)
#define __itt_heap_free_begin_ptr ITTNOTIFY_NAME(heap_free_begin)
#else  
#define __itt_heap_free_begin(h, addr)
#define __itt_heap_free_begin_ptr   0
#endif 
#else  
#define __itt_heap_free_begin_ptr   0
#endif 



void ITTAPI __itt_heap_free_end(__itt_heap_function h, void* addr);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_free_end, (__itt_heap_function h, void* addr))
#define __itt_heap_free_end     ITTNOTIFY_VOID(heap_free_end)
#define __itt_heap_free_end_ptr ITTNOTIFY_NAME(heap_free_end)
#else  
#define __itt_heap_free_end(h, addr)
#define __itt_heap_free_end_ptr   0
#endif 
#else  
#define __itt_heap_free_end_ptr   0
#endif 



void ITTAPI __itt_heap_reallocate_begin(__itt_heap_function h, void* addr, size_t new_size, int initialized);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_reallocate_begin, (__itt_heap_function h, void* addr, size_t new_size, int initialized))
#define __itt_heap_reallocate_begin     ITTNOTIFY_VOID(heap_reallocate_begin)
#define __itt_heap_reallocate_begin_ptr ITTNOTIFY_NAME(heap_reallocate_begin)
#else  
#define __itt_heap_reallocate_begin(h, addr, new_size, initialized)
#define __itt_heap_reallocate_begin_ptr   0
#endif 
#else  
#define __itt_heap_reallocate_begin_ptr   0
#endif 



void ITTAPI __itt_heap_reallocate_end(__itt_heap_function h, void* addr, void** new_addr, size_t new_size, int initialized);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_reallocate_end, (__itt_heap_function h, void* addr, void** new_addr, size_t new_size, int initialized))
#define __itt_heap_reallocate_end     ITTNOTIFY_VOID(heap_reallocate_end)
#define __itt_heap_reallocate_end_ptr ITTNOTIFY_NAME(heap_reallocate_end)
#else  
#define __itt_heap_reallocate_end(h, addr, new_addr, new_size, initialized)
#define __itt_heap_reallocate_end_ptr   0
#endif 
#else  
#define __itt_heap_reallocate_end_ptr   0
#endif 



void ITTAPI __itt_heap_internal_access_begin(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_internal_access_begin,  (void))
#define __itt_heap_internal_access_begin      ITTNOTIFY_VOID(heap_internal_access_begin)
#define __itt_heap_internal_access_begin_ptr  ITTNOTIFY_NAME(heap_internal_access_begin)
#else  
#define __itt_heap_internal_access_begin()
#define __itt_heap_internal_access_begin_ptr  0
#endif 
#else  
#define __itt_heap_internal_access_begin_ptr  0
#endif 



void ITTAPI __itt_heap_internal_access_end(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_internal_access_end, (void))
#define __itt_heap_internal_access_end     ITTNOTIFY_VOID(heap_internal_access_end)
#define __itt_heap_internal_access_end_ptr ITTNOTIFY_NAME(heap_internal_access_end)
#else  
#define __itt_heap_internal_access_end()
#define __itt_heap_internal_access_end_ptr 0
#endif 
#else  
#define __itt_heap_internal_access_end_ptr 0
#endif 



void ITTAPI __itt_heap_record_memory_growth_begin(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_record_memory_growth_begin,  (void))
#define __itt_heap_record_memory_growth_begin      ITTNOTIFY_VOID(heap_record_memory_growth_begin)
#define __itt_heap_record_memory_growth_begin_ptr  ITTNOTIFY_NAME(heap_record_memory_growth_begin)
#else  
#define __itt_heap_record_memory_growth_begin()
#define __itt_heap_record_memory_growth_begin_ptr  0
#endif 
#else  
#define __itt_heap_record_memory_growth_begin_ptr  0
#endif 



void ITTAPI __itt_heap_record_memory_growth_end(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_record_memory_growth_end, (void))
#define __itt_heap_record_memory_growth_end     ITTNOTIFY_VOID(heap_record_memory_growth_end)
#define __itt_heap_record_memory_growth_end_ptr ITTNOTIFY_NAME(heap_record_memory_growth_end)
#else  
#define __itt_heap_record_memory_growth_end()
#define __itt_heap_record_memory_growth_end_ptr 0
#endif 
#else  
#define __itt_heap_record_memory_growth_end_ptr 0
#endif 




#define __itt_heap_leaks 0x00000001


#define __itt_heap_growth 0x00000002



void ITTAPI __itt_heap_reset_detection(unsigned int reset_mask);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_reset_detection,  (unsigned int reset_mask))
#define __itt_heap_reset_detection      ITTNOTIFY_VOID(heap_reset_detection)
#define __itt_heap_reset_detection_ptr  ITTNOTIFY_NAME(heap_reset_detection)
#else  
#define __itt_heap_reset_detection()
#define __itt_heap_reset_detection_ptr  0
#endif 
#else  
#define __itt_heap_reset_detection_ptr  0
#endif 



void ITTAPI __itt_heap_record(unsigned int record_mask);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_record, (unsigned int record_mask))
#define __itt_heap_record     ITTNOTIFY_VOID(heap_record)
#define __itt_heap_record_ptr ITTNOTIFY_NAME(heap_record)
#else  
#define __itt_heap_record()
#define __itt_heap_record_ptr 0
#endif 
#else  
#define __itt_heap_record_ptr 0
#endif 









#pragma pack(push, 8)

typedef struct ___itt_domain
{
volatile int flags; 
const char* nameA;  
#if defined(UNICODE) || defined(_UNICODE)
const wchar_t* nameW; 
#else  
void* nameW;
#endif 
int   extra1; 
void* extra2; 
struct ___itt_domain* next;
} __itt_domain;

#pragma pack(pop)



#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_domain* ITTAPI __itt_domain_createA(const char    *name);
__itt_domain* ITTAPI __itt_domain_createW(const wchar_t *name);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_domain_create     __itt_domain_createW
#  define __itt_domain_create_ptr __itt_domain_createW_ptr
#else 
#  define __itt_domain_create     __itt_domain_createA
#  define __itt_domain_create_ptr __itt_domain_createA_ptr
#endif 
#else  
__itt_domain* ITTAPI __itt_domain_create(const char *name);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_domain*, domain_createA, (const char    *name))
ITT_STUB(ITTAPI, __itt_domain*, domain_createW, (const wchar_t *name))
#else  
ITT_STUB(ITTAPI, __itt_domain*, domain_create,  (const char    *name))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_domain_createA     ITTNOTIFY_DATA(domain_createA)
#define __itt_domain_createA_ptr ITTNOTIFY_NAME(domain_createA)
#define __itt_domain_createW     ITTNOTIFY_DATA(domain_createW)
#define __itt_domain_createW_ptr ITTNOTIFY_NAME(domain_createW)
#else 
#define __itt_domain_create     ITTNOTIFY_DATA(domain_create)
#define __itt_domain_create_ptr ITTNOTIFY_NAME(domain_create)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_domain_createA(name) (__itt_domain*)0
#define __itt_domain_createA_ptr 0
#define __itt_domain_createW(name) (__itt_domain*)0
#define __itt_domain_createW_ptr 0
#else 
#define __itt_domain_create(name)  (__itt_domain*)0
#define __itt_domain_create_ptr 0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_domain_createA_ptr 0
#define __itt_domain_createW_ptr 0
#else 
#define __itt_domain_create_ptr  0
#endif 
#endif 






#pragma pack(push, 8)

typedef struct ___itt_id
{
unsigned long long d1, d2, d3;
} __itt_id;

#pragma pack(pop)


static const __itt_id __itt_null = { 0, 0, 0 };



ITT_INLINE __itt_id ITTAPI __itt_id_make(void* addr, unsigned long long extra) ITT_INLINE_ATTRIBUTE;
ITT_INLINE __itt_id ITTAPI __itt_id_make(void* addr, unsigned long long extra)
{
__itt_id id = __itt_null;
id.d1 = (unsigned long long)((uintptr_t)addr);
id.d2 = (unsigned long long)extra;
id.d3 = (unsigned long long)0; 
return id;
}


void ITTAPI __itt_id_create(const __itt_domain *domain, __itt_id id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, id_create, (const __itt_domain *domain, __itt_id id))
#define __itt_id_create(d,x) ITTNOTIFY_VOID_D1(id_create,d,x)
#define __itt_id_create_ptr  ITTNOTIFY_NAME(id_create)
#else  
#define __itt_id_create(domain,id)
#define __itt_id_create_ptr 0
#endif 
#else  
#define __itt_id_create_ptr 0
#endif 



void ITTAPI __itt_id_destroy(const __itt_domain *domain, __itt_id id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, id_destroy, (const __itt_domain *domain, __itt_id id))
#define __itt_id_destroy(d,x) ITTNOTIFY_VOID_D1(id_destroy,d,x)
#define __itt_id_destroy_ptr  ITTNOTIFY_NAME(id_destroy)
#else  
#define __itt_id_destroy(domain,id)
#define __itt_id_destroy_ptr 0
#endif 
#else  
#define __itt_id_destroy_ptr 0
#endif 






#pragma pack(push, 8)

typedef struct ___itt_string_handle
{
const char* strA; 
#if defined(UNICODE) || defined(_UNICODE)
const wchar_t* strW; 
#else  
void* strW;
#endif 
int   extra1; 
void* extra2; 
struct ___itt_string_handle* next;
} __itt_string_handle;

#pragma pack(pop)



#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_string_handle* ITTAPI __itt_string_handle_createA(const char    *name);
__itt_string_handle* ITTAPI __itt_string_handle_createW(const wchar_t *name);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_string_handle_create     __itt_string_handle_createW
#  define __itt_string_handle_create_ptr __itt_string_handle_createW_ptr
#else 
#  define __itt_string_handle_create     __itt_string_handle_createA
#  define __itt_string_handle_create_ptr __itt_string_handle_createA_ptr
#endif 
#else  
__itt_string_handle* ITTAPI __itt_string_handle_create(const char *name);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_string_handle*, string_handle_createA, (const char    *name))
ITT_STUB(ITTAPI, __itt_string_handle*, string_handle_createW, (const wchar_t *name))
#else  
ITT_STUB(ITTAPI, __itt_string_handle*, string_handle_create,  (const char    *name))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_string_handle_createA     ITTNOTIFY_DATA(string_handle_createA)
#define __itt_string_handle_createA_ptr ITTNOTIFY_NAME(string_handle_createA)
#define __itt_string_handle_createW     ITTNOTIFY_DATA(string_handle_createW)
#define __itt_string_handle_createW_ptr ITTNOTIFY_NAME(string_handle_createW)
#else 
#define __itt_string_handle_create     ITTNOTIFY_DATA(string_handle_create)
#define __itt_string_handle_create_ptr ITTNOTIFY_NAME(string_handle_create)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_string_handle_createA(name) (__itt_string_handle*)0
#define __itt_string_handle_createA_ptr 0
#define __itt_string_handle_createW(name) (__itt_string_handle*)0
#define __itt_string_handle_createW_ptr 0
#else 
#define __itt_string_handle_create(name)  (__itt_string_handle*)0
#define __itt_string_handle_create_ptr 0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_string_handle_createA_ptr 0
#define __itt_string_handle_createW_ptr 0
#else 
#define __itt_string_handle_create_ptr  0
#endif 
#endif 




typedef unsigned long long __itt_timestamp;


#define __itt_timestamp_none ((__itt_timestamp)-1LL)




__itt_timestamp ITTAPI __itt_get_timestamp(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, __itt_timestamp, get_timestamp, (void))
#define __itt_get_timestamp      ITTNOTIFY_DATA(get_timestamp)
#define __itt_get_timestamp_ptr  ITTNOTIFY_NAME(get_timestamp)
#else  
#define __itt_get_timestamp()
#define __itt_get_timestamp_ptr 0
#endif 
#else  
#define __itt_get_timestamp_ptr 0
#endif 








void ITTAPI __itt_region_begin(const __itt_domain *domain, __itt_id id, __itt_id parentid, __itt_string_handle *name);


void ITTAPI __itt_region_end(const __itt_domain *domain, __itt_id id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, region_begin, (const __itt_domain *domain, __itt_id id, __itt_id parentid, __itt_string_handle *name))
ITT_STUBV(ITTAPI, void, region_end,   (const __itt_domain *domain, __itt_id id))
#define __itt_region_begin(d,x,y,z) ITTNOTIFY_VOID_D3(region_begin,d,x,y,z)
#define __itt_region_begin_ptr      ITTNOTIFY_NAME(region_begin)
#define __itt_region_end(d,x)       ITTNOTIFY_VOID_D1(region_end,d,x)
#define __itt_region_end_ptr        ITTNOTIFY_NAME(region_end)
#else  
#define __itt_region_begin(d,x,y,z)
#define __itt_region_begin_ptr 0
#define __itt_region_end(d,x)
#define __itt_region_end_ptr   0
#endif 
#else  
#define __itt_region_begin_ptr 0
#define __itt_region_end_ptr   0
#endif 






void ITTAPI __itt_frame_begin_v3(const __itt_domain *domain, __itt_id *id);


void ITTAPI __itt_frame_end_v3(const __itt_domain *domain, __itt_id *id);


void ITTAPI __itt_frame_submit_v3(const __itt_domain *domain, __itt_id *id,
__itt_timestamp begin, __itt_timestamp end);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, frame_begin_v3,  (const __itt_domain *domain, __itt_id *id))
ITT_STUBV(ITTAPI, void, frame_end_v3,    (const __itt_domain *domain, __itt_id *id))
ITT_STUBV(ITTAPI, void, frame_submit_v3, (const __itt_domain *domain, __itt_id *id, __itt_timestamp begin, __itt_timestamp end))
#define __itt_frame_begin_v3(d,x)      ITTNOTIFY_VOID_D1(frame_begin_v3,d,x)
#define __itt_frame_begin_v3_ptr       ITTNOTIFY_NAME(frame_begin_v3)
#define __itt_frame_end_v3(d,x)        ITTNOTIFY_VOID_D1(frame_end_v3,d,x)
#define __itt_frame_end_v3_ptr         ITTNOTIFY_NAME(frame_end_v3)
#define __itt_frame_submit_v3(d,x,b,e) ITTNOTIFY_VOID_D3(frame_submit_v3,d,x,b,e)
#define __itt_frame_submit_v3_ptr      ITTNOTIFY_NAME(frame_submit_v3)
#else  
#define __itt_frame_begin_v3(domain,id)
#define __itt_frame_begin_v3_ptr 0
#define __itt_frame_end_v3(domain,id)
#define __itt_frame_end_v3_ptr   0
#define __itt_frame_submit_v3(domain,id,begin,end)
#define __itt_frame_submit_v3_ptr   0
#endif 
#else  
#define __itt_frame_begin_v3_ptr 0
#define __itt_frame_end_v3_ptr   0
#define __itt_frame_submit_v3_ptr   0
#endif 






void ITTAPI __itt_task_group(const __itt_domain *domain, __itt_id id, __itt_id parentid, __itt_string_handle *name);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, task_group, (const __itt_domain *domain, __itt_id id, __itt_id parentid, __itt_string_handle *name))
#define __itt_task_group(d,x,y,z) ITTNOTIFY_VOID_D3(task_group,d,x,y,z)
#define __itt_task_group_ptr      ITTNOTIFY_NAME(task_group)
#else  
#define __itt_task_group(d,x,y,z)
#define __itt_task_group_ptr 0
#endif 
#else  
#define __itt_task_group_ptr 0
#endif 






void ITTAPI __itt_task_begin(const __itt_domain *domain, __itt_id taskid, __itt_id parentid, __itt_string_handle *name);


void ITTAPI __itt_task_begin_fn(const __itt_domain *domain, __itt_id taskid, __itt_id parentid, void* fn);


void ITTAPI __itt_task_end(const __itt_domain *domain);


void ITTAPI __itt_task_begin_overlapped(const __itt_domain* domain, __itt_id taskid, __itt_id parentid, __itt_string_handle* name);


void ITTAPI __itt_task_end_overlapped(const __itt_domain *domain, __itt_id taskid);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, task_begin,    (const __itt_domain *domain, __itt_id id, __itt_id parentid, __itt_string_handle *name))
ITT_STUBV(ITTAPI, void, task_begin_fn, (const __itt_domain *domain, __itt_id id, __itt_id parentid, void* fn))
ITT_STUBV(ITTAPI, void, task_end,      (const __itt_domain *domain))
ITT_STUBV(ITTAPI, void, task_begin_overlapped, (const __itt_domain *domain, __itt_id taskid, __itt_id parentid, __itt_string_handle *name))
ITT_STUBV(ITTAPI, void, task_end_overlapped,   (const __itt_domain *domain, __itt_id taskid))
#define __itt_task_begin(d,x,y,z)    ITTNOTIFY_VOID_D3(task_begin,d,x,y,z)
#define __itt_task_begin_ptr         ITTNOTIFY_NAME(task_begin)
#define __itt_task_begin_fn(d,x,y,z) ITTNOTIFY_VOID_D3(task_begin_fn,d,x,y,z)
#define __itt_task_begin_fn_ptr      ITTNOTIFY_NAME(task_begin_fn)
#define __itt_task_end(d)            ITTNOTIFY_VOID_D0(task_end,d)
#define __itt_task_end_ptr           ITTNOTIFY_NAME(task_end)
#define __itt_task_begin_overlapped(d,x,y,z) ITTNOTIFY_VOID_D3(task_begin_overlapped,d,x,y,z)
#define __itt_task_begin_overlapped_ptr      ITTNOTIFY_NAME(task_begin_overlapped)
#define __itt_task_end_overlapped(d,x)       ITTNOTIFY_VOID_D1(task_end_overlapped,d,x)
#define __itt_task_end_overlapped_ptr        ITTNOTIFY_NAME(task_end_overlapped)
#else  
#define __itt_task_begin(domain,id,parentid,name)
#define __itt_task_begin_ptr    0
#define __itt_task_begin_fn(domain,id,parentid,fn)
#define __itt_task_begin_fn_ptr 0
#define __itt_task_end(domain)
#define __itt_task_end_ptr      0
#define __itt_task_begin_overlapped(domain,taskid,parentid,name)
#define __itt_task_begin_overlapped_ptr         0
#define __itt_task_end_overlapped(domain,taskid)
#define __itt_task_end_overlapped_ptr           0
#endif 
#else  
#define __itt_task_begin_ptr    0
#define __itt_task_begin_fn_ptr 0
#define __itt_task_end_ptr      0
#define __itt_task_begin_overlapped_ptr 0
#define __itt_task_end_overlapped_ptr   0
#endif 






void ITTAPI __itt_counter_inc_v3(const __itt_domain *domain, __itt_string_handle *name);


void ITTAPI __itt_counter_inc_delta_v3(const __itt_domain *domain, __itt_string_handle *name, unsigned long long delta);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, counter_inc_v3,       (const __itt_domain *domain, __itt_string_handle *name))
ITT_STUBV(ITTAPI, void, counter_inc_delta_v3, (const __itt_domain *domain, __itt_string_handle *name, unsigned long long delta))
#define __itt_counter_inc_v3(d,x)         ITTNOTIFY_VOID_D1(counter_inc_v3,d,x)
#define __itt_counter_inc_v3_ptr          ITTNOTIFY_NAME(counter_inc_v3)
#define __itt_counter_inc_delta_v3(d,x,y) ITTNOTIFY_VOID_D2(counter_inc_delta_v3,d,x,y)
#define __itt_counter_inc_delta_v3_ptr    ITTNOTIFY_NAME(counter_inc_delta_v3)
#else  
#define __itt_counter_inc_v3(domain,name)
#define __itt_counter_inc_v3_ptr       0
#define __itt_counter_inc_delta_v3(domain,name,delta)
#define __itt_counter_inc_delta_v3_ptr 0
#endif 
#else  
#define __itt_counter_inc_v3_ptr       0
#define __itt_counter_inc_delta_v3_ptr 0
#endif 






typedef enum
{
__itt_scope_unknown = 0,
__itt_scope_global,
__itt_scope_track_group,
__itt_scope_track,
__itt_scope_task,
__itt_scope_marker
} __itt_scope;


#define __itt_marker_scope_unknown  __itt_scope_unknown
#define __itt_marker_scope_global   __itt_scope_global
#define __itt_marker_scope_process  __itt_scope_track_group
#define __itt_marker_scope_thread   __itt_scope_track
#define __itt_marker_scope_task     __itt_scope_task



void ITTAPI __itt_marker(const __itt_domain *domain, __itt_id id, __itt_string_handle *name, __itt_scope scope);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, marker, (const __itt_domain *domain, __itt_id id, __itt_string_handle *name, __itt_scope scope))
#define __itt_marker(d,x,y,z) ITTNOTIFY_VOID_D3(marker,d,x,y,z)
#define __itt_marker_ptr      ITTNOTIFY_NAME(marker)
#else  
#define __itt_marker(domain,id,name,scope)
#define __itt_marker_ptr 0
#endif 
#else  
#define __itt_marker_ptr 0
#endif 






typedef enum {
__itt_metadata_unknown = 0,
__itt_metadata_u64,     
__itt_metadata_s64,     
__itt_metadata_u32,     
__itt_metadata_s32,     
__itt_metadata_u16,     
__itt_metadata_s16,     
__itt_metadata_float,   
__itt_metadata_double   
} __itt_metadata_type;


void ITTAPI __itt_metadata_add(const __itt_domain *domain, __itt_id id, __itt_string_handle *key, __itt_metadata_type type, size_t count, void *data);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, metadata_add, (const __itt_domain *domain, __itt_id id, __itt_string_handle *key, __itt_metadata_type type, size_t count, void *data))
#define __itt_metadata_add(d,x,y,z,a,b) ITTNOTIFY_VOID_D5(metadata_add,d,x,y,z,a,b)
#define __itt_metadata_add_ptr          ITTNOTIFY_NAME(metadata_add)
#else  
#define __itt_metadata_add(d,x,y,z,a,b)
#define __itt_metadata_add_ptr 0
#endif 
#else  
#define __itt_metadata_add_ptr 0
#endif 



#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_metadata_str_addA(const __itt_domain *domain, __itt_id id, __itt_string_handle *key, const char *data, size_t length);
void ITTAPI __itt_metadata_str_addW(const __itt_domain *domain, __itt_id id, __itt_string_handle *key, const wchar_t *data, size_t length);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_metadata_str_add     __itt_metadata_str_addW
#  define __itt_metadata_str_add_ptr __itt_metadata_str_addW_ptr
#else 
#  define __itt_metadata_str_add     __itt_metadata_str_addA
#  define __itt_metadata_str_add_ptr __itt_metadata_str_addA_ptr
#endif 
#else 
void ITTAPI __itt_metadata_str_add(const __itt_domain *domain, __itt_id id, __itt_string_handle *key, const char *data, size_t length);
#endif


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, metadata_str_addA, (const __itt_domain *domain, __itt_id id, __itt_string_handle *key, const char *data, size_t length))
ITT_STUBV(ITTAPI, void, metadata_str_addW, (const __itt_domain *domain, __itt_id id, __itt_string_handle *key, const wchar_t *data, size_t length))
#else  
ITT_STUBV(ITTAPI, void, metadata_str_add, (const __itt_domain *domain, __itt_id id, __itt_string_handle *key, const char *data, size_t length))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_metadata_str_addA(d,x,y,z,a) ITTNOTIFY_VOID_D4(metadata_str_addA,d,x,y,z,a)
#define __itt_metadata_str_addA_ptr        ITTNOTIFY_NAME(metadata_str_addA)
#define __itt_metadata_str_addW(d,x,y,z,a) ITTNOTIFY_VOID_D4(metadata_str_addW,d,x,y,z,a)
#define __itt_metadata_str_addW_ptr        ITTNOTIFY_NAME(metadata_str_addW)
#else 
#define __itt_metadata_str_add(d,x,y,z,a)  ITTNOTIFY_VOID_D4(metadata_str_add,d,x,y,z,a)
#define __itt_metadata_str_add_ptr         ITTNOTIFY_NAME(metadata_str_add)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_metadata_str_addA(d,x,y,z,a)
#define __itt_metadata_str_addA_ptr 0
#define __itt_metadata_str_addW(d,x,y,z,a)
#define __itt_metadata_str_addW_ptr 0
#else 
#define __itt_metadata_str_add(d,x,y,z,a)
#define __itt_metadata_str_add_ptr 0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_metadata_str_addA_ptr 0
#define __itt_metadata_str_addW_ptr 0
#else 
#define __itt_metadata_str_add_ptr  0
#endif 
#endif 



void ITTAPI __itt_metadata_add_with_scope(const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, __itt_metadata_type type, size_t count, void *data);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, metadata_add_with_scope, (const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, __itt_metadata_type type, size_t count, void *data))
#define __itt_metadata_add_with_scope(d,x,y,z,a,b) ITTNOTIFY_VOID_D5(metadata_add_with_scope,d,x,y,z,a,b)
#define __itt_metadata_add_with_scope_ptr          ITTNOTIFY_NAME(metadata_add_with_scope)
#else  
#define __itt_metadata_add_with_scope(d,x,y,z,a,b)
#define __itt_metadata_add_with_scope_ptr 0
#endif 
#else  
#define __itt_metadata_add_with_scope_ptr 0
#endif 



#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_metadata_str_add_with_scopeA(const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, const char *data, size_t length);
void ITTAPI __itt_metadata_str_add_with_scopeW(const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, const wchar_t *data, size_t length);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_metadata_str_add_with_scope     __itt_metadata_str_add_with_scopeW
#  define __itt_metadata_str_add_with_scope_ptr __itt_metadata_str_add_with_scopeW_ptr
#else 
#  define __itt_metadata_str_add_with_scope     __itt_metadata_str_add_with_scopeA
#  define __itt_metadata_str_add_with_scope_ptr __itt_metadata_str_add_with_scopeA_ptr
#endif 
#else 
void ITTAPI __itt_metadata_str_add_with_scope(const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, const char *data, size_t length);
#endif


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, metadata_str_add_with_scopeA, (const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, const char *data, size_t length))
ITT_STUBV(ITTAPI, void, metadata_str_add_with_scopeW, (const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, const wchar_t *data, size_t length))
#else  
ITT_STUBV(ITTAPI, void, metadata_str_add_with_scope, (const __itt_domain *domain, __itt_scope scope, __itt_string_handle *key, const char *data, size_t length))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_metadata_str_add_with_scopeA(d,x,y,z,a) ITTNOTIFY_VOID_D4(metadata_str_add_with_scopeA,d,x,y,z,a)
#define __itt_metadata_str_add_with_scopeA_ptr        ITTNOTIFY_NAME(metadata_str_add_with_scopeA)
#define __itt_metadata_str_add_with_scopeW(d,x,y,z,a) ITTNOTIFY_VOID_D4(metadata_str_add_with_scopeW,d,x,y,z,a)
#define __itt_metadata_str_add_with_scopeW_ptr        ITTNOTIFY_NAME(metadata_str_add_with_scopeW)
#else 
#define __itt_metadata_str_add_with_scope(d,x,y,z,a)  ITTNOTIFY_VOID_D4(metadata_str_add_with_scope,d,x,y,z,a)
#define __itt_metadata_str_add_with_scope_ptr         ITTNOTIFY_NAME(metadata_str_add_with_scope)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_metadata_str_add_with_scopeA(d,x,y,z,a)
#define __itt_metadata_str_add_with_scopeA_ptr  0
#define __itt_metadata_str_add_with_scopeW(d,x,y,z,a)
#define __itt_metadata_str_add_with_scopeW_ptr  0
#else 
#define __itt_metadata_str_add_with_scope(d,x,y,z,a)
#define __itt_metadata_str_add_with_scope_ptr   0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_metadata_str_add_with_scopeA_ptr  0
#define __itt_metadata_str_add_with_scopeW_ptr  0
#else 
#define __itt_metadata_str_add_with_scope_ptr   0
#endif 
#endif 







typedef enum
{
__itt_relation_is_unknown = 0,
__itt_relation_is_dependent_on,         
__itt_relation_is_sibling_of,           
__itt_relation_is_parent_of,            
__itt_relation_is_continuation_of,      
__itt_relation_is_child_of,             
__itt_relation_is_continued_by,         
__itt_relation_is_predecessor_to        
} __itt_relation;


void ITTAPI __itt_relation_add_to_current(const __itt_domain *domain, __itt_relation relation, __itt_id tail);


void ITTAPI __itt_relation_add(const __itt_domain *domain, __itt_id head, __itt_relation relation, __itt_id tail);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, relation_add_to_current, (const __itt_domain *domain, __itt_relation relation, __itt_id tail))
ITT_STUBV(ITTAPI, void, relation_add,            (const __itt_domain *domain, __itt_id head, __itt_relation relation, __itt_id tail))
#define __itt_relation_add_to_current(d,x,y) ITTNOTIFY_VOID_D2(relation_add_to_current,d,x,y)
#define __itt_relation_add_to_current_ptr    ITTNOTIFY_NAME(relation_add_to_current)
#define __itt_relation_add(d,x,y,z)          ITTNOTIFY_VOID_D3(relation_add,d,x,y,z)
#define __itt_relation_add_ptr               ITTNOTIFY_NAME(relation_add)
#else  
#define __itt_relation_add_to_current(d,x,y)
#define __itt_relation_add_to_current_ptr 0
#define __itt_relation_add(d,x,y,z)
#define __itt_relation_add_ptr 0
#endif 
#else  
#define __itt_relation_add_to_current_ptr 0
#define __itt_relation_add_ptr 0
#endif 




#pragma pack(push, 8)

typedef struct ___itt_clock_info
{
unsigned long long clock_freq; 
unsigned long long clock_base; 
} __itt_clock_info;

#pragma pack(pop)



typedef void (ITTAPI *__itt_get_clock_info_fn)(__itt_clock_info* clock_info, void* data);



#pragma pack(push, 8)

typedef struct ___itt_clock_domain
{
__itt_clock_info info;      
__itt_get_clock_info_fn fn; 
void* fn_data;              
int   extra1;               
void* extra2;               
struct ___itt_clock_domain* next;
} __itt_clock_domain;

#pragma pack(pop)



__itt_clock_domain* ITTAPI __itt_clock_domain_create(__itt_get_clock_info_fn fn, void* fn_data);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, __itt_clock_domain*, clock_domain_create, (__itt_get_clock_info_fn fn, void* fn_data))
#define __itt_clock_domain_create     ITTNOTIFY_DATA(clock_domain_create)
#define __itt_clock_domain_create_ptr ITTNOTIFY_NAME(clock_domain_create)
#else  
#define __itt_clock_domain_create(fn,fn_data) (__itt_clock_domain*)0
#define __itt_clock_domain_create_ptr 0
#endif 
#else  
#define __itt_clock_domain_create_ptr 0
#endif 



void ITTAPI __itt_clock_domain_reset(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, clock_domain_reset, (void))
#define __itt_clock_domain_reset     ITTNOTIFY_VOID(clock_domain_reset)
#define __itt_clock_domain_reset_ptr ITTNOTIFY_NAME(clock_domain_reset)
#else  
#define __itt_clock_domain_reset()
#define __itt_clock_domain_reset_ptr 0
#endif 
#else  
#define __itt_clock_domain_reset_ptr 0
#endif 



void ITTAPI __itt_id_create_ex(const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id);


void ITTAPI __itt_id_destroy_ex(const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, id_create_ex,  (const __itt_domain *domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id))
ITT_STUBV(ITTAPI, void, id_destroy_ex, (const __itt_domain *domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id))
#define __itt_id_create_ex(d,x,y,z)  ITTNOTIFY_VOID_D3(id_create_ex,d,x,y,z)
#define __itt_id_create_ex_ptr       ITTNOTIFY_NAME(id_create_ex)
#define __itt_id_destroy_ex(d,x,y,z) ITTNOTIFY_VOID_D3(id_destroy_ex,d,x,y,z)
#define __itt_id_destroy_ex_ptr      ITTNOTIFY_NAME(id_destroy_ex)
#else  
#define __itt_id_create_ex(domain,clock_domain,timestamp,id)
#define __itt_id_create_ex_ptr    0
#define __itt_id_destroy_ex(domain,clock_domain,timestamp,id)
#define __itt_id_destroy_ex_ptr 0
#endif 
#else  
#define __itt_id_create_ex_ptr    0
#define __itt_id_destroy_ex_ptr 0
#endif 



void ITTAPI __itt_task_begin_ex(const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id taskid, __itt_id parentid, __itt_string_handle* name);


void ITTAPI __itt_task_begin_fn_ex(const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id taskid, __itt_id parentid, void* fn);


void ITTAPI __itt_task_end_ex(const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, task_begin_ex,        (const __itt_domain *domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id, __itt_id parentid, __itt_string_handle *name))
ITT_STUBV(ITTAPI, void, task_begin_fn_ex,     (const __itt_domain *domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id, __itt_id parentid, void* fn))
ITT_STUBV(ITTAPI, void, task_end_ex,          (const __itt_domain *domain, __itt_clock_domain* clock_domain, unsigned long long timestamp))
#define __itt_task_begin_ex(d,x,y,z,a,b)      ITTNOTIFY_VOID_D5(task_begin_ex,d,x,y,z,a,b)
#define __itt_task_begin_ex_ptr               ITTNOTIFY_NAME(task_begin_ex)
#define __itt_task_begin_fn_ex(d,x,y,z,a,b)   ITTNOTIFY_VOID_D5(task_begin_fn_ex,d,x,y,z,a,b)
#define __itt_task_begin_fn_ex_ptr            ITTNOTIFY_NAME(task_begin_fn_ex)
#define __itt_task_end_ex(d,x,y)              ITTNOTIFY_VOID_D2(task_end_ex,d,x,y)
#define __itt_task_end_ex_ptr                 ITTNOTIFY_NAME(task_end_ex)
#else  
#define __itt_task_begin_ex(domain,clock_domain,timestamp,id,parentid,name)
#define __itt_task_begin_ex_ptr          0
#define __itt_task_begin_fn_ex(domain,clock_domain,timestamp,id,parentid,fn)
#define __itt_task_begin_fn_ex_ptr       0
#define __itt_task_end_ex(domain,clock_domain,timestamp)
#define __itt_task_end_ex_ptr            0
#endif 
#else  
#define __itt_task_begin_ex_ptr          0
#define __itt_task_begin_fn_ex_ptr       0
#define __itt_task_end_ex_ptr            0
#endif 



void ITTAPI __itt_marker_ex(const __itt_domain *domain,  __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id, __itt_string_handle *name, __itt_scope scope);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, marker_ex,    (const __itt_domain *domain,  __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id id, __itt_string_handle *name, __itt_scope scope))
#define __itt_marker_ex(d,x,y,z,a,b)    ITTNOTIFY_VOID_D5(marker_ex,d,x,y,z,a,b)
#define __itt_marker_ex_ptr             ITTNOTIFY_NAME(marker_ex)
#else  
#define __itt_marker_ex(domain,clock_domain,timestamp,id,name,scope)
#define __itt_marker_ex_ptr    0
#endif 
#else  
#define __itt_marker_ex_ptr    0
#endif 



void ITTAPI __itt_relation_add_to_current_ex(const __itt_domain *domain,  __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_relation relation, __itt_id tail);


void ITTAPI __itt_relation_add_ex(const __itt_domain *domain,  __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id head, __itt_relation relation, __itt_id tail);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, relation_add_to_current_ex, (const __itt_domain *domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_relation relation, __itt_id tail))
ITT_STUBV(ITTAPI, void, relation_add_ex,            (const __itt_domain *domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id head, __itt_relation relation, __itt_id tail))
#define __itt_relation_add_to_current_ex(d,x,y,z,a) ITTNOTIFY_VOID_D4(relation_add_to_current_ex,d,x,y,z,a)
#define __itt_relation_add_to_current_ex_ptr        ITTNOTIFY_NAME(relation_add_to_current_ex)
#define __itt_relation_add_ex(d,x,y,z,a,b)          ITTNOTIFY_VOID_D5(relation_add_ex,d,x,y,z,a,b)
#define __itt_relation_add_ex_ptr                   ITTNOTIFY_NAME(relation_add_ex)
#else  
#define __itt_relation_add_to_current_ex(domain,clock_domain,timestame,relation,tail)
#define __itt_relation_add_to_current_ex_ptr 0
#define __itt_relation_add_ex(domain,clock_domain,timestamp,head,relation,tail)
#define __itt_relation_add_ex_ptr 0
#endif 
#else  
#define __itt_relation_add_to_current_ex_ptr 0
#define __itt_relation_add_ex_ptr 0
#endif 



typedef enum ___itt_track_group_type
{
__itt_track_group_type_normal = 0
} __itt_track_group_type;



#pragma pack(push, 8)

typedef struct ___itt_track_group
{
__itt_string_handle* name;     
struct ___itt_track* track;    
__itt_track_group_type tgtype; 
int   extra1;                  
void* extra2;                  
struct ___itt_track_group* next;
} __itt_track_group;

#pragma pack(pop)



typedef enum ___itt_track_type
{
__itt_track_type_normal = 0
#ifdef INTEL_ITTNOTIFY_API_PRIVATE
, __itt_track_type_queue
#endif 
} __itt_track_type;


#pragma pack(push, 8)

typedef struct ___itt_track
{
__itt_string_handle* name; 
__itt_track_group* group;  
__itt_track_type ttype;    
int   extra1;              
void* extra2;              
struct ___itt_track* next;
} __itt_track;

#pragma pack(pop)



__itt_track_group* ITTAPI __itt_track_group_create(__itt_string_handle* name, __itt_track_group_type track_group_type);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, __itt_track_group*, track_group_create, (__itt_string_handle* name, __itt_track_group_type track_group_type))
#define __itt_track_group_create     ITTNOTIFY_DATA(track_group_create)
#define __itt_track_group_create_ptr ITTNOTIFY_NAME(track_group_create)
#else  
#define __itt_track_group_create(name)  (__itt_track_group*)0
#define __itt_track_group_create_ptr 0
#endif 
#else  
#define __itt_track_group_create_ptr 0
#endif 



__itt_track* ITTAPI __itt_track_create(__itt_track_group* track_group, __itt_string_handle* name, __itt_track_type track_type);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, __itt_track*, track_create, (__itt_track_group* track_group,__itt_string_handle* name, __itt_track_type track_type))
#define __itt_track_create     ITTNOTIFY_DATA(track_create)
#define __itt_track_create_ptr ITTNOTIFY_NAME(track_create)
#else  
#define __itt_track_create(track_group,name,track_type)  (__itt_track*)0
#define __itt_track_create_ptr 0
#endif 
#else  
#define __itt_track_create_ptr 0
#endif 



void ITTAPI __itt_set_track(__itt_track* track);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, set_track, (__itt_track *track))
#define __itt_set_track     ITTNOTIFY_VOID(set_track)
#define __itt_set_track_ptr ITTNOTIFY_NAME(set_track)
#else  
#define __itt_set_track(track)
#define __itt_set_track_ptr 0
#endif 
#else  
#define __itt_set_track_ptr 0
#endif 






typedef int __itt_event;


#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_event LIBITTAPI __itt_event_createA(const char    *name, int namelen);
__itt_event LIBITTAPI __itt_event_createW(const wchar_t *name, int namelen);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_event_create     __itt_event_createW
#  define __itt_event_create_ptr __itt_event_createW_ptr
#else
#  define __itt_event_create     __itt_event_createA
#  define __itt_event_create_ptr __itt_event_createA_ptr
#endif 
#else  
__itt_event LIBITTAPI __itt_event_create(const char *name, int namelen);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(LIBITTAPI, __itt_event, event_createA, (const char    *name, int namelen))
ITT_STUB(LIBITTAPI, __itt_event, event_createW, (const wchar_t *name, int namelen))
#else  
ITT_STUB(LIBITTAPI, __itt_event, event_create,  (const char    *name, int namelen))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_event_createA     ITTNOTIFY_DATA(event_createA)
#define __itt_event_createA_ptr ITTNOTIFY_NAME(event_createA)
#define __itt_event_createW     ITTNOTIFY_DATA(event_createW)
#define __itt_event_createW_ptr ITTNOTIFY_NAME(event_createW)
#else 
#define __itt_event_create      ITTNOTIFY_DATA(event_create)
#define __itt_event_create_ptr  ITTNOTIFY_NAME(event_create)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_event_createA(name, namelen) (__itt_event)0
#define __itt_event_createA_ptr 0
#define __itt_event_createW(name, namelen) (__itt_event)0
#define __itt_event_createW_ptr 0
#else 
#define __itt_event_create(name, namelen)  (__itt_event)0
#define __itt_event_create_ptr  0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_event_createA_ptr 0
#define __itt_event_createW_ptr 0
#else 
#define __itt_event_create_ptr  0
#endif 
#endif 



int LIBITTAPI __itt_event_start(__itt_event event);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(LIBITTAPI, int, event_start, (__itt_event event))
#define __itt_event_start     ITTNOTIFY_DATA(event_start)
#define __itt_event_start_ptr ITTNOTIFY_NAME(event_start)
#else  
#define __itt_event_start(event) (int)0
#define __itt_event_start_ptr 0
#endif 
#else  
#define __itt_event_start_ptr 0
#endif 



int LIBITTAPI __itt_event_end(__itt_event event);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(LIBITTAPI, int, event_end, (__itt_event event))
#define __itt_event_end     ITTNOTIFY_DATA(event_end)
#define __itt_event_end_ptr ITTNOTIFY_NAME(event_end)
#else  
#define __itt_event_end(event) (int)0
#define __itt_event_end_ptr 0
#endif 
#else  
#define __itt_event_end_ptr 0
#endif 







typedef enum
{
__itt_e_first = 0,
__itt_e_char = 0,  
__itt_e_uchar,     
__itt_e_int16,     
__itt_e_uint16,    
__itt_e_int32,     
__itt_e_uint32,    
__itt_e_int64,     
__itt_e_uint64,    
__itt_e_float,     
__itt_e_double,    
__itt_e_last = __itt_e_double
} __itt_av_data_type;



#if ITT_PLATFORM==ITT_PLATFORM_WIN
int ITTAPI __itt_av_saveA(void *data, int rank, const int *dimensions, int type, const char *filePath, int columnOrder);
int ITTAPI __itt_av_saveW(void *data, int rank, const int *dimensions, int type, const wchar_t *filePath, int columnOrder);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_av_save     __itt_av_saveW
#  define __itt_av_save_ptr __itt_av_saveW_ptr
#else 
#  define __itt_av_save     __itt_av_saveA
#  define __itt_av_save_ptr __itt_av_saveA_ptr
#endif 
#else  
int ITTAPI __itt_av_save(void *data, int rank, const int *dimensions, int type, const char *filePath, int columnOrder);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, int, av_saveA, (void *data, int rank, const int *dimensions, int type, const char *filePath, int columnOrder))
ITT_STUB(ITTAPI, int, av_saveW, (void *data, int rank, const int *dimensions, int type, const wchar_t *filePath, int columnOrder))
#else  
ITT_STUB(ITTAPI, int, av_save,  (void *data, int rank, const int *dimensions, int type, const char *filePath, int columnOrder))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_av_saveA     ITTNOTIFY_DATA(av_saveA)
#define __itt_av_saveA_ptr ITTNOTIFY_NAME(av_saveA)
#define __itt_av_saveW     ITTNOTIFY_DATA(av_saveW)
#define __itt_av_saveW_ptr ITTNOTIFY_NAME(av_saveW)
#else 
#define __itt_av_save     ITTNOTIFY_DATA(av_save)
#define __itt_av_save_ptr ITTNOTIFY_NAME(av_save)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_av_saveA(name)
#define __itt_av_saveA_ptr 0
#define __itt_av_saveW(name)
#define __itt_av_saveW_ptr 0
#else 
#define __itt_av_save(name)
#define __itt_av_save_ptr 0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_av_saveA_ptr 0
#define __itt_av_saveW_ptr 0
#else 
#define __itt_av_save_ptr 0
#endif 
#endif 


void ITTAPI __itt_enable_attach(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, enable_attach, (void))
#define __itt_enable_attach     ITTNOTIFY_VOID(enable_attach)
#define __itt_enable_attach_ptr ITTNOTIFY_NAME(enable_attach)
#else  
#define __itt_enable_attach()
#define __itt_enable_attach_ptr 0
#endif 
#else  
#define __itt_enable_attach_ptr 0
#endif 









#ifdef __cplusplus
}
#endif 

#endif 

#ifdef INTEL_ITTNOTIFY_API_PRIVATE

#ifndef _ITTNOTIFY_PRIVATE_
#define _ITTNOTIFY_PRIVATE_

#ifdef __cplusplus
extern "C" {
#endif 


void ITTAPI __itt_task_begin_overlapped_ex(const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id taskid, __itt_id parentid, __itt_string_handle* name);


void ITTAPI __itt_task_end_overlapped_ex(const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id taskid);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, task_begin_overlapped_ex,       (const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id taskid, __itt_id parentid, __itt_string_handle* name))
ITT_STUBV(ITTAPI, void, task_end_overlapped_ex,         (const __itt_domain* domain, __itt_clock_domain* clock_domain, unsigned long long timestamp, __itt_id taskid))
#define __itt_task_begin_overlapped_ex(d,x,y,z,a,b)     ITTNOTIFY_VOID_D5(task_begin_overlapped_ex,d,x,y,z,a,b)
#define __itt_task_begin_overlapped_ex_ptr              ITTNOTIFY_NAME(task_begin_overlapped_ex)
#define __itt_task_end_overlapped_ex(d,x,y,z)           ITTNOTIFY_VOID_D3(task_end_overlapped_ex,d,x,y,z)
#define __itt_task_end_overlapped_ex_ptr                ITTNOTIFY_NAME(task_end_overlapped_ex)
#else  
#define __itt_task_begin_overlapped_ex(domain,clock_domain,timestamp,taskid,parentid,name)
#define __itt_task_begin_overlapped_ex_ptr      0
#define __itt_task_end_overlapped_ex(domain,clock_domain,timestamp,taskid)
#define __itt_task_end_overlapped_ex_ptr        0
#endif 
#else  
#define __itt_task_begin_overlapped_ex_ptr      0
#define __itt_task_end_overlapped_ptr           0
#define __itt_task_end_overlapped_ex_ptr        0
#endif 




typedef int __itt_mark_type;


#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_mark_type ITTAPI __itt_mark_createA(const char    *name);
__itt_mark_type ITTAPI __itt_mark_createW(const wchar_t *name);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_mark_create     __itt_mark_createW
#  define __itt_mark_create_ptr __itt_mark_createW_ptr
#else 
#  define __itt_mark_create     __itt_mark_createA
#  define __itt_mark_create_ptr __itt_mark_createA_ptr
#endif 
#else 
__itt_mark_type ITTAPI __itt_mark_create(const char *name);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_mark_type, mark_createA, (const char    *name))
ITT_STUB(ITTAPI, __itt_mark_type, mark_createW, (const wchar_t *name))
#else  
ITT_STUB(ITTAPI, __itt_mark_type, mark_create,  (const char *name))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_createA     ITTNOTIFY_DATA(mark_createA)
#define __itt_mark_createA_ptr ITTNOTIFY_NAME(mark_createA)
#define __itt_mark_createW     ITTNOTIFY_DATA(mark_createW)
#define __itt_mark_createW_ptr ITTNOTIFY_NAME(mark_createW)
#else 
#define __itt_mark_create      ITTNOTIFY_DATA(mark_create)
#define __itt_mark_create_ptr  ITTNOTIFY_NAME(mark_create)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_createA(name) (__itt_mark_type)0
#define __itt_mark_createA_ptr 0
#define __itt_mark_createW(name) (__itt_mark_type)0
#define __itt_mark_createW_ptr 0
#else 
#define __itt_mark_create(name)  (__itt_mark_type)0
#define __itt_mark_create_ptr  0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_createA_ptr 0
#define __itt_mark_createW_ptr 0
#else 
#define __itt_mark_create_ptr  0
#endif 
#endif 



#if ITT_PLATFORM==ITT_PLATFORM_WIN
int ITTAPI __itt_markA(__itt_mark_type mt, const char    *parameter);
int ITTAPI __itt_markW(__itt_mark_type mt, const wchar_t *parameter);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_mark     __itt_markW
#  define __itt_mark_ptr __itt_markW_ptr
#else 
#  define __itt_mark     __itt_markA
#  define __itt_mark_ptr __itt_markA_ptr
#endif 
#else 
int ITTAPI __itt_mark(__itt_mark_type mt, const char *parameter);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, int, markA, (__itt_mark_type mt, const char    *parameter))
ITT_STUB(ITTAPI, int, markW, (__itt_mark_type mt, const wchar_t *parameter))
#else  
ITT_STUB(ITTAPI, int, mark,  (__itt_mark_type mt, const char *parameter))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_markA     ITTNOTIFY_DATA(markA)
#define __itt_markA_ptr ITTNOTIFY_NAME(markA)
#define __itt_markW     ITTNOTIFY_DATA(markW)
#define __itt_markW_ptr ITTNOTIFY_NAME(markW)
#else 
#define __itt_mark      ITTNOTIFY_DATA(mark)
#define __itt_mark_ptr  ITTNOTIFY_NAME(mark)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_markA(mt, parameter) (int)0
#define __itt_markA_ptr 0
#define __itt_markW(mt, parameter) (int)0
#define __itt_markW_ptr 0
#else 
#define __itt_mark(mt, parameter)  (int)0
#define __itt_mark_ptr  0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_markA_ptr 0
#define __itt_markW_ptr 0
#else 
#define __itt_mark_ptr  0
#endif 
#endif 



#if ITT_PLATFORM==ITT_PLATFORM_WIN
int ITTAPI __itt_mark_globalA(__itt_mark_type mt, const char    *parameter);
int ITTAPI __itt_mark_globalW(__itt_mark_type mt, const wchar_t *parameter);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_mark_global     __itt_mark_globalW
#  define __itt_mark_global_ptr __itt_mark_globalW_ptr
#else 
#  define __itt_mark_global     __itt_mark_globalA
#  define __itt_mark_global_ptr __itt_mark_globalA_ptr
#endif 
#else 
int ITTAPI __itt_mark_global(__itt_mark_type mt, const char *parameter);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, int, mark_globalA, (__itt_mark_type mt, const char    *parameter))
ITT_STUB(ITTAPI, int, mark_globalW, (__itt_mark_type mt, const wchar_t *parameter))
#else  
ITT_STUB(ITTAPI, int, mark_global,  (__itt_mark_type mt, const char *parameter))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_globalA     ITTNOTIFY_DATA(mark_globalA)
#define __itt_mark_globalA_ptr ITTNOTIFY_NAME(mark_globalA)
#define __itt_mark_globalW     ITTNOTIFY_DATA(mark_globalW)
#define __itt_mark_globalW_ptr ITTNOTIFY_NAME(mark_globalW)
#else 
#define __itt_mark_global      ITTNOTIFY_DATA(mark_global)
#define __itt_mark_global_ptr  ITTNOTIFY_NAME(mark_global)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_globalA(mt, parameter) (int)0
#define __itt_mark_globalA_ptr 0
#define __itt_mark_globalW(mt, parameter) (int)0
#define __itt_mark_globalW_ptr 0
#else 
#define __itt_mark_global(mt, parameter)  (int)0
#define __itt_mark_global_ptr  0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_globalA_ptr 0
#define __itt_mark_globalW_ptr 0
#else 
#define __itt_mark_global_ptr  0
#endif 
#endif 



int ITTAPI __itt_mark_off(__itt_mark_type mt);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, int, mark_off, (__itt_mark_type mt))
#define __itt_mark_off     ITTNOTIFY_DATA(mark_off)
#define __itt_mark_off_ptr ITTNOTIFY_NAME(mark_off)
#else  
#define __itt_mark_off(mt) (int)0
#define __itt_mark_off_ptr 0
#endif 
#else  
#define __itt_mark_off_ptr 0
#endif 



int ITTAPI __itt_mark_global_off(__itt_mark_type mt);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, int, mark_global_off, (__itt_mark_type mt))
#define __itt_mark_global_off     ITTNOTIFY_DATA(mark_global_off)
#define __itt_mark_global_off_ptr ITTNOTIFY_NAME(mark_global_off)
#else  
#define __itt_mark_global_off(mt) (int)0
#define __itt_mark_global_off_ptr 0
#endif 
#else  
#define __itt_mark_global_off_ptr 0
#endif 





typedef struct ___itt_counter *__itt_counter;


#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_counter ITTAPI __itt_counter_createA(const char    *name, const char    *domain);
__itt_counter ITTAPI __itt_counter_createW(const wchar_t *name, const wchar_t *domain);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_counter_create     __itt_counter_createW
#  define __itt_counter_create_ptr __itt_counter_createW_ptr
#else 
#  define __itt_counter_create     __itt_counter_createA
#  define __itt_counter_create_ptr __itt_counter_createA_ptr
#endif 
#else 
__itt_counter ITTAPI __itt_counter_create(const char *name, const char *domain);
#endif 


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_counter, counter_createA, (const char    *name, const char    *domain))
ITT_STUB(ITTAPI, __itt_counter, counter_createW, (const wchar_t *name, const wchar_t *domain))
#else  
ITT_STUB(ITTAPI, __itt_counter, counter_create,  (const char *name, const char *domain))
#endif 
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_counter_createA     ITTNOTIFY_DATA(counter_createA)
#define __itt_counter_createA_ptr ITTNOTIFY_NAME(counter_createA)
#define __itt_counter_createW     ITTNOTIFY_DATA(counter_createW)
#define __itt_counter_createW_ptr ITTNOTIFY_NAME(counter_createW)
#else 
#define __itt_counter_create     ITTNOTIFY_DATA(counter_create)
#define __itt_counter_create_ptr ITTNOTIFY_NAME(counter_create)
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_counter_createA(name, domain)
#define __itt_counter_createA_ptr 0
#define __itt_counter_createW(name, domain)
#define __itt_counter_createW_ptr 0
#else 
#define __itt_counter_create(name, domain)
#define __itt_counter_create_ptr  0
#endif 
#endif 
#else  
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_counter_createA_ptr 0
#define __itt_counter_createW_ptr 0
#else 
#define __itt_counter_create_ptr  0
#endif 
#endif 



void ITTAPI __itt_counter_destroy(__itt_counter id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, counter_destroy, (__itt_counter id))
#define __itt_counter_destroy     ITTNOTIFY_VOID(counter_destroy)
#define __itt_counter_destroy_ptr ITTNOTIFY_NAME(counter_destroy)
#else  
#define __itt_counter_destroy(id)
#define __itt_counter_destroy_ptr 0
#endif 
#else  
#define __itt_counter_destroy_ptr 0
#endif 



void ITTAPI __itt_counter_inc(__itt_counter id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, counter_inc, (__itt_counter id))
#define __itt_counter_inc     ITTNOTIFY_VOID(counter_inc)
#define __itt_counter_inc_ptr ITTNOTIFY_NAME(counter_inc)
#else  
#define __itt_counter_inc(id)
#define __itt_counter_inc_ptr 0
#endif 
#else  
#define __itt_counter_inc_ptr 0
#endif 



void ITTAPI __itt_counter_inc_delta(__itt_counter id, unsigned long long value);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, counter_inc_delta, (__itt_counter id, unsigned long long value))
#define __itt_counter_inc_delta     ITTNOTIFY_VOID(counter_inc_delta)
#define __itt_counter_inc_delta_ptr ITTNOTIFY_NAME(counter_inc_delta)
#else  
#define __itt_counter_inc_delta(id, value)
#define __itt_counter_inc_delta_ptr 0
#endif 
#else  
#define __itt_counter_inc_delta_ptr 0
#endif 





typedef struct ___itt_caller *__itt_caller;


__itt_caller ITTAPI __itt_stack_caller_create(void);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, __itt_caller, stack_caller_create, (void))
#define __itt_stack_caller_create     ITTNOTIFY_DATA(stack_caller_create)
#define __itt_stack_caller_create_ptr ITTNOTIFY_NAME(stack_caller_create)
#else  
#define __itt_stack_caller_create() (__itt_caller)0
#define __itt_stack_caller_create_ptr 0
#endif 
#else  
#define __itt_stack_caller_create_ptr 0
#endif 



void ITTAPI __itt_stack_caller_destroy(__itt_caller id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, stack_caller_destroy, (__itt_caller id))
#define __itt_stack_caller_destroy     ITTNOTIFY_VOID(stack_caller_destroy)
#define __itt_stack_caller_destroy_ptr ITTNOTIFY_NAME(stack_caller_destroy)
#else  
#define __itt_stack_caller_destroy(id)
#define __itt_stack_caller_destroy_ptr 0
#endif 
#else  
#define __itt_stack_caller_destroy_ptr 0
#endif 



void ITTAPI __itt_stack_callee_enter(__itt_caller id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, stack_callee_enter, (__itt_caller id))
#define __itt_stack_callee_enter     ITTNOTIFY_VOID(stack_callee_enter)
#define __itt_stack_callee_enter_ptr ITTNOTIFY_NAME(stack_callee_enter)
#else  
#define __itt_stack_callee_enter(id)
#define __itt_stack_callee_enter_ptr 0
#endif 
#else  
#define __itt_stack_callee_enter_ptr 0
#endif 



void ITTAPI __itt_stack_callee_leave(__itt_caller id);


#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, stack_callee_leave, (__itt_caller id))
#define __itt_stack_callee_leave     ITTNOTIFY_VOID(stack_callee_leave)
#define __itt_stack_callee_leave_ptr ITTNOTIFY_NAME(stack_callee_leave)
#else  
#define __itt_stack_callee_leave(id)
#define __itt_stack_callee_leave_ptr 0
#endif 
#else  
#define __itt_stack_callee_leave_ptr 0
#endif 






#include <stdarg.h>


typedef enum __itt_error_code
{
__itt_error_success       = 0, 
__itt_error_no_module     = 1, 

__itt_error_no_symbol     = 2, 

__itt_error_unknown_group = 3, 

__itt_error_cant_read_env = 4, 

__itt_error_env_too_long  = 5, 

__itt_error_system        = 6  

} __itt_error_code;

typedef void (__itt_error_handler_t)(__itt_error_code code, va_list);
__itt_error_handler_t* __itt_set_error_handler(__itt_error_handler_t*);

const char* ITTAPI __itt_api_version(void);



#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#define __itt_error_handler ITT_JOIN(INTEL_ITTNOTIFY_PREFIX, error_handler)
void __itt_error_handler(__itt_error_code code, va_list args);
extern const int ITTNOTIFY_NAME(err);
#define __itt_err ITTNOTIFY_NAME(err)
ITT_STUB(ITTAPI, const char*, api_version, (void))
#define __itt_api_version     ITTNOTIFY_DATA(api_version)
#define __itt_api_version_ptr ITTNOTIFY_NAME(api_version)
#else  
#define __itt_api_version()   (const char*)0
#define __itt_api_version_ptr 0
#endif 
#else  
#define __itt_api_version_ptr 0
#endif 


#ifdef __cplusplus
}
#endif 

#endif 

#endif 
