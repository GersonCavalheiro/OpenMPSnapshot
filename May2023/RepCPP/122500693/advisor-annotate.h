



#ifndef _ADVISOR_ANNOTATE_H_
#define _ADVISOR_ANNOTATE_H_


#define INTEL_ADVISOR_ANNOTATION_VERSION 1.0

#ifdef ANNOTATE_EXPAND_NULL

#define ANNOTATE_SITE_BEGIN(_SITE)
#define ANNOTATE_SITE_END(...)
#define ANNOTATE_TASK_BEGIN(_TASK)
#define ANNOTATE_TASK_END(...)
#define ANNOTATE_ITERATION_TASK(_TASK)
#define ANNOTATE_LOCK_ACQUIRE(_ADDR)
#define ANNOTATE_LOCK_RELEASE(_ADDR)
#define ANNOTATE_RECORD_ALLOCATION(_ADDR, _SIZE)
#define ANNOTATE_RECORD_DEALLOCATION(_ADDR)
#define ANNOTATE_INDUCTION_USES(_ADDR, _SIZE)
#define ANNOTATE_REDUCTION_USES(_ADDR, _SIZE)
#define ANNOTATE_OBSERVE_USES(_ADDR, _SIZE)
#define ANNOTATE_CLEAR_USES(_ADDR)
#define ANNOTATE_DISABLE_OBSERVATION_PUSH
#define ANNOTATE_DISABLE_OBSERVATION_POP
#define ANNOTATE_DISABLE_COLLECTION_PUSH
#define ANNOTATE_DISABLE_COLLECTION_POP
#define ANNOTATE_AGGREGATE_TASK(_COUNT)

#else 

#if defined(WIN32) || defined(_WIN32)

#define ANNOTATEAPI __cdecl

#ifndef ANNOTATE_DECLARE
#include <windows.h>

typedef HMODULE lib_t;

#define __itt_get_proc(lib, name) GetProcAddress(lib, name)
#define __itt_load_lib(name)      LoadLibraryA(name)
#define __itt_unload_lib(handle)  FreeLibrary(handle)
#define __itt_system_error()      (int)GetLastError()
#endif 

#else 

#if defined _M_IX86 || __i386__
#   define ANNOTATEAPI __attribute__ ((cdecl))
#else
#   define ANNOTATEAPI 
#endif


#ifndef ANNOTATE_DECLARE
#include <pthread.h>
#include <dlfcn.h>
#include <errno.h>

typedef void* lib_t;

#define __itt_get_proc(lib, name) dlsym(lib, name)
#define __itt_load_lib(name)      dlopen(name, RTLD_LAZY)
#define __itt_unload_lib(handle)  dlclose(handle)
#define __itt_system_error()      errno
#endif 

#endif 

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif 

#ifndef _ITTNOTIFY_H_ 


typedef void* __itt_model_site;             
typedef void* __itt_model_site_instance;    
typedef void* __itt_model_task;             
typedef void* __itt_model_task_instance;    

typedef enum {
__itt_model_disable_observation,
__itt_model_disable_collection
} __itt_model_disable;

#endif 






#ifndef _ITTNOTIFY_H_ 
#define ITT_NOTIFY_DECL(_text) _text
#else
#define ITT_NOTIFY_DECL(_text)
#endif


#if defined(__cplusplus) && defined(WIN32)
#define _ANNOTATE_ROUTINES_ADDR __annotate_routines_s
#else
#define _ANNOTATE_ROUTINES_ADDR __annotate_routines_init( __annotate_routines() )
#endif 


#define _ANNOTATE_DECLARE_0(_BASENAME) \
typedef void (ANNOTATEAPI * __annotate_##_BASENAME##_t)(); \
static __inline void ANNOTATEAPI __annotate_##_BASENAME##_t_nop() { }; \
ITT_NOTIFY_DECL( extern void ANNOTATEAPI __itt_model_##_BASENAME(); )

#define _ANNOTATE_DECLARE_0_INT(_BASENAME) \
typedef int (ANNOTATEAPI * __annotate_##_BASENAME##_t)(); \
static __inline int ANNOTATEAPI __annotate_##_BASENAME##_t_nop() { return 0; }; \
ITT_NOTIFY_DECL( extern void ANNOTATEAPI __itt_model_##_BASENAME(); )

#define _ANNOTATE_CALL_0(_BASENAME) { _ANNOTATE_ROUTINES_ADDR->_BASENAME(); }

#define _ANNOTATE_DECLARE_1(_BASENAME, _P1TYPE) \
typedef void (ANNOTATEAPI * __annotate_##_BASENAME##_t)(_P1TYPE p1); \
static __inline void ANNOTATEAPI __annotate_##_BASENAME##_t_nop(_P1TYPE p1) { (void)p1; }; \
ITT_NOTIFY_DECL( extern void ANNOTATEAPI __itt_model_##_BASENAME(_P1TYPE p1); )

#define _ANNOTATE_CALL_1(_BASENAME, _P1) { _ANNOTATE_ROUTINES_ADDR->_BASENAME(_P1); }

#define _ANNOTATE_DECLARE_2(_BASENAME, _P1TYPE, _P2TYPE) \
typedef void (ANNOTATEAPI * __annotate_##_BASENAME##_t)(_P1TYPE p1, _P2TYPE p2); \
static __inline void ANNOTATEAPI __annotate_##_BASENAME##_t_nop(_P1TYPE p1, _P2TYPE p2) { (void)p1; (void)p2; }; \
ITT_NOTIFY_DECL( extern void ANNOTATEAPI __itt_model_##_BASENAME(_P1TYPE p1, _P2TYPE p2); )

#define _ANNOTATE_CALL_2(_BASENAME, _P1, _P2) { _ANNOTATE_ROUTINES_ADDR->_BASENAME((_P1), (_P2)); }




_ANNOTATE_DECLARE_1(site_beginA, const char *)
_ANNOTATE_DECLARE_0(site_end_2)
_ANNOTATE_DECLARE_1(task_beginA, const char *)
_ANNOTATE_DECLARE_0(task_end_2)
_ANNOTATE_DECLARE_1(iteration_taskA, const char *)
_ANNOTATE_DECLARE_1(lock_acquire_2, void *)
_ANNOTATE_DECLARE_1(lock_release_2, void *)
_ANNOTATE_DECLARE_2(record_allocation, void *, size_t)
_ANNOTATE_DECLARE_1(record_deallocation, void *)
_ANNOTATE_DECLARE_2(induction_uses, void *, size_t)
_ANNOTATE_DECLARE_2(reduction_uses, void *, size_t)
_ANNOTATE_DECLARE_2(observe_uses, void *, size_t)
_ANNOTATE_DECLARE_1(clear_uses, void *)
_ANNOTATE_DECLARE_1(disable_push, __itt_model_disable)
_ANNOTATE_DECLARE_0(disable_pop)
_ANNOTATE_DECLARE_1(aggregate_task, size_t)
_ANNOTATE_DECLARE_0_INT(is_collection_disabled)


struct __annotate_routines {
volatile int                        initialized;
__annotate_site_beginA_t            site_beginA;
__annotate_site_end_2_t             site_end_2;
__annotate_task_beginA_t            task_beginA;
__annotate_task_end_2_t             task_end_2;
__annotate_iteration_taskA_t        iteration_taskA;
__annotate_lock_acquire_2_t         lock_acquire_2;
__annotate_lock_release_2_t         lock_release_2;
__annotate_record_allocation_t      record_allocation;
__annotate_record_deallocation_t    record_deallocation;
__annotate_induction_uses_t         induction_uses;
__annotate_reduction_uses_t         reduction_uses;
__annotate_observe_uses_t           observe_uses;
__annotate_clear_uses_t             clear_uses;
__annotate_disable_push_t           disable_push;
__annotate_disable_pop_t            disable_pop;
__annotate_aggregate_task_t         aggregate_task;
__annotate_is_collection_disabled_t is_collection_disabled;
};


static __inline struct __annotate_routines* __annotate_routines()
{
static struct __annotate_routines __annotate_routines;
return &__annotate_routines;
}



#ifdef ANNOTATE_DECLARE
extern struct __annotate_routines* ANNOTATEAPI __annotate_routines_init(struct __annotate_routines* itt);
#else
#ifdef ANNOTATE_DEFINE

#else
static __inline 
#endif
struct __annotate_routines*
ANNOTATEAPI
__annotate_routines_init(struct __annotate_routines* itt) {

if (itt->initialized) {
return itt;
} else {


int do_disable_pop = 0;
#if !(defined(WIN32) || defined(_WIN32))
char* lib_name = NULL;
#endif

lib_t itt_notify = 0;

#if defined(WIN32) || defined(_WIN32)
itt_notify = __itt_load_lib("libittnotify.dll");
#else
if (sizeof(void*) > 4) {
lib_name = getenv("INTEL_LIBITTNOTIFY64");
} else {
lib_name = getenv("INTEL_LIBITTNOTIFY32");
}
if (lib_name) {
itt_notify = __itt_load_lib(lib_name);
}
#endif
if (itt_notify != NULL) {


__annotate_disable_push_t disable_push;
__annotate_is_collection_disabled_t is_collection_disabled;
disable_push            = (__annotate_disable_push_t)       __itt_get_proc(itt_notify, "__itt_model_disable_push");
is_collection_disabled  = (__annotate_is_collection_disabled_t) __itt_get_proc(itt_notify, "__itt_model_is_collection_disabled");
if (disable_push) {
if ( ! (is_collection_disabled && is_collection_disabled()) )
{
disable_push(__itt_model_disable_observation);
do_disable_pop = 1;
}
}
itt->site_beginA         = (__annotate_site_beginA_t)        __itt_get_proc(itt_notify, "__itt_model_site_beginA");
itt->site_end_2          = (__annotate_site_end_2_t)         __itt_get_proc(itt_notify, "__itt_model_site_end_2");
itt->task_beginA         = (__annotate_task_beginA_t)        __itt_get_proc(itt_notify, "__itt_model_task_beginA");
itt->task_end_2          = (__annotate_task_end_2_t)         __itt_get_proc(itt_notify, "__itt_model_task_end_2");
itt->iteration_taskA     = (__annotate_iteration_taskA_t)    __itt_get_proc(itt_notify, "__itt_model_iteration_taskA");
itt->lock_acquire_2      = (__annotate_lock_acquire_2_t)     __itt_get_proc(itt_notify, "__itt_model_lock_acquire_2");
itt->lock_release_2      = (__annotate_lock_release_2_t)     __itt_get_proc(itt_notify, "__itt_model_lock_release_2");
itt->record_allocation   = (__annotate_record_allocation_t)  __itt_get_proc(itt_notify, "__itt_model_record_allocation");
itt->record_deallocation = (__annotate_record_deallocation_t)__itt_get_proc(itt_notify, "__itt_model_record_deallocation");
itt->induction_uses      = (__annotate_induction_uses_t)     __itt_get_proc(itt_notify, "__itt_model_induction_uses");
itt->reduction_uses      = (__annotate_reduction_uses_t)     __itt_get_proc(itt_notify, "__itt_model_reduction_uses");
itt->observe_uses        = (__annotate_observe_uses_t)       __itt_get_proc(itt_notify, "__itt_model_observe_uses");
itt->clear_uses          = (__annotate_clear_uses_t)         __itt_get_proc(itt_notify, "__itt_model_clear_uses");
itt->disable_push        = disable_push;
itt->disable_pop         = (__annotate_disable_pop_t)        __itt_get_proc(itt_notify, "__itt_model_disable_pop");
itt->aggregate_task      = (__annotate_aggregate_task_t)     __itt_get_proc(itt_notify, "__itt_model_aggregate_task");
itt->is_collection_disabled = is_collection_disabled;
}

if (!itt->site_beginA)         itt->site_beginA =       __annotate_site_beginA_t_nop;
if (!itt->site_end_2)          itt->site_end_2 =        __annotate_site_end_2_t_nop;
if (!itt->task_beginA)         itt->task_beginA =       __annotate_task_beginA_t_nop;
if (!itt->task_end_2)          itt->task_end_2 =        __annotate_task_end_2_t_nop;
if (!itt->iteration_taskA)     itt->iteration_taskA =   __annotate_iteration_taskA_t_nop;
if (!itt->lock_acquire_2)      itt->lock_acquire_2 =    __annotate_lock_acquire_2_t_nop;
if (!itt->lock_release_2)      itt->lock_release_2 =    __annotate_lock_release_2_t_nop;
if (!itt->record_allocation)   itt->record_allocation = __annotate_record_allocation_t_nop;
if (!itt->record_deallocation) itt->record_deallocation=__annotate_record_deallocation_t_nop;
if (!itt->induction_uses)      itt->induction_uses =    __annotate_induction_uses_t_nop;
if (!itt->reduction_uses)      itt->reduction_uses =    __annotate_reduction_uses_t_nop;
if (!itt->observe_uses)        itt->observe_uses =      __annotate_observe_uses_t_nop;
if (!itt->clear_uses)          itt->clear_uses =        __annotate_clear_uses_t_nop;
if (!itt->disable_push)        itt->disable_push =      __annotate_disable_push_t_nop;
if (!itt->disable_pop)         itt->disable_pop =       __annotate_disable_pop_t_nop;
if (!itt->aggregate_task)      itt->aggregate_task =    __annotate_aggregate_task_t_nop;
if (!itt->is_collection_disabled) itt->is_collection_disabled = __annotate_is_collection_disabled_t_nop;

itt->initialized = 1;

if (do_disable_pop) {
itt->disable_pop();
}
}
return itt;
}
#endif 



#if defined(__cplusplus) && defined(WIN32)

static struct __annotate_routines* __annotate_routines_s = __annotate_routines_init( __annotate_routines() );
#endif


#if defined(__cplusplus)

#if defined(WIN32) && defined(__CLR_VER)
#pragma managed(push, on)
#define ANNOTATE_CLR_NOINLINE __declspec(noinline)
#else
#define ANNOTATE_CLR_NOINLINE
#endif
class Annotate {
public:
static ANNOTATE_CLR_NOINLINE void SiteBegin(const char* site)      { _ANNOTATE_ROUTINES_ADDR->site_beginA(site); }
static ANNOTATE_CLR_NOINLINE void SiteEnd()                        { _ANNOTATE_ROUTINES_ADDR->site_end_2(); }
static ANNOTATE_CLR_NOINLINE void TaskBegin(const char* task)      { _ANNOTATE_ROUTINES_ADDR->task_beginA(task); }
static ANNOTATE_CLR_NOINLINE void TaskEnd()                        { _ANNOTATE_ROUTINES_ADDR->task_end_2(); }
static ANNOTATE_CLR_NOINLINE void IterationTask(const char* task)  { _ANNOTATE_ROUTINES_ADDR->iteration_taskA(task); }
static ANNOTATE_CLR_NOINLINE void LockAcquire(void* lockId)        { _ANNOTATE_ROUTINES_ADDR->lock_acquire_2(lockId); }
static ANNOTATE_CLR_NOINLINE void LockRelease(void* lockId)        { _ANNOTATE_ROUTINES_ADDR->lock_release_2(lockId); }
static ANNOTATE_CLR_NOINLINE void RecordAllocation(void *p, size_t s) { _ANNOTATE_ROUTINES_ADDR->record_allocation(p, s); }
static ANNOTATE_CLR_NOINLINE void RecordDeallocation(void *p)      { _ANNOTATE_ROUTINES_ADDR->record_deallocation(p); }
static ANNOTATE_CLR_NOINLINE void InductionUses(void *p, size_t s) { _ANNOTATE_ROUTINES_ADDR->induction_uses(p, s); }
static ANNOTATE_CLR_NOINLINE void ReductionUses(void *p, size_t s) { _ANNOTATE_ROUTINES_ADDR->reduction_uses(p, s); }
static ANNOTATE_CLR_NOINLINE void ObserveUses(void *p, size_t s)   { _ANNOTATE_ROUTINES_ADDR->observe_uses(p, s); }
static ANNOTATE_CLR_NOINLINE void ClearUses(void *p)               { _ANNOTATE_ROUTINES_ADDR->clear_uses(p); }
static ANNOTATE_CLR_NOINLINE void DisablePush(__itt_model_disable d) { _ANNOTATE_ROUTINES_ADDR->disable_push(d); }
static ANNOTATE_CLR_NOINLINE void DisablePop()                     { _ANNOTATE_ROUTINES_ADDR->disable_pop(); }
static ANNOTATE_CLR_NOINLINE void AggregateTask(size_t c)          { _ANNOTATE_ROUTINES_ADDR->aggregate_task(c); }
};
#if defined(WIN32) && defined(__CLR_VER)
#pragma managed(pop)
#endif
#undef ANNOTATE_CLR_NOINLINE
#endif

#if defined(__cplusplus) && defined(WIN32) && defined(__CLR_VER)

#define ANNOTATE_SITE_BEGIN(_SITE) Annotate::SiteBegin(#_SITE)
#define ANNOTATE_SITE_END(...) Annotate::SiteEnd()
#define ANNOTATE_TASK_BEGIN(_TASK) Annotate::TaskBegin(#_TASK)
#define ANNOTATE_TASK_END(...) Annotate::TaskEnd()
#define ANNOTATE_ITERATION_TASK(_TASK) Annotate::IterationTask(#_TASK)
#define ANNOTATE_LOCK_ACQUIRE(_ADDR) Annotate::LockAcquire(_ADDR)
#define ANNOTATE_LOCK_RELEASE(_ADDR) Annotate::LockRelease(_ADDR)
#define ANNOTATE_RECORD_ALLOCATION(_ADDR, _SIZE) Annotate::RecordAllocation((_ADDR), (_SIZE))
#define ANNOTATE_RECORD_DEALLOCATION(_ADDR) Annotate::RecordDeallocation(_ADDR)
#define ANNOTATE_INDUCTION_USES(_ADDR, _SIZE) Annotate::InductionUses((_ADDR), (_SIZE))
#define ANNOTATE_REDUCTION_USES(_ADDR, _SIZE) Annotate::ReductionUses((_ADDR), (_SIZE))
#define ANNOTATE_OBSERVE_USES(_ADDR, _SIZE) Annotate::ObserveUses((_ADDR), (_SIZE))
#define ANNOTATE_CLEAR_USES(_ADDR) Annotate::ClearUses(_ADDR)
#define ANNOTATE_DISABLE_OBSERVATION_PUSH Annotate::DisablePush(itt_model_disable_observation)
#define ANNOTATE_DISABLE_OBSERVATION_POP Annotate::DisablePop()
#define ANNOTATE_DISABLE_COLLECTION_PUSH Annotate::DisablePush(__itt_model_disable_collection)
#define ANNOTATE_DISABLE_COLLECTION_POP Annotate::DisablePop()
#define ANNOTATE_AGGREGATE_TASK(_COUNT) Annotate::AggregateTask(_COUNT)

#else


#define ANNOTATE_SITE_BEGIN(_SITE) _ANNOTATE_CALL_1(site_beginA, #_SITE)


#define ANNOTATE_SITE_END(...) _ANNOTATE_CALL_0(site_end_2)


#define ANNOTATE_TASK_BEGIN(_TASK) _ANNOTATE_CALL_1(task_beginA, #_TASK)


#define ANNOTATE_TASK_END(...) _ANNOTATE_CALL_0(task_end_2)


#define ANNOTATE_ITERATION_TASK(_TASK) _ANNOTATE_CALL_1(iteration_taskA, #_TASK)


#define ANNOTATE_LOCK_ACQUIRE(_ADDR) _ANNOTATE_CALL_1(lock_acquire_2, (_ADDR))


#define ANNOTATE_LOCK_RELEASE(_ADDR) _ANNOTATE_CALL_1(lock_release_2, (_ADDR))


#define ANNOTATE_RECORD_ALLOCATION(_ADDR, _SIZE) _ANNOTATE_CALL_2(record_allocation, (_ADDR), (_SIZE))


#define ANNOTATE_RECORD_DEALLOCATION(_ADDR) _ANNOTATE_CALL_1(record_deallocation, (_ADDR))


#define ANNOTATE_INDUCTION_USES(_ADDR, _SIZE) _ANNOTATE_CALL_2(induction_uses, (_ADDR), (_SIZE))


#define ANNOTATE_REDUCTION_USES(_ADDR, _SIZE) _ANNOTATE_CALL_2(reduction_uses, (_ADDR), (_SIZE))


#define ANNOTATE_OBSERVE_USES(_ADDR, _SIZE) _ANNOTATE_CALL_2(observe_uses, (_ADDR), (_SIZE))


#define ANNOTATE_CLEAR_USES(_ADDR) _ANNOTATE_CALL_1(clear_uses, (_ADDR))


#define ANNOTATE_DISABLE_OBSERVATION_PUSH _ANNOTATE_CALL_1(disable_push, __itt_model_disable_observation)


#define ANNOTATE_DISABLE_OBSERVATION_POP _ANNOTATE_CALL_0(disable_pop)


#define ANNOTATE_DISABLE_COLLECTION_PUSH _ANNOTATE_CALL_1(disable_push, __itt_model_disable_collection)


#define ANNOTATE_DISABLE_COLLECTION_POP _ANNOTATE_CALL_0(disable_pop)


#define ANNOTATE_AGGREGATE_TASK(_COUNT) _ANNOTATE_CALL_1(aggregate_task, (_COUNT))

#endif

#ifdef __cplusplus
}
#endif 

#endif 

#endif 
