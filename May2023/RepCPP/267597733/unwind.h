



#ifndef __CLANG_UNWIND_H
#define __CLANG_UNWIND_H

#if defined(__APPLE__) && __has_include_next(<unwind.h>)

# ifndef _GNU_SOURCE
#  define _SHOULD_UNDEFINE_GNU_SOURCE
#  define _GNU_SOURCE
# endif
# ifdef HIDE_EXPORTS
#  include_next <unwind.h>
# else
#  pragma GCC visibility push(default)
#  include_next <unwind.h>
#  pragma GCC visibility pop
# endif
# ifdef _SHOULD_UNDEFINE_GNU_SOURCE
#  undef _GNU_SOURCE
#  undef _SHOULD_UNDEFINE_GNU_SOURCE
# endif
#else

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


#ifndef HIDE_EXPORTS
#pragma GCC visibility push(default)
#endif

typedef uintptr_t _Unwind_Word;
typedef intptr_t _Unwind_Sword;
typedef uintptr_t _Unwind_Ptr;
typedef uintptr_t _Unwind_Internal_Ptr;
typedef uint64_t _Unwind_Exception_Class;

typedef intptr_t _sleb128_t;
typedef uintptr_t _uleb128_t;

struct _Unwind_Context;
#if defined(__arm__) && !(defined(__USING_SJLJ_EXCEPTIONS__) || defined(__ARM_DWARF_EH__))
struct _Unwind_Control_Block;
typedef struct _Unwind_Control_Block _Unwind_Exception; 
#else
struct _Unwind_Exception;
typedef struct _Unwind_Exception _Unwind_Exception;
#endif
typedef enum {
_URC_NO_REASON = 0,
#if defined(__arm__) && !defined(__USING_SJLJ_EXCEPTIONS__) && \
!defined(__ARM_DWARF_EH__)
_URC_OK = 0, 
#endif
_URC_FOREIGN_EXCEPTION_CAUGHT = 1,

_URC_FATAL_PHASE2_ERROR = 2,
_URC_FATAL_PHASE1_ERROR = 3,
_URC_NORMAL_STOP = 4,

_URC_END_OF_STACK = 5,
_URC_HANDLER_FOUND = 6,
_URC_INSTALL_CONTEXT = 7,
_URC_CONTINUE_UNWIND = 8,
#if defined(__arm__) && !defined(__USING_SJLJ_EXCEPTIONS__) && \
!defined(__ARM_DWARF_EH__)
_URC_FAILURE = 9 
#endif
} _Unwind_Reason_Code;

typedef enum {
_UA_SEARCH_PHASE = 1,
_UA_CLEANUP_PHASE = 2,

_UA_HANDLER_FRAME = 4,
_UA_FORCE_UNWIND = 8,
_UA_END_OF_STACK = 16 
} _Unwind_Action;

typedef void (*_Unwind_Exception_Cleanup_Fn)(_Unwind_Reason_Code,
_Unwind_Exception *);

#if defined(__arm__) && !(defined(__USING_SJLJ_EXCEPTIONS__) || defined(__ARM_DWARF_EH__))
typedef struct _Unwind_Control_Block _Unwind_Control_Block;
typedef uint32_t _Unwind_EHT_Header;

struct _Unwind_Control_Block {
uint64_t exception_class;
void (*exception_cleanup)(_Unwind_Reason_Code, _Unwind_Control_Block *);

struct {
uint32_t reserved1; 
uint32_t reserved2; 
uint32_t reserved3; 
uint32_t reserved4; 
uint32_t reserved5;
} unwinder_cache;

struct {
uint32_t sp;
uint32_t bitpattern[5];
} barrier_cache;

struct {
uint32_t bitpattern[4];
} cleanup_cache;

struct {
uint32_t fnstart;         
_Unwind_EHT_Header *ehtp; 
uint32_t additional;      
uint32_t reserved1;
} pr_cache;
long long int : 0; 
} __attribute__((__aligned__(8)));
#else
struct _Unwind_Exception {
_Unwind_Exception_Class exception_class;
_Unwind_Exception_Cleanup_Fn exception_cleanup;
#if !defined (__USING_SJLJ_EXCEPTIONS__) && defined (__SEH__)
_Unwind_Word private_[6];
#else
_Unwind_Word private_1;
_Unwind_Word private_2;
#endif

} __attribute__((__aligned__));
#endif

typedef _Unwind_Reason_Code (*_Unwind_Stop_Fn)(int, _Unwind_Action,
_Unwind_Exception_Class,
_Unwind_Exception *,
struct _Unwind_Context *,
void *);

typedef _Unwind_Reason_Code (*_Unwind_Personality_Fn)(int, _Unwind_Action,
_Unwind_Exception_Class,
_Unwind_Exception *,
struct _Unwind_Context *);
typedef _Unwind_Personality_Fn __personality_routine;

typedef _Unwind_Reason_Code (*_Unwind_Trace_Fn)(struct _Unwind_Context *,
void *);

#if defined(__arm__) && !(defined(__USING_SJLJ_EXCEPTIONS__) || defined(__ARM_DWARF_EH__))
typedef enum {
_UVRSC_CORE = 0,        
_UVRSC_VFP = 1,         
_UVRSC_WMMXD = 3,       
_UVRSC_WMMXC = 4        
} _Unwind_VRS_RegClass;

typedef enum {
_UVRSD_UINT32 = 0,
_UVRSD_VFPX = 1,
_UVRSD_UINT64 = 3,
_UVRSD_FLOAT = 4,
_UVRSD_DOUBLE = 5
} _Unwind_VRS_DataRepresentation;

typedef enum {
_UVRSR_OK = 0,
_UVRSR_NOT_IMPLEMENTED = 1,
_UVRSR_FAILED = 2
} _Unwind_VRS_Result;

typedef uint32_t _Unwind_State;
#define _US_VIRTUAL_UNWIND_FRAME  ((_Unwind_State)0)
#define _US_UNWIND_FRAME_STARTING ((_Unwind_State)1)
#define _US_UNWIND_FRAME_RESUME   ((_Unwind_State)2)
#define _US_ACTION_MASK           ((_Unwind_State)3)
#define _US_FORCE_UNWIND          ((_Unwind_State)8)

_Unwind_VRS_Result _Unwind_VRS_Get(struct _Unwind_Context *__context,
_Unwind_VRS_RegClass __regclass,
uint32_t __regno,
_Unwind_VRS_DataRepresentation __representation,
void *__valuep);

_Unwind_VRS_Result _Unwind_VRS_Set(struct _Unwind_Context *__context,
_Unwind_VRS_RegClass __regclass,
uint32_t __regno,
_Unwind_VRS_DataRepresentation __representation,
void *__valuep);

static __inline__
_Unwind_Word _Unwind_GetGR(struct _Unwind_Context *__context, int __index) {
_Unwind_Word __value;
_Unwind_VRS_Get(__context, _UVRSC_CORE, __index, _UVRSD_UINT32, &__value);
return __value;
}

static __inline__
void _Unwind_SetGR(struct _Unwind_Context *__context, int __index,
_Unwind_Word __value) {
_Unwind_VRS_Set(__context, _UVRSC_CORE, __index, _UVRSD_UINT32, &__value);
}

static __inline__
_Unwind_Word _Unwind_GetIP(struct _Unwind_Context *__context) {
_Unwind_Word __ip = _Unwind_GetGR(__context, 15);
return __ip & ~(_Unwind_Word)(0x1); 
}

static __inline__
void _Unwind_SetIP(struct _Unwind_Context *__context, _Unwind_Word __value) {
_Unwind_Word __thumb_mode_bit = _Unwind_GetGR(__context, 15) & 0x1;
_Unwind_SetGR(__context, 15, __value | __thumb_mode_bit);
}
#else
_Unwind_Word _Unwind_GetGR(struct _Unwind_Context *, int);
void _Unwind_SetGR(struct _Unwind_Context *, int, _Unwind_Word);

_Unwind_Word _Unwind_GetIP(struct _Unwind_Context *);
void _Unwind_SetIP(struct _Unwind_Context *, _Unwind_Word);
#endif


_Unwind_Word _Unwind_GetIPInfo(struct _Unwind_Context *, int *);

_Unwind_Word _Unwind_GetCFA(struct _Unwind_Context *);

_Unwind_Word _Unwind_GetBSP(struct _Unwind_Context *);

void *_Unwind_GetLanguageSpecificData(struct _Unwind_Context *);

_Unwind_Ptr _Unwind_GetRegionStart(struct _Unwind_Context *);


#if !defined(__APPLE__) || !defined(__arm__)
_Unwind_Reason_Code _Unwind_RaiseException(_Unwind_Exception *);
_Unwind_Reason_Code _Unwind_ForcedUnwind(_Unwind_Exception *, _Unwind_Stop_Fn,
void *);
void _Unwind_DeleteException(_Unwind_Exception *);
void _Unwind_Resume(_Unwind_Exception *);
_Unwind_Reason_Code _Unwind_Resume_or_Rethrow(_Unwind_Exception *);

#endif

_Unwind_Reason_Code _Unwind_Backtrace(_Unwind_Trace_Fn, void *);


typedef struct SjLj_Function_Context *_Unwind_FunctionContext_t;

void _Unwind_SjLj_Register(_Unwind_FunctionContext_t);
void _Unwind_SjLj_Unregister(_Unwind_FunctionContext_t);
_Unwind_Reason_Code _Unwind_SjLj_RaiseException(_Unwind_Exception *);
_Unwind_Reason_Code _Unwind_SjLj_ForcedUnwind(_Unwind_Exception *,
_Unwind_Stop_Fn, void *);
void _Unwind_SjLj_Resume(_Unwind_Exception *);
_Unwind_Reason_Code _Unwind_SjLj_Resume_or_Rethrow(_Unwind_Exception *);

void *_Unwind_FindEnclosingFunction(void *);

#ifdef __APPLE__

_Unwind_Ptr _Unwind_GetDataRelBase(struct _Unwind_Context *)
__attribute__((__unavailable__));
_Unwind_Ptr _Unwind_GetTextRelBase(struct _Unwind_Context *)
__attribute__((__unavailable__));


void __register_frame(const void *);
void __deregister_frame(const void *);

struct dwarf_eh_bases {
uintptr_t tbase;
uintptr_t dbase;
uintptr_t func;
};
void *_Unwind_Find_FDE(const void *, struct dwarf_eh_bases *);

void __register_frame_info_bases(const void *, void *, void *, void *)
__attribute__((__unavailable__));
void __register_frame_info(const void *, void *) __attribute__((__unavailable__));
void __register_frame_info_table_bases(const void *, void*, void *, void *)
__attribute__((__unavailable__));
void __register_frame_info_table(const void *, void *)
__attribute__((__unavailable__));
void __register_frame_table(const void *) __attribute__((__unavailable__));
void __deregister_frame_info(const void *) __attribute__((__unavailable__));
void __deregister_frame_info_bases(const void *)__attribute__((__unavailable__));

#else

_Unwind_Ptr _Unwind_GetDataRelBase(struct _Unwind_Context *);
_Unwind_Ptr _Unwind_GetTextRelBase(struct _Unwind_Context *);

#endif


#ifndef HIDE_EXPORTS
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif

#endif 
