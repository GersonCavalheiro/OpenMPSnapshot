#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>\n
# 1 "bicg.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "bicg.c"
# 1 "utilities/polybench.h" 1
# 26 "utilities/polybench.h"
# 1 "/usr/include/stdlib.h" 1 3 4
# 61 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/Availability.h" 1 3 4
# 202 "/usr/include/Availability.h" 3 4
# 1 "/usr/include/AvailabilityInternal.h" 1 3 4
# 203 "/usr/include/Availability.h" 2 3 4
# 62 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/_types.h" 1 3 4
# 27 "/usr/include/_types.h" 3 4
# 1 "/usr/include/sys/_types.h" 1 3 4
# 32 "/usr/include/sys/_types.h" 3 4
# 1 "/usr/include/sys/cdefs.h" 1 3 4
# 587 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/sys/_symbol_aliasing.h" 1 3 4
# 588 "/usr/include/sys/cdefs.h" 2 3 4
# 653 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/sys/_posix_availability.h" 1 3 4
# 654 "/usr/include/sys/cdefs.h" 2 3 4
# 33 "/usr/include/sys/_types.h" 2 3 4
# 1 "/usr/include/machine/_types.h" 1 3 4
# 32 "/usr/include/machine/_types.h" 3 4
# 1 "/usr/include/i386/_types.h" 1 3 4
# 37 "/usr/include/i386/_types.h" 3 4
# 37 "/usr/include/i386/_types.h" 3 4
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef short __int16_t;
typedef unsigned short __uint16_t;
typedef int __int32_t;
typedef unsigned int __uint32_t;
typedef long long __int64_t;
typedef unsigned long long __uint64_t;
typedef long __darwin_intptr_t;
typedef unsigned int __darwin_natural_t;
# 70 "/usr/include/i386/_types.h" 3 4
typedef int __darwin_ct_rune_t;
typedef union {
char __mbstate8[128];
long long _mbstateL;
} __mbstate_t;
typedef __mbstate_t __darwin_mbstate_t;
typedef long int __darwin_ptrdiff_t;
typedef long unsigned int __darwin_size_t;
typedef __builtin_va_list __darwin_va_list;
typedef int __darwin_wchar_t;
typedef __darwin_wchar_t __darwin_rune_t;
typedef int __darwin_wint_t;
typedef unsigned long __darwin_clock_t;
typedef __uint32_t __darwin_socklen_t;
typedef long __darwin_ssize_t;
typedef long __darwin_time_t;
# 33 "/usr/include/machine/_types.h" 2 3 4
# 34 "/usr/include/sys/_types.h" 2 3 4
# 55 "/usr/include/sys/_types.h" 3 4
typedef __int64_t __darwin_blkcnt_t;
typedef __int32_t __darwin_blksize_t;
typedef __int32_t __darwin_dev_t;
typedef unsigned int __darwin_fsblkcnt_t;
typedef unsigned int __darwin_fsfilcnt_t;
typedef __uint32_t __darwin_gid_t;
typedef __uint32_t __darwin_id_t;
typedef __uint64_t __darwin_ino64_t;
typedef __darwin_ino64_t __darwin_ino_t;
typedef __darwin_natural_t __darwin_mach_port_name_t;
typedef __darwin_mach_port_name_t __darwin_mach_port_t;
typedef __uint16_t __darwin_mode_t;
typedef __int64_t __darwin_off_t;
typedef __int32_t __darwin_pid_t;
typedef __uint32_t __darwin_sigset_t;
typedef __int32_t __darwin_suseconds_t;
typedef __uint32_t __darwin_uid_t;
typedef __uint32_t __darwin_useconds_t;
typedef unsigned char __darwin_uuid_t[16];
typedef char __darwin_uuid_string_t[37];
# 1 "/usr/include/sys/_pthread/_pthread_types.h" 1 3 4
# 57 "/usr/include/sys/_pthread/_pthread_types.h" 3 4
struct __darwin_pthread_handler_rec {
void (*__routine)(void *);
void *__arg;
struct __darwin_pthread_handler_rec *__next;
};
struct _opaque_pthread_attr_t {
long __sig;
char __opaque[56];
};
struct _opaque_pthread_cond_t {
long __sig;
char __opaque[40];
};
struct _opaque_pthread_condattr_t {
long __sig;
char __opaque[8];
};
struct _opaque_pthread_mutex_t {
long __sig;
char __opaque[56];
};
struct _opaque_pthread_mutexattr_t {
long __sig;
char __opaque[8];
};
struct _opaque_pthread_once_t {
long __sig;
char __opaque[8];
};
struct _opaque_pthread_rwlock_t {
long __sig;
char __opaque[192];
};
struct _opaque_pthread_rwlockattr_t {
long __sig;
char __opaque[16];
};
struct _opaque_pthread_t {
long __sig;
struct __darwin_pthread_handler_rec *__cleanup_stack;
char __opaque[8176];
};
typedef struct _opaque_pthread_attr_t __darwin_pthread_attr_t;
typedef struct _opaque_pthread_cond_t __darwin_pthread_cond_t;
typedef struct _opaque_pthread_condattr_t __darwin_pthread_condattr_t;
typedef unsigned long __darwin_pthread_key_t;
typedef struct _opaque_pthread_mutex_t __darwin_pthread_mutex_t;
typedef struct _opaque_pthread_mutexattr_t __darwin_pthread_mutexattr_t;
typedef struct _opaque_pthread_once_t __darwin_pthread_once_t;
typedef struct _opaque_pthread_rwlock_t __darwin_pthread_rwlock_t;
typedef struct _opaque_pthread_rwlockattr_t __darwin_pthread_rwlockattr_t;
typedef struct _opaque_pthread_t *__darwin_pthread_t;
# 81 "/usr/include/sys/_types.h" 2 3 4
# 28 "/usr/include/_types.h" 2 3 4
# 40 "/usr/include/_types.h" 3 4
typedef int __darwin_nl_item;
typedef int __darwin_wctrans_t;
typedef __uint32_t __darwin_wctype_t;
# 64 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/sys/wait.h" 1 3 4
# 79 "/usr/include/sys/wait.h" 3 4
typedef enum {
P_ALL,
P_PID,
P_PGID
} idtype_t;
# 1 "/usr/include/sys/_types/_pid_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_pid_t.h" 3 4
typedef __darwin_pid_t pid_t;
# 90 "/usr/include/sys/wait.h" 2 3 4
# 1 "/usr/include/sys/_types/_id_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_id_t.h" 3 4
typedef __darwin_id_t id_t;
# 91 "/usr/include/sys/wait.h" 2 3 4
# 109 "/usr/include/sys/wait.h" 3 4
# 1 "/usr/include/sys/signal.h" 1 3 4
# 73 "/usr/include/sys/signal.h" 3 4
# 1 "/usr/include/sys/appleapiopts.h" 1 3 4
# 74 "/usr/include/sys/signal.h" 2 3 4
# 82 "/usr/include/sys/signal.h" 3 4
# 1 "/usr/include/machine/signal.h" 1 3 4
# 32 "/usr/include/machine/signal.h" 3 4
# 1 "/usr/include/i386/signal.h" 1 3 4
# 39 "/usr/include/i386/signal.h" 3 4
typedef int sig_atomic_t;
# 33 "/usr/include/machine/signal.h" 2 3 4
# 83 "/usr/include/sys/signal.h" 2 3 4
# 146 "/usr/include/sys/signal.h" 3 4
# 1 "/usr/include/machine/_mcontext.h" 1 3 4
# 29 "/usr/include/machine/_mcontext.h" 3 4
# 1 "/usr/include/i386/_mcontext.h" 1 3 4
# 34 "/usr/include/i386/_mcontext.h" 3 4
# 1 "/usr/include/mach/machine/_structs.h" 1 3 4
# 33 "/usr/include/mach/machine/_structs.h" 3 4
# 1 "/usr/include/mach/i386/_structs.h" 1 3 4
# 36 "/usr/include/mach/i386/_structs.h" 3 4
# 1 "/usr/include/machine/types.h" 1 3 4
# 35 "/usr/include/machine/types.h" 3 4
# 1 "/usr/include/i386/types.h" 1 3 4
# 76 "/usr/include/i386/types.h" 3 4
# 1 "/usr/include/sys/_types/_int8_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_int8_t.h" 3 4
typedef signed char int8_t;
# 77 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_int16_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_int16_t.h" 3 4
typedef short int16_t;
# 78 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_int32_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_int32_t.h" 3 4
typedef int int32_t;
# 79 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_int64_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_int64_t.h" 3 4
typedef long long int64_t;
# 80 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_u_int8_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_u_int8_t.h" 3 4
typedef unsigned char u_int8_t;
# 82 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_u_int16_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_u_int16_t.h" 3 4
typedef unsigned short u_int16_t;
# 83 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_u_int32_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_u_int32_t.h" 3 4
typedef unsigned int u_int32_t;
# 84 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_u_int64_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_u_int64_t.h" 3 4
typedef unsigned long long u_int64_t;
# 85 "/usr/include/i386/types.h" 2 3 4
typedef int64_t register_t;
# 1 "/usr/include/sys/_types/_intptr_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_intptr_t.h" 3 4
# 1 "/usr/include/machine/types.h" 1 3 4
# 31 "/usr/include/sys/_types/_intptr_t.h" 2 3 4
typedef __darwin_intptr_t intptr_t;
# 93 "/usr/include/i386/types.h" 2 3 4
# 1 "/usr/include/sys/_types/_uintptr_t.h" 1 3 4
# 30 "/usr/include/sys/_types/_uintptr_t.h" 3 4
typedef unsigned long uintptr_t;
# 94 "/usr/include/i386/types.h" 2 3 4
typedef u_int64_t user_addr_t;
typedef u_int64_t user_size_t;
typedef int64_t user_ssize_t;
typedef int64_t user_long_t;
typedef u_int64_t user_ulong_t;
typedef int64_t user_time_t;
typedef int64_t user_off_t;
typedef u_int64_t syscall_arg_t;
# 36 "/usr/include/machine/types.h" 2 3 4
# 37 "/usr/include/mach/i386/_structs.h" 2 3 4
# 46 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_i386_thread_state
{
unsigned int __eax;
unsigned int __ebx;
unsigned int __ecx;
unsigned int __edx;
unsigned int __edi;
unsigned int __esi;
unsigned int __ebp;
unsigned int __esp;
unsigned int __ss;
unsigned int __eflags;
unsigned int __eip;
unsigned int __cs;
unsigned int __ds;
unsigned int __es;
unsigned int __fs;
unsigned int __gs;
};
# 92 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_fp_control
{
unsigned short __invalid :1,
__denorm :1,
__zdiv :1,
__ovrfl :1,
__undfl :1,
__precis :1,
:2,
__pc :2,
__rc :2,
:1,
:3;
};
typedef struct __darwin_fp_control __darwin_fp_control_t;
# 150 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_fp_status
{
unsigned short __invalid :1,
__denorm :1,
__zdiv :1,
__ovrfl :1,
__undfl :1,
__precis :1,
__stkflt :1,
__errsumm :1,
__c0 :1,
__c1 :1,
__c2 :1,
__tos :3,
__c3 :1,
__busy :1;
};
typedef struct __darwin_fp_status __darwin_fp_status_t;
# 194 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_mmst_reg
{
char __mmst_reg[10];
char __mmst_rsrv[6];
};
# 213 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_xmm_reg
{
char __xmm_reg[16];
};
# 229 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_ymm_reg
{
char __ymm_reg[32];
};
# 245 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_zmm_reg
{
char __zmm_reg[64];
};
# 259 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_opmask_reg
{
char __opmask_reg[8];
};
# 281 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_i386_float_state
{
int __fpu_reserved[2];
struct __darwin_fp_control __fpu_fcw;
struct __darwin_fp_status __fpu_fsw;
__uint8_t __fpu_ftw;
__uint8_t __fpu_rsrv1;
__uint16_t __fpu_fop;
__uint32_t __fpu_ip;
__uint16_t __fpu_cs;
__uint16_t __fpu_rsrv2;
__uint32_t __fpu_dp;
__uint16_t __fpu_ds;
__uint16_t __fpu_rsrv3;
__uint32_t __fpu_mxcsr;
__uint32_t __fpu_mxcsrmask;
struct __darwin_mmst_reg __fpu_stmm0;
struct __darwin_mmst_reg __fpu_stmm1;
struct __darwin_mmst_reg __fpu_stmm2;
struct __darwin_mmst_reg __fpu_stmm3;
struct __darwin_mmst_reg __fpu_stmm4;
struct __darwin_mmst_reg __fpu_stmm5;
struct __darwin_mmst_reg __fpu_stmm6;
struct __darwin_mmst_reg __fpu_stmm7;
struct __darwin_xmm_reg __fpu_xmm0;
struct __darwin_xmm_reg __fpu_xmm1;
struct __darwin_xmm_reg __fpu_xmm2;
struct __darwin_xmm_reg __fpu_xmm3;
struct __darwin_xmm_reg __fpu_xmm4;
struct __darwin_xmm_reg __fpu_xmm5;
struct __darwin_xmm_reg __fpu_xmm6;
struct __darwin_xmm_reg __fpu_xmm7;
char __fpu_rsrv4[14*16];
int __fpu_reserved1;
};
struct __darwin_i386_avx_state
{
int __fpu_reserved[2];
struct __darwin_fp_control __fpu_fcw;
struct __darwin_fp_status __fpu_fsw;
__uint8_t __fpu_ftw;
__uint8_t __fpu_rsrv1;
__uint16_t __fpu_fop;
__uint32_t __fpu_ip;
__uint16_t __fpu_cs;
__uint16_t __fpu_rsrv2;
__uint32_t __fpu_dp;
__uint16_t __fpu_ds;
__uint16_t __fpu_rsrv3;
__uint32_t __fpu_mxcsr;
__uint32_t __fpu_mxcsrmask;
struct __darwin_mmst_reg __fpu_stmm0;
struct __darwin_mmst_reg __fpu_stmm1;
struct __darwin_mmst_reg __fpu_stmm2;
struct __darwin_mmst_reg __fpu_stmm3;
struct __darwin_mmst_reg __fpu_stmm4;
struct __darwin_mmst_reg __fpu_stmm5;
struct __darwin_mmst_reg __fpu_stmm6;
struct __darwin_mmst_reg __fpu_stmm7;
struct __darwin_xmm_reg __fpu_xmm0;
struct __darwin_xmm_reg __fpu_xmm1;
struct __darwin_xmm_reg __fpu_xmm2;
struct __darwin_xmm_reg __fpu_xmm3;
struct __darwin_xmm_reg __fpu_xmm4;
struct __darwin_xmm_reg __fpu_xmm5;
struct __darwin_xmm_reg __fpu_xmm6;
struct __darwin_xmm_reg __fpu_xmm7;
char __fpu_rsrv4[14*16];
int __fpu_reserved1;
char __avx_reserved1[64];
struct __darwin_xmm_reg __fpu_ymmh0;
struct __darwin_xmm_reg __fpu_ymmh1;
struct __darwin_xmm_reg __fpu_ymmh2;
struct __darwin_xmm_reg __fpu_ymmh3;
struct __darwin_xmm_reg __fpu_ymmh4;
struct __darwin_xmm_reg __fpu_ymmh5;
struct __darwin_xmm_reg __fpu_ymmh6;
struct __darwin_xmm_reg __fpu_ymmh7;
};
struct __darwin_i386_avx512_state
{
int __fpu_reserved[2];
struct __darwin_fp_control __fpu_fcw;
struct __darwin_fp_status __fpu_fsw;
__uint8_t __fpu_ftw;
__uint8_t __fpu_rsrv1;
__uint16_t __fpu_fop;
__uint32_t __fpu_ip;
__uint16_t __fpu_cs;
__uint16_t __fpu_rsrv2;
__uint32_t __fpu_dp;
__uint16_t __fpu_ds;
__uint16_t __fpu_rsrv3;
__uint32_t __fpu_mxcsr;
__uint32_t __fpu_mxcsrmask;
struct __darwin_mmst_reg __fpu_stmm0;
struct __darwin_mmst_reg __fpu_stmm1;
struct __darwin_mmst_reg __fpu_stmm2;
struct __darwin_mmst_reg __fpu_stmm3;
struct __darwin_mmst_reg __fpu_stmm4;
struct __darwin_mmst_reg __fpu_stmm5;
struct __darwin_mmst_reg __fpu_stmm6;
struct __darwin_mmst_reg __fpu_stmm7;
struct __darwin_xmm_reg __fpu_xmm0;
struct __darwin_xmm_reg __fpu_xmm1;
struct __darwin_xmm_reg __fpu_xmm2;
struct __darwin_xmm_reg __fpu_xmm3;
struct __darwin_xmm_reg __fpu_xmm4;
struct __darwin_xmm_reg __fpu_xmm5;
struct __darwin_xmm_reg __fpu_xmm6;
struct __darwin_xmm_reg __fpu_xmm7;
char __fpu_rsrv4[14*16];
int __fpu_reserved1;
char __avx_reserved1[64];
struct __darwin_xmm_reg __fpu_ymmh0;
struct __darwin_xmm_reg __fpu_ymmh1;
struct __darwin_xmm_reg __fpu_ymmh2;
struct __darwin_xmm_reg __fpu_ymmh3;
struct __darwin_xmm_reg __fpu_ymmh4;
struct __darwin_xmm_reg __fpu_ymmh5;
struct __darwin_xmm_reg __fpu_ymmh6;
struct __darwin_xmm_reg __fpu_ymmh7;
struct __darwin_opmask_reg __fpu_k0;
struct __darwin_opmask_reg __fpu_k1;
struct __darwin_opmask_reg __fpu_k2;
struct __darwin_opmask_reg __fpu_k3;
struct __darwin_opmask_reg __fpu_k4;
struct __darwin_opmask_reg __fpu_k5;
struct __darwin_opmask_reg __fpu_k6;
struct __darwin_opmask_reg __fpu_k7;
struct __darwin_ymm_reg __fpu_zmmh0;
struct __darwin_ymm_reg __fpu_zmmh1;
struct __darwin_ymm_reg __fpu_zmmh2;
struct __darwin_ymm_reg __fpu_zmmh3;
struct __darwin_ymm_reg __fpu_zmmh4;
struct __darwin_ymm_reg __fpu_zmmh5;
struct __darwin_ymm_reg __fpu_zmmh6;
struct __darwin_ymm_reg __fpu_zmmh7;
};
# 575 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_i386_exception_state
{
__uint16_t __trapno;
__uint16_t __cpu;
__uint32_t __err;
__uint32_t __faultvaddr;
};
# 595 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_x86_debug_state32
{
unsigned int __dr0;
unsigned int __dr1;
unsigned int __dr2;
unsigned int __dr3;
unsigned int __dr4;
unsigned int __dr5;
unsigned int __dr6;
unsigned int __dr7;
};
# 627 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_x86_thread_state64
{
__uint64_t __rax;
__uint64_t __rbx;
__uint64_t __rcx;
__uint64_t __rdx;
__uint64_t __rdi;
__uint64_t __rsi;
__uint64_t __rbp;
__uint64_t __rsp;
__uint64_t __r8;
__uint64_t __r9;
__uint64_t __r10;
__uint64_t __r11;
__uint64_t __r12;
__uint64_t __r13;
__uint64_t __r14;
__uint64_t __r15;
__uint64_t __rip;
__uint64_t __rflags;
__uint64_t __cs;
__uint64_t __fs;
__uint64_t __gs;
};
# 682 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_x86_float_state64
{
int __fpu_reserved[2];
struct __darwin_fp_control __fpu_fcw;
struct __darwin_fp_status __fpu_fsw;
__uint8_t __fpu_ftw;
__uint8_t __fpu_rsrv1;
__uint16_t __fpu_fop;
__uint32_t __fpu_ip;
__uint16_t __fpu_cs;
__uint16_t __fpu_rsrv2;
__uint32_t __fpu_dp;
__uint16_t __fpu_ds;
__uint16_t __fpu_rsrv3;
__uint32_t __fpu_mxcsr;
__uint32_t __fpu_mxcsrmask;
struct __darwin_mmst_reg __fpu_stmm0;
struct __darwin_mmst_reg __fpu_stmm1;
struct __darwin_mmst_reg __fpu_stmm2;
struct __darwin_mmst_reg __fpu_stmm3;
struct __darwin_mmst_reg __fpu_stmm4;
struct __darwin_mmst_reg __fpu_stmm5;
struct __darwin_mmst_reg __fpu_stmm6;
struct __darwin_mmst_reg __fpu_stmm7;
struct __darwin_xmm_reg __fpu_xmm0;
struct __darwin_xmm_reg __fpu_xmm1;
struct __darwin_xmm_reg __fpu_xmm2;
struct __darwin_xmm_reg __fpu_xmm3;
struct __darwin_xmm_reg __fpu_xmm4;
struct __darwin_xmm_reg __fpu_xmm5;
struct __darwin_xmm_reg __fpu_xmm6;
struct __darwin_xmm_reg __fpu_xmm7;
struct __darwin_xmm_reg __fpu_xmm8;
struct __darwin_xmm_reg __fpu_xmm9;
struct __darwin_xmm_reg __fpu_xmm10;
struct __darwin_xmm_reg __fpu_xmm11;
struct __darwin_xmm_reg __fpu_xmm12;
struct __darwin_xmm_reg __fpu_xmm13;
struct __darwin_xmm_reg __fpu_xmm14;
struct __darwin_xmm_reg __fpu_xmm15;
char __fpu_rsrv4[6*16];
int __fpu_reserved1;
};
struct __darwin_x86_avx_state64
{
int __fpu_reserved[2];
struct __darwin_fp_control __fpu_fcw;
struct __darwin_fp_status __fpu_fsw;
__uint8_t __fpu_ftw;
__uint8_t __fpu_rsrv1;
__uint16_t __fpu_fop;
__uint32_t __fpu_ip;
__uint16_t __fpu_cs;
__uint16_t __fpu_rsrv2;
__uint32_t __fpu_dp;
__uint16_t __fpu_ds;
__uint16_t __fpu_rsrv3;
__uint32_t __fpu_mxcsr;
__uint32_t __fpu_mxcsrmask;
struct __darwin_mmst_reg __fpu_stmm0;
struct __darwin_mmst_reg __fpu_stmm1;
struct __darwin_mmst_reg __fpu_stmm2;
struct __darwin_mmst_reg __fpu_stmm3;
struct __darwin_mmst_reg __fpu_stmm4;
struct __darwin_mmst_reg __fpu_stmm5;
struct __darwin_mmst_reg __fpu_stmm6;
struct __darwin_mmst_reg __fpu_stmm7;
struct __darwin_xmm_reg __fpu_xmm0;
struct __darwin_xmm_reg __fpu_xmm1;
struct __darwin_xmm_reg __fpu_xmm2;
struct __darwin_xmm_reg __fpu_xmm3;
struct __darwin_xmm_reg __fpu_xmm4;
struct __darwin_xmm_reg __fpu_xmm5;
struct __darwin_xmm_reg __fpu_xmm6;
struct __darwin_xmm_reg __fpu_xmm7;
struct __darwin_xmm_reg __fpu_xmm8;
struct __darwin_xmm_reg __fpu_xmm9;
struct __darwin_xmm_reg __fpu_xmm10;
struct __darwin_xmm_reg __fpu_xmm11;
struct __darwin_xmm_reg __fpu_xmm12;
struct __darwin_xmm_reg __fpu_xmm13;
struct __darwin_xmm_reg __fpu_xmm14;
struct __darwin_xmm_reg __fpu_xmm15;
char __fpu_rsrv4[6*16];
int __fpu_reserved1;
char __avx_reserved1[64];
struct __darwin_xmm_reg __fpu_ymmh0;
struct __darwin_xmm_reg __fpu_ymmh1;
struct __darwin_xmm_reg __fpu_ymmh2;
struct __darwin_xmm_reg __fpu_ymmh3;
struct __darwin_xmm_reg __fpu_ymmh4;
struct __darwin_xmm_reg __fpu_ymmh5;
struct __darwin_xmm_reg __fpu_ymmh6;
struct __darwin_xmm_reg __fpu_ymmh7;
struct __darwin_xmm_reg __fpu_ymmh8;
struct __darwin_xmm_reg __fpu_ymmh9;
struct __darwin_xmm_reg __fpu_ymmh10;
struct __darwin_xmm_reg __fpu_ymmh11;
struct __darwin_xmm_reg __fpu_ymmh12;
struct __darwin_xmm_reg __fpu_ymmh13;
struct __darwin_xmm_reg __fpu_ymmh14;
struct __darwin_xmm_reg __fpu_ymmh15;
};
struct __darwin_x86_avx512_state64
{
int __fpu_reserved[2];
struct __darwin_fp_control __fpu_fcw;
struct __darwin_fp_status __fpu_fsw;
__uint8_t __fpu_ftw;
__uint8_t __fpu_rsrv1;
__uint16_t __fpu_fop;
__uint32_t __fpu_ip;
__uint16_t __fpu_cs;
__uint16_t __fpu_rsrv2;
__uint32_t __fpu_dp;
__uint16_t __fpu_ds;
__uint16_t __fpu_rsrv3;
__uint32_t __fpu_mxcsr;
__uint32_t __fpu_mxcsrmask;
struct __darwin_mmst_reg __fpu_stmm0;
struct __darwin_mmst_reg __fpu_stmm1;
struct __darwin_mmst_reg __fpu_stmm2;
struct __darwin_mmst_reg __fpu_stmm3;
struct __darwin_mmst_reg __fpu_stmm4;
struct __darwin_mmst_reg __fpu_stmm5;
struct __darwin_mmst_reg __fpu_stmm6;
struct __darwin_mmst_reg __fpu_stmm7;
struct __darwin_xmm_reg __fpu_xmm0;
struct __darwin_xmm_reg __fpu_xmm1;
struct __darwin_xmm_reg __fpu_xmm2;
struct __darwin_xmm_reg __fpu_xmm3;
struct __darwin_xmm_reg __fpu_xmm4;
struct __darwin_xmm_reg __fpu_xmm5;
struct __darwin_xmm_reg __fpu_xmm6;
struct __darwin_xmm_reg __fpu_xmm7;
struct __darwin_xmm_reg __fpu_xmm8;
struct __darwin_xmm_reg __fpu_xmm9;
struct __darwin_xmm_reg __fpu_xmm10;
struct __darwin_xmm_reg __fpu_xmm11;
struct __darwin_xmm_reg __fpu_xmm12;
struct __darwin_xmm_reg __fpu_xmm13;
struct __darwin_xmm_reg __fpu_xmm14;
struct __darwin_xmm_reg __fpu_xmm15;
char __fpu_rsrv4[6*16];
int __fpu_reserved1;
char __avx_reserved1[64];
struct __darwin_xmm_reg __fpu_ymmh0;
struct __darwin_xmm_reg __fpu_ymmh1;
struct __darwin_xmm_reg __fpu_ymmh2;
struct __darwin_xmm_reg __fpu_ymmh3;
struct __darwin_xmm_reg __fpu_ymmh4;
struct __darwin_xmm_reg __fpu_ymmh5;
struct __darwin_xmm_reg __fpu_ymmh6;
struct __darwin_xmm_reg __fpu_ymmh7;
struct __darwin_xmm_reg __fpu_ymmh8;
struct __darwin_xmm_reg __fpu_ymmh9;
struct __darwin_xmm_reg __fpu_ymmh10;
struct __darwin_xmm_reg __fpu_ymmh11;
struct __darwin_xmm_reg __fpu_ymmh12;
struct __darwin_xmm_reg __fpu_ymmh13;
struct __darwin_xmm_reg __fpu_ymmh14;
struct __darwin_xmm_reg __fpu_ymmh15;
struct __darwin_opmask_reg __fpu_k0;
struct __darwin_opmask_reg __fpu_k1;
struct __darwin_opmask_reg __fpu_k2;
struct __darwin_opmask_reg __fpu_k3;
struct __darwin_opmask_reg __fpu_k4;
struct __darwin_opmask_reg __fpu_k5;
struct __darwin_opmask_reg __fpu_k6;
struct __darwin_opmask_reg __fpu_k7;
struct __darwin_ymm_reg __fpu_zmmh0;
struct __darwin_ymm_reg __fpu_zmmh1;
struct __darwin_ymm_reg __fpu_zmmh2;
struct __darwin_ymm_reg __fpu_zmmh3;
struct __darwin_ymm_reg __fpu_zmmh4;
struct __darwin_ymm_reg __fpu_zmmh5;
struct __darwin_ymm_reg __fpu_zmmh6;
struct __darwin_ymm_reg __fpu_zmmh7;
struct __darwin_ymm_reg __fpu_zmmh8;
struct __darwin_ymm_reg __fpu_zmmh9;
struct __darwin_ymm_reg __fpu_zmmh10;
struct __darwin_ymm_reg __fpu_zmmh11;
struct __darwin_ymm_reg __fpu_zmmh12;
struct __darwin_ymm_reg __fpu_zmmh13;
struct __darwin_ymm_reg __fpu_zmmh14;
struct __darwin_ymm_reg __fpu_zmmh15;
struct __darwin_zmm_reg __fpu_zmm16;
struct __darwin_zmm_reg __fpu_zmm17;
struct __darwin_zmm_reg __fpu_zmm18;
struct __darwin_zmm_reg __fpu_zmm19;
struct __darwin_zmm_reg __fpu_zmm20;
struct __darwin_zmm_reg __fpu_zmm21;
struct __darwin_zmm_reg __fpu_zmm22;
struct __darwin_zmm_reg __fpu_zmm23;
struct __darwin_zmm_reg __fpu_zmm24;
struct __darwin_zmm_reg __fpu_zmm25;
struct __darwin_zmm_reg __fpu_zmm26;
struct __darwin_zmm_reg __fpu_zmm27;
struct __darwin_zmm_reg __fpu_zmm28;
struct __darwin_zmm_reg __fpu_zmm29;
struct __darwin_zmm_reg __fpu_zmm30;
struct __darwin_zmm_reg __fpu_zmm31;
};
# 1140 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_x86_exception_state64
{
__uint16_t __trapno;
__uint16_t __cpu;
__uint32_t __err;
__uint64_t __faultvaddr;
};
# 1160 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_x86_debug_state64
{
__uint64_t __dr0;
__uint64_t __dr1;
__uint64_t __dr2;
__uint64_t __dr3;
__uint64_t __dr4;
__uint64_t __dr5;
__uint64_t __dr6;
__uint64_t __dr7;
};
# 1188 "/usr/include/mach/i386/_structs.h" 3 4
struct __darwin_x86_cpmu_state64
{
__uint64_t __ctrs[16];
};
# 34 "/usr/include/mach/machine/_structs.h" 2 3 4
# 35 "/usr/include/i386/_mcontext.h" 2 3 4
struct __darwin_mcontext32
{
struct __darwin_i386_exception_state __es;
struct __darwin_i386_thread_state __ss;
struct __darwin_i386_float_state __fs;
};
struct __darwin_mcontext_avx32
{
struct __darwin_i386_exception_state __es;
struct __darwin_i386_thread_state __ss;
struct __darwin_i386_avx_state __fs;
};
struct __darwin_mcontext_avx512_32
{
struct __darwin_i386_exception_state __es;
struct __darwin_i386_thread_state __ss;
struct __darwin_i386_avx512_state __fs;
};
# 97 "/usr/include/i386/_mcontext.h" 3 4
struct __darwin_mcontext64
{
struct __darwin_x86_exception_state64 __es;
struct __darwin_x86_thread_state64 __ss;
struct __darwin_x86_float_state64 __fs;
};
struct __darwin_mcontext_avx64
{
struct __darwin_x86_exception_state64 __es;
struct __darwin_x86_thread_state64 __ss;
struct __darwin_x86_avx_state64 __fs;
};
struct __darwin_mcontext_avx512_64
{
struct __darwin_x86_exception_state64 __es;
struct __darwin_x86_thread_state64 __ss;
struct __darwin_x86_avx512_state64 __fs;
};
# 156 "/usr/include/i386/_mcontext.h" 3 4
typedef struct __darwin_mcontext64 *mcontext_t;
# 30 "/usr/include/machine/_mcontext.h" 2 3 4
# 147 "/usr/include/sys/signal.h" 2 3 4
# 1 "/usr/include/sys/_pthread/_pthread_attr_t.h" 1 3 4
# 31 "/usr/include/sys/_pthread/_pthread_attr_t.h" 3 4
typedef __darwin_pthread_attr_t pthread_attr_t;
# 149 "/usr/include/sys/signal.h" 2 3 4
# 1 "/usr/include/sys/_types/_sigaltstack.h" 1 3 4
# 42 "/usr/include/sys/_types/_sigaltstack.h" 3 4
struct __darwin_sigaltstack
{
void *ss_sp;
__darwin_size_t ss_size;
int ss_flags;
};
typedef struct __darwin_sigaltstack stack_t;
# 151 "/usr/include/sys/signal.h" 2 3 4
# 1 "/usr/include/sys/_types/_ucontext.h" 1 3 4
# 39 "/usr/include/sys/_types/_ucontext.h" 3 4
# 1 "/usr/include/machine/_mcontext.h" 1 3 4
# 40 "/usr/include/sys/_types/_ucontext.h" 2 3 4
struct __darwin_ucontext
{
int uc_onstack;
__darwin_sigset_t uc_sigmask;
struct __darwin_sigaltstack uc_stack;
struct __darwin_ucontext *uc_link;
__darwin_size_t uc_mcsize;
struct __darwin_mcontext64 *uc_mcontext;
};
typedef struct __darwin_ucontext ucontext_t;
# 152 "/usr/include/sys/signal.h" 2 3 4
# 1 "/usr/include/sys/_types/_sigset_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_sigset_t.h" 3 4
typedef __darwin_sigset_t sigset_t;
# 155 "/usr/include/sys/signal.h" 2 3 4
# 1 "/usr/include/sys/_types/_size_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_size_t.h" 3 4
typedef __darwin_size_t size_t;
# 156 "/usr/include/sys/signal.h" 2 3 4
# 1 "/usr/include/sys/_types/_uid_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_uid_t.h" 3 4
typedef __darwin_uid_t uid_t;
# 157 "/usr/include/sys/signal.h" 2 3 4
union sigval {
int sival_int;
void *sival_ptr;
};
struct sigevent {
int sigev_notify;
int sigev_signo;
union sigval sigev_value;
void (*sigev_notify_function)(union sigval);
pthread_attr_t *sigev_notify_attributes;
};
typedef struct __siginfo {
int si_signo;
int si_errno;
int si_code;
pid_t si_pid;
uid_t si_uid;
int si_status;
void *si_addr;
union sigval si_value;
long si_band;
unsigned long __pad[7];
} siginfo_t;
# 269 "/usr/include/sys/signal.h" 3 4
union __sigaction_u {
void (*__sa_handler)(int);
void (*__sa_sigaction)(int, struct __siginfo *,
void *);
};
struct __sigaction {
union __sigaction_u __sigaction_u;
void (*sa_tramp)(void *, int, int, siginfo_t *, void *);
sigset_t sa_mask;
int sa_flags;
};
struct sigaction {
union __sigaction_u __sigaction_u;
sigset_t sa_mask;
int sa_flags;
};
# 331 "/usr/include/sys/signal.h" 3 4
typedef void (*sig_t)(int);
# 348 "/usr/include/sys/signal.h" 3 4
struct sigvec {
void (*sv_handler)(int);
int sv_mask;
int sv_flags;
};
# 367 "/usr/include/sys/signal.h" 3 4
struct sigstack {
char *ss_sp;
int ss_onstack;
};
# 389 "/usr/include/sys/signal.h" 3 4
void (*signal(int, void (*)(int)))(int);
# 110 "/usr/include/sys/wait.h" 2 3 4
# 1 "/usr/include/sys/resource.h" 1 3 4
# 72 "/usr/include/sys/resource.h" 3 4
# 1 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include/stdint.h" 1 3 4
# 9 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include/stdint.h" 3 4
# 1 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 1 3 4
# 32 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 3 4
# 1 "/usr/include/_types/_uint8_t.h" 1 3 4
# 31 "/usr/include/_types/_uint8_t.h" 3 4
typedef unsigned char uint8_t;
# 33 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 2 3 4
# 1 "/usr/include/_types/_uint16_t.h" 1 3 4
# 31 "/usr/include/_types/_uint16_t.h" 3 4
typedef unsigned short uint16_t;
# 34 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 2 3 4
# 1 "/usr/include/_types/_uint32_t.h" 1 3 4
# 31 "/usr/include/_types/_uint32_t.h" 3 4
typedef unsigned int uint32_t;
# 35 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 2 3 4
# 1 "/usr/include/_types/_uint64_t.h" 1 3 4
# 31 "/usr/include/_types/_uint64_t.h" 3 4
typedef unsigned long long uint64_t;
# 36 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 2 3 4
typedef int8_t int_least8_t;
typedef int16_t int_least16_t;
typedef int32_t int_least32_t;
typedef int64_t int_least64_t;
typedef uint8_t uint_least8_t;
typedef uint16_t uint_least16_t;
typedef uint32_t uint_least32_t;
typedef uint64_t uint_least64_t;
typedef int8_t int_fast8_t;
typedef int16_t int_fast16_t;
typedef int32_t int_fast32_t;
typedef int64_t int_fast64_t;
typedef uint8_t uint_fast8_t;
typedef uint16_t uint_fast16_t;
typedef uint32_t uint_fast32_t;
typedef uint64_t uint_fast64_t;
# 67 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 3 4
# 1 "/usr/include/_types/_intmax_t.h" 1 3 4
# 32 "/usr/include/_types/_intmax_t.h" 3 4
typedef long int intmax_t;
# 68 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 2 3 4
# 1 "/usr/include/_types/_uintmax_t.h" 1 3 4
# 32 "/usr/include/_types/_uintmax_t.h" 3 4
typedef long unsigned int uintmax_t;
# 69 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include-fixed/stdint.h" 2 3 4
# 10 "/opt/local/lib/gcc6/gcc/x86_64-apple-darwin16/6.2.0/include/stdint.h" 2 3 4
# 73 "/usr/include/sys/resource.h" 2 3 4
# 1 "/usr/include/sys/_types/_timeval.h" 1 3 4
# 34 "/usr/include/sys/_types/_timeval.h" 3 4
struct timeval
{
__darwin_time_t tv_sec;
__darwin_suseconds_t tv_usec;
};
# 81 "/usr/include/sys/resource.h" 2 3 4
# 89 "/usr/include/sys/resource.h" 3 4
typedef __uint64_t rlim_t;
# 152 "/usr/include/sys/resource.h" 3 4
struct rusage {
struct timeval ru_utime;
struct timeval ru_stime;
# 163 "/usr/include/sys/resource.h" 3 4
long ru_maxrss;
long ru_ixrss;
long ru_idrss;
long ru_isrss;
long ru_minflt;
long ru_majflt;
long ru_nswap;
long ru_inblock;
long ru_oublock;
long ru_msgsnd;
long ru_msgrcv;
long ru_nsignals;
long ru_nvcsw;
long ru_nivcsw;
};
# 193 "/usr/include/sys/resource.h" 3 4
typedef void *rusage_info_t;
struct rusage_info_v0 {
uint8_t ri_uuid[16];
uint64_t ri_user_time;
uint64_t ri_system_time;
uint64_t ri_pkg_idle_wkups;
uint64_t ri_interrupt_wkups;
uint64_t ri_pageins;
uint64_t ri_wired_size;
uint64_t ri_resident_size;
uint64_t ri_phys_footprint;
uint64_t ri_proc_start_abstime;
uint64_t ri_proc_exit_abstime;
};
struct rusage_info_v1 {
uint8_t ri_uuid[16];
uint64_t ri_user_time;
uint64_t ri_system_time;
uint64_t ri_pkg_idle_wkups;
uint64_t ri_interrupt_wkups;
uint64_t ri_pageins;
uint64_t ri_wired_size;
uint64_t ri_resident_size;
uint64_t ri_phys_footprint;
uint64_t ri_proc_start_abstime;
uint64_t ri_proc_exit_abstime;
uint64_t ri_child_user_time;
uint64_t ri_child_system_time;
uint64_t ri_child_pkg_idle_wkups;
uint64_t ri_child_interrupt_wkups;
uint64_t ri_child_pageins;
uint64_t ri_child_elapsed_abstime;
};
struct rusage_info_v2 {
uint8_t ri_uuid[16];
uint64_t ri_user_time;
uint64_t ri_system_time;
uint64_t ri_pkg_idle_wkups;
uint64_t ri_interrupt_wkups;
uint64_t ri_pageins;
uint64_t ri_wired_size;
uint64_t ri_resident_size;
uint64_t ri_phys_footprint;
uint64_t ri_proc_start_abstime;
uint64_t ri_proc_exit_abstime;
uint64_t ri_child_user_time;
uint64_t ri_child_system_time;
uint64_t ri_child_pkg_idle_wkups;
uint64_t ri_child_interrupt_wkups;
uint64_t ri_child_pageins;
uint64_t ri_child_elapsed_abstime;
uint64_t ri_diskio_bytesread;
uint64_t ri_diskio_byteswritten;
};
struct rusage_info_v3 {
uint8_t ri_uuid[16];
uint64_t ri_user_time;
uint64_t ri_system_time;
uint64_t ri_pkg_idle_wkups;
uint64_t ri_interrupt_wkups;
uint64_t ri_pageins;
uint64_t ri_wired_size;
uint64_t ri_resident_size;
uint64_t ri_phys_footprint;
uint64_t ri_proc_start_abstime;
uint64_t ri_proc_exit_abstime;
uint64_t ri_child_user_time;
uint64_t ri_child_system_time;
uint64_t ri_child_pkg_idle_wkups;
uint64_t ri_child_interrupt_wkups;
uint64_t ri_child_pageins;
uint64_t ri_child_elapsed_abstime;
uint64_t ri_diskio_bytesread;
uint64_t ri_diskio_byteswritten;
uint64_t ri_cpu_time_qos_default;
uint64_t ri_cpu_time_qos_maintenance;
uint64_t ri_cpu_time_qos_background;
uint64_t ri_cpu_time_qos_utility;
uint64_t ri_cpu_time_qos_legacy;
uint64_t ri_cpu_time_qos_user_initiated;
uint64_t ri_cpu_time_qos_user_interactive;
uint64_t ri_billed_system_time;
uint64_t ri_serviced_system_time;
};
struct rusage_info_v4 {
uint8_t ri_uuid[16];
uint64_t ri_user_time;
uint64_t ri_system_time;
uint64_t ri_pkg_idle_wkups;
uint64_t ri_interrupt_wkups;
uint64_t ri_pageins;
uint64_t ri_wired_size;
uint64_t ri_resident_size;
uint64_t ri_phys_footprint;
uint64_t ri_proc_start_abstime;
uint64_t ri_proc_exit_abstime;
uint64_t ri_child_user_time;
uint64_t ri_child_system_time;
uint64_t ri_child_pkg_idle_wkups;
uint64_t ri_child_interrupt_wkups;
uint64_t ri_child_pageins;
uint64_t ri_child_elapsed_abstime;
uint64_t ri_diskio_bytesread;
uint64_t ri_diskio_byteswritten;
uint64_t ri_cpu_time_qos_default;
uint64_t ri_cpu_time_qos_maintenance;
uint64_t ri_cpu_time_qos_background;
uint64_t ri_cpu_time_qos_utility;
uint64_t ri_cpu_time_qos_legacy;
uint64_t ri_cpu_time_qos_user_initiated;
uint64_t ri_cpu_time_qos_user_interactive;
uint64_t ri_billed_system_time;
uint64_t ri_serviced_system_time;
uint64_t ri_logical_writes;
uint64_t ri_lifetime_max_phys_footprint;
uint64_t ri_instructions;
uint64_t ri_cycles;
uint64_t ri_billed_energy;
uint64_t ri_serviced_energy;
uint64_t ri_unused[2];
};
typedef struct rusage_info_v4 rusage_info_current;
# 365 "/usr/include/sys/resource.h" 3 4
struct rlimit {
rlim_t rlim_cur;
rlim_t rlim_max;
};
# 393 "/usr/include/sys/resource.h" 3 4
struct proc_rlimit_control_wakeupmon {
uint32_t wm_flags;
int32_t wm_rate;
};
# 424 "/usr/include/sys/resource.h" 3 4
int getpriority(int, id_t);
int getiopolicy_np(int, int) ;
int getrlimit(int, struct rlimit *) __asm("_" "getrlimit" );
int getrusage(int, struct rusage *);
int setpriority(int, id_t, int);
int setiopolicy_np(int, int, int) ;
int setrlimit(int, const struct rlimit *) __asm("_" "setrlimit" );
# 111 "/usr/include/sys/wait.h" 2 3 4
# 186 "/usr/include/sys/wait.h" 3 4
# 1 "/usr/include/machine/endian.h" 1 3 4
# 35 "/usr/include/machine/endian.h" 3 4
# 1 "/usr/include/i386/endian.h" 1 3 4
# 99 "/usr/include/i386/endian.h" 3 4
# 1 "/usr/include/sys/_endian.h" 1 3 4
# 130 "/usr/include/sys/_endian.h" 3 4
# 1 "/usr/include/libkern/_OSByteOrder.h" 1 3 4
# 66 "/usr/include/libkern/_OSByteOrder.h" 3 4
# 1 "/usr/include/libkern/i386/_OSByteOrder.h" 1 3 4
# 44 "/usr/include/libkern/i386/_OSByteOrder.h" 3 4
static inline
__uint16_t
_OSSwapInt16(
__uint16_t _data
)
{
return ((__uint16_t)((_data << 8) | (_data >> 8)));
}
static inline
__uint32_t
_OSSwapInt32(
__uint32_t _data
)
{
__asm__ ("bswap   %0" : "+r" (_data));
return _data;
}
# 91 "/usr/include/libkern/i386/_OSByteOrder.h" 3 4
static inline
__uint64_t
_OSSwapInt64(
__uint64_t _data
)
{
__asm__ ("bswap   %0" : "+r" (_data));
return _data;
}
# 67 "/usr/include/libkern/_OSByteOrder.h" 2 3 4
# 131 "/usr/include/sys/_endian.h" 2 3 4
# 100 "/usr/include/i386/endian.h" 2 3 4
# 36 "/usr/include/machine/endian.h" 2 3 4
# 187 "/usr/include/sys/wait.h" 2 3 4
union wait {
int w_status;
struct {
unsigned int w_Termsig:7,
w_Coredump:1,
w_Retcode:8,
w_Filler:16;
} w_T;
struct {
unsigned int w_Stopval:8,
w_Stopsig:8,
w_Filler:16;
} w_S;
};
# 247 "/usr/include/sys/wait.h" 3 4
pid_t wait(int *) __asm("_" "wait" );
pid_t waitpid(pid_t, int *, int) __asm("_" "waitpid" );
int waitid(idtype_t, id_t, siginfo_t *, int) __asm("_" "waitid" );
pid_t wait3(int *, int, struct rusage *);
pid_t wait4(pid_t, int *, int, struct rusage *);
# 66 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/alloca.h" 1 3 4
# 31 "/usr/include/alloca.h" 3 4
void *alloca(size_t);
# 68 "/usr/include/stdlib.h" 2 3 4
# 76 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/sys/_types/_ct_rune_t.h" 1 3 4
# 32 "/usr/include/sys/_types/_ct_rune_t.h" 3 4
typedef __darwin_ct_rune_t ct_rune_t;
# 77 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/sys/_types/_rune_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_rune_t.h" 3 4
typedef __darwin_rune_t rune_t;
# 78 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/sys/_types/_wchar_t.h" 1 3 4
# 34 "/usr/include/sys/_types/_wchar_t.h" 3 4
typedef __darwin_wchar_t wchar_t;
# 81 "/usr/include/stdlib.h" 2 3 4
typedef struct {
int quot;
int rem;
} div_t;
typedef struct {
long quot;
long rem;
} ldiv_t;
typedef struct {
long long quot;
long long rem;
} lldiv_t;
# 1 "/usr/include/sys/_types/_null.h" 1 3 4
# 100 "/usr/include/stdlib.h" 2 3 4
# 117 "/usr/include/stdlib.h" 3 4
extern int __mb_cur_max;
# 135 "/usr/include/stdlib.h" 3 4
void abort(void) __attribute__((noreturn));
int abs(int) __attribute__((const));
int atexit(void (* )(void));
double atof(const char *);
int atoi(const char *);
long atol(const char *);
long long
atoll(const char *);
void *bsearch(const void *__key, const void *__base, size_t __nel,
size_t __width, int (* __compar)(const void *, const void *));
void *calloc(size_t __count, size_t __size) __attribute__((__warn_unused_result__)) __attribute__((alloc_size(1,2)));
div_t div(int, int) __attribute__((const));
void exit(int) __attribute__((noreturn));
void free(void *);
char *getenv(const char *);
long labs(long) __attribute__((const));
ldiv_t ldiv(long, long) __attribute__((const));
long long
llabs(long long);
lldiv_t lldiv(long long, long long);
void *malloc(size_t __size) __attribute__((__warn_unused_result__)) __attribute__((alloc_size(1)));
int mblen(const char *__s, size_t __n);
size_t mbstowcs(wchar_t * restrict , const char * restrict, size_t);
int mbtowc(wchar_t * restrict, const char * restrict, size_t);
int posix_memalign(void **__memptr, size_t __alignment, size_t __size) ;
void qsort(void *__base, size_t __nel, size_t __width,
int (* __compar)(const void *, const void *));
int rand(void) ;
void *realloc(void *__ptr, size_t __size) __attribute__((__warn_unused_result__)) __attribute__((alloc_size(2)));
void srand(unsigned) ;
double strtod(const char *, char **) __asm("_" "strtod" );
float strtof(const char *, char **) __asm("_" "strtof" );
long strtol(const char *__str, char **__endptr, int __base);
long double
strtold(const char *, char **);
long long
strtoll(const char *__str, char **__endptr, int __base);
unsigned long
strtoul(const char *__str, char **__endptr, int __base);
unsigned long long
strtoull(const char *__str, char **__endptr, int __base);
# 192 "/usr/include/stdlib.h" 3 4
int system(const char *) __asm("_" "system" );
size_t wcstombs(char * restrict, const wchar_t * restrict, size_t);
int wctomb(char *, wchar_t);
void _Exit(int) __attribute__((noreturn));
long a64l(const char *);
double drand48(void);
char *ecvt(double, int, int *restrict, int *restrict);
double erand48(unsigned short[3]);
char *fcvt(double, int, int *restrict, int *restrict);
char *gcvt(double, int, char *);
int getsubopt(char **, char * const *, char **);
int grantpt(int);
char *initstate(unsigned, char *, size_t);
long jrand48(unsigned short[3]) ;
char *l64a(long);
void lcong48(unsigned short[7]);
long lrand48(void) ;
char *mktemp(char *);
int mkstemp(char *);
long mrand48(void) ;
long nrand48(unsigned short[3]) ;
int posix_openpt(int);
char *ptsname(int);
int putenv(char *) __asm("_" "putenv" );
long random(void) ;
int rand_r(unsigned *) ;
char *realpath(const char * restrict, char * restrict) __asm("_" "realpath" "$DARWIN_EXTSN");
unsigned short
*seed48(unsigned short[3]);
int setenv(const char * __name, const char * __value, int __overwrite) __asm("_" "setenv" );
void setkey(const char *) __asm("_" "setkey" );
char *setstate(const char *);
void srand48(long);
void srandom(unsigned);
int unlockpt(int);
int unsetenv(const char *) __asm("_" "unsetenv" );
# 261 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/sys/_types/_dev_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_dev_t.h" 3 4
typedef __darwin_dev_t dev_t;
# 262 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/sys/_types/_mode_t.h" 1 3 4
# 31 "/usr/include/sys/_types/_mode_t.h" 3 4
typedef __darwin_mode_t mode_t;
# 263 "/usr/include/stdlib.h" 2 3 4
uint32_t arc4random(void);
void arc4random_addrandom(unsigned char * , int )
void arc4random_buf(void * __buf, size_t __nbytes) ;
void arc4random_stir(void);
uint32_t
arc4random_uniform(uint32_t __upper_bound) ;
char *cgetcap(char *, const char *, int);
int cgetclose(void);
int cgetent(char **, char **, const char *);
int cgetfirst(char **, char **);
int cgetmatch(const char *, const char *);
int cgetnext(char **, char **);
int cgetnum(char *, const char *, long *);
int cgetset(const char *);
int cgetstr(char *, const char *, char **);
int cgetustr(char *, const char *, char **);
int daemon(int, int) __asm("_" "daemon" "$1050") __attribute__((deprecated("Use posix_spawn APIs instead."))) ;
char *devname(dev_t, mode_t);
char *devname_r(dev_t, mode_t, char *buf, int len);
char *getbsize(int *, long *);
int getloadavg(double [], int);
const char
*getprogname(void);
int heapsort(void *__base, size_t __nel, size_t __width,
int (* __compar)(const void *, const void *));
int mergesort(void *__base, size_t __nel, size_t __width,
int (* __compar)(const void *, const void *));
void psort(void *__base, size_t __nel, size_t __width,
int (* __compar)(const void *, const void *)) ;
void psort_r(void *__base, size_t __nel, size_t __width, void *,
int (* __compar)(void *, const void *, const void *)) ;
void qsort_r(void *__base, size_t __nel, size_t __width, void *,
int (* __compar)(void *, const void *, const void *));
int radixsort(const unsigned char **__base, int __nel, const unsigned char *__table,
unsigned __endbyte);
void setprogname(const char *);
int sradixsort(const unsigned char **__base, int __nel, const unsigned char *__table,
unsigned __endbyte);
void sranddev(void);
void srandomdev(void);
void *reallocf(void *__ptr, size_t __size) __attribute__((alloc_size(2)));
long long
strtoq(const char *__str, char **__endptr, int __base);
unsigned long long
strtouq(const char *__str, char **__endptr, int __base);
extern char *suboptarg;
void *valloc(size_t) __attribute__((alloc_size(1)));
# 27 "utilities/polybench.h" 2
# 176 "utilities/polybench.h"
# 176 "utilities/polybench.h"
extern double polybench_program_total_flops;
extern void polybench_timer_start();
extern void polybench_timer_stop();
extern void polybench_timer_print();
extern void polybench_timer_start();
extern void polybench_timer_stop();
extern void polybench_timer_print();
# 199 "utilities/polybench.h"
extern void* polybench_alloc_data(unsigned long long int n, int elt_size);
# 4 "bicg.c" 2
# 1 "./linear-algebra/kernels/bicg/bicg.h" 1
# 8 "bicg.c" 2
static
void init_array (int nx, int ny,
double A[4000 + 0][4000 + 0],
double r[4000 + 0],
double p[4000 + 0])
{
int i, j;
for (i = 0; i < ny; i++)
p[i] = i * M_PI;
for (i = 0; i < nx; i++) {
r[i] = i * M_PI;
for (j = 0; j < ny; j++)
A[i][j] = ((double) i*(j+1))/nx;
}
}
static
void print_array(int nx, int ny,
double s[4000 + 0],
double q[4000 + 0])
{
int i;
for (i = 0; i < ny; i++) {
fprintf (stderr, "%0.2lf ", s[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
for (i = 0; i < nx; i++) {
fprintf (stderr, "%0.2lf ", q[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_bicg(int nx, int ny,
double A[4000 + 0][4000 + 0],
double s[4000 + 0],
double q[4000 + 0],
double p[4000 + 0],
double r[4000 + 0])
{
int i, j;
#pragma scop
for (i = 0; i < ny; i++)
s[i] = 0;
for (i = 0; i < nx; i++)
{
q[i] = 0;
for (j = 0; j < ny; j++)
{
s[j] = s[j] + r[i] * A[i][j];
q[i] = q[i] + A[i][j] * p[j];
}
}
#pragma endscop
}
int main(int argc, char** argv)
{
int nx = 4000;
int ny = 4000;
double (*A)[4000 + 0][4000 + 0]; A = (double(*)[4000 + 0][4000 + 0])polybench_alloc_data ((4000 + 0) * (4000 + 0), sizeof(double));;
double (*s)[4000 + 0]; s = (double(*)[4000 + 0])polybench_alloc_data (4000 + 0, sizeof(double));;
double (*q)[4000 + 0]; q = (double(*)[4000 + 0])polybench_alloc_data (4000 + 0, sizeof(double));;
double (*p)[4000 + 0]; p = (double(*)[4000 + 0])polybench_alloc_data (4000 + 0, sizeof(double));;
double (*r)[4000 + 0]; r = (double(*)[4000 + 0])polybench_alloc_data (4000 + 0, sizeof(double));;
init_array (nx, ny,
*A,
*r,
*p);
polybench_timer_start();;
kernel_bicg (nx, ny,
*A,
*s,
*q,
*p,
*r);
polybench_timer_stop();;
polybench_timer_print();;
if (argc > 42 && ! strcmp(argv[0], "")) print_array(nx, ny, *s, *q);
free((void*)A);;
free((void*)s);;
free((void*)q);;
free((void*)p);;
free((void*)r);;
return 0;
}
