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
typedef int __darwin_nl_item;
typedef int __darwin_wctrans_t;
typedef __uint32_t __darwin_wctype_t;
typedef enum {
P_ALL,
P_PID,
P_PGID
} idtype_t;
typedef __darwin_pid_t pid_t;
typedef __darwin_id_t id_t;
typedef int sig_atomic_t;
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
struct __darwin_mmst_reg
{
char __mmst_reg[10];
char __mmst_rsrv[6];
};
struct __darwin_xmm_reg
{
char __xmm_reg[16];
};
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
struct __darwin_i386_exception_state
{
__uint16_t __trapno;
__uint16_t __cpu;
__uint32_t __err;
__uint32_t __faultvaddr;
};
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
struct __darwin_x86_exception_state64
{
__uint16_t __trapno;
__uint16_t __cpu;
__uint32_t __err;
__uint64_t __faultvaddr;
};
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
typedef struct __darwin_mcontext64 *mcontext_t;
typedef __darwin_pthread_attr_t pthread_attr_t;
struct __darwin_sigaltstack
{
void *ss_sp;
__darwin_size_t ss_size;
int ss_flags;
};
typedef struct __darwin_sigaltstack stack_t;
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
typedef __darwin_sigset_t sigset_t;
typedef __darwin_size_t size_t;
typedef __darwin_uid_t uid_t;
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
typedef void (*sig_t)(int);
struct sigvec {
void (*sv_handler)(int);
int sv_mask;
int sv_flags;
};
struct sigstack {
char *ss_sp;
int ss_onstack;
};
void (*signal(int, void (*)(int)))(int);
typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
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
typedef __darwin_intptr_t intptr_t;
typedef unsigned long uintptr_t;
typedef long int intmax_t;
typedef long unsigned int uintmax_t;
struct timeval
{
__darwin_time_t tv_sec;
__darwin_suseconds_t tv_usec;
};
typedef __uint64_t rlim_t;
struct rusage {
struct timeval ru_utime;
struct timeval ru_stime;
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
typedef struct rusage_info_v3 rusage_info_current;
struct rlimit {
rlim_t rlim_cur;
rlim_t rlim_max;
};
struct proc_rlimit_control_wakeupmon {
uint32_t wm_flags;
int32_t wm_rate;
};
int getpriority(int, id_t);
int getiopolicy_np(int, int) ;
int getrlimit(int, struct rlimit *) __asm("_" "getrlimit" );
int getrusage(int, struct rusage *);
int setpriority(int, id_t, int);
int setiopolicy_np(int, int, int) ;
int setrlimit(int, const struct rlimit *) __asm("_" "setrlimit" );
static __inline__
__uint16_t
_OSSwapInt16(
__uint16_t _data
)
{
return ((__uint16_t)((_data << 8) | (_data >> 8)));
}
static __inline__
__uint32_t
_OSSwapInt32(
__uint32_t _data
)
{
__asm__ ("bswap   %0" : "+r" (_data));
return _data;
}
static __inline__
__uint64_t
_OSSwapInt64(
__uint64_t _data
)
{
__asm__ ("bswap   %0" : "+r" (_data));
return _data;
}
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
pid_t wait(int *) __asm("_" "wait" );
pid_t waitpid(pid_t, int *, int) __asm("_" "waitpid" );
int waitid(idtype_t, id_t, siginfo_t *, int) __asm("_" "waitid" );
pid_t wait3(int *, int, struct rusage *);
pid_t wait4(pid_t, int *, int, struct rusage *);
void *alloca(size_t);
typedef __darwin_ct_rune_t ct_rune_t;
typedef __darwin_rune_t rune_t;
typedef __darwin_wchar_t wchar_t;
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
extern int __mb_cur_max;
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
void *calloc(size_t __count, size_t __size) ;
div_t div(int, int) __attribute__((const));
void exit(int) __attribute__((noreturn));
void free(void *);
char *getenv(const char *);
long labs(long) __attribute__((const));
ldiv_t ldiv(long, long) __attribute__((const));
long long
llabs(long long);
lldiv_t lldiv(long long, long long);
void *malloc(size_t __size) ;
int mblen(const char *__s, size_t __n);
size_t mbstowcs(wchar_t * , const char * , size_t);
int mbtowc(wchar_t * , const char * , size_t);
int posix_memalign(void **__memptr, size_t __alignment, size_t __size) ;
void qsort(void *__base, size_t __nel, size_t __width,
int (* __compar)(const void *, const void *));
int rand(void) ;
void *realloc(void *__ptr, size_t __size) ;
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
int system(const char *) __asm("_" "system" );
size_t wcstombs(char * , const wchar_t * , size_t);
int wctomb(char *, wchar_t);
void _Exit(int) __attribute__((noreturn));
long a64l(const char *);
double drand48(void);
char *ecvt(double, int, int *, int *);
double erand48(unsigned short[3]);
char *fcvt(double, int, int *, int *);
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
char *realpath(const char * , char * ) __asm("_" "realpath" "$DARWIN_EXTSN");
unsigned short
*seed48(unsigned short[3]);
int setenv(const char * __name, const char * __value, int __overwrite) __asm("_" "setenv" );
void setkey(const char *) __asm("_" "setkey" );
char *setstate(const char *);
void srand48(long);
void srandom(unsigned);
int unlockpt(int);
int unsetenv(const char *) __asm("_" "unsetenv" );
typedef unsigned char u_int8_t;
typedef unsigned short u_int16_t;
typedef unsigned int u_int32_t;
typedef unsigned long long u_int64_t;
typedef int64_t register_t;
typedef u_int64_t user_addr_t;
typedef u_int64_t user_size_t;
typedef int64_t user_ssize_t;
typedef int64_t user_long_t;
typedef u_int64_t user_ulong_t;
typedef int64_t user_time_t;
typedef int64_t user_off_t;
typedef u_int64_t syscall_arg_t;
typedef __darwin_dev_t dev_t;
typedef __darwin_mode_t mode_t;
uint32_t arc4random(void);
void arc4random_addrandom(unsigned char * , int )
;
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
void *reallocf(void *__ptr, size_t __size);
long long
strtoq(const char *__str, char **__endptr, int __base);
unsigned long long
strtouq(const char *__str, char **__endptr, int __base);
extern char *suboptarg;
void *valloc(size_t);
typedef __darwin_va_list va_list;
int renameat(int, const char *, int, const char *) ;
int renamex_np(const char *, const char *, unsigned int) ;
int renameatx_np(int, const char *, int, const char *, unsigned int) ;
typedef __darwin_off_t fpos_t;
struct __sbuf {
unsigned char *_base;
int _size;
};
struct __sFILEX;
typedef struct __sFILE {
unsigned char *_p;
int _r;
int _w;
short _flags;
short _file;
struct __sbuf _bf;
int _lbfsize;
void *_cookie;
int (* _close)(void *);
int (* _read) (void *, char *, int);
fpos_t (* _seek) (void *, fpos_t, int);
int (* _write)(void *, const char *, int);
struct __sbuf _ub;
struct __sFILEX *_extra;
int _ur;
unsigned char _ubuf[3];
unsigned char _nbuf[1];
struct __sbuf _lb;
int _blksize;
fpos_t _offset;
} FILE;
extern FILE *__stdinp;
extern FILE *__stdoutp;
extern FILE *__stderrp;
void clearerr(FILE *);
int fclose(FILE *);
int feof(FILE *);
int ferror(FILE *);
int fflush(FILE *);
int fgetc(FILE *);
int fgetpos(FILE * , fpos_t *);
char *fgets(char * , int, FILE *);
FILE *fopen(const char * __filename, const char * __mode) __asm("_" "fopen" );
int fprintf(FILE * , const char * , ...) __attribute__((__format__ (__printf__, 2, 3)));
int fputc(int, FILE *);
int fputs(const char * , FILE * ) __asm("_" "fputs" );
size_t fread(void * __ptr, size_t __size, size_t __nitems, FILE * __stream);
FILE *freopen(const char * , const char * ,
FILE * ) __asm("_" "freopen" );
int fscanf(FILE * , const char * , ...) __attribute__((__format__ (__scanf__, 2, 3)));
int fseek(FILE *, long, int);
int fsetpos(FILE *, const fpos_t *);
long ftell(FILE *);
size_t fwrite(const void * __ptr, size_t __size, size_t __nitems, FILE * __stream) __asm("_" "fwrite" );
int getc(FILE *);
int getchar(void);
char *gets(char *);
void perror(const char *);
int printf(const char * , ...) __attribute__((__format__ (__printf__, 1, 2)));
int putc(int, FILE *);
int putchar(int);
int puts(const char *);
int remove(const char *);
int rename (const char *__old, const char *__new);
void rewind(FILE *);
int scanf(const char * , ...) __attribute__((__format__ (__scanf__, 1, 2)));
void setbuf(FILE * , char * );
int setvbuf(FILE * , char * , int, size_t);
int sprintf(char * , const char * , ...) __attribute__((__format__ (__printf__, 2, 3))) ;
int sscanf(const char * , const char * , ...) __attribute__((__format__ (__scanf__, 2, 3)));
FILE *tmpfile(void);
__attribute__((deprecated("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of tmpnam(3), it is highly recommended that you use mkstemp(3) instead.")))
char *tmpnam(char *);
int ungetc(int, FILE *);
int vfprintf(FILE * , const char * , va_list) __attribute__((__format__ (__printf__, 2, 0)));
int vprintf(const char * , va_list) __attribute__((__format__ (__printf__, 1, 0)));
int vsprintf(char * , const char * , va_list) __attribute__((__format__ (__printf__, 2, 0))) ;
char *ctermid(char *);
FILE *fdopen(int, const char *) __asm("_" "fdopen" );
int fileno(FILE *);
int pclose(FILE *) ;
FILE *popen(const char *, const char *) __asm("_" "popen" ) ;
int __srget(FILE *);
int __svfscanf(FILE *, const char *, va_list) __attribute__((__format__ (__scanf__, 2, 0)));
int __swbuf(int, FILE *);
extern __inline __attribute__ ((__always_inline__)) int __sputc(int _c, FILE *_p) {
if (--_p->_w >= 0 || (_p->_w >= _p->_lbfsize && (char)_c != '\n'))
return (*_p->_p++ = _c);
else
return (__swbuf(_c, _p));
}
void flockfile(FILE *);
int ftrylockfile(FILE *);
void funlockfile(FILE *);
int getc_unlocked(FILE *);
int getchar_unlocked(void);
int putc_unlocked(int, FILE *);
int putchar_unlocked(int);
int getw(FILE *);
int putw(int, FILE *);
__attribute__((deprecated("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of tempnam(3), it is highly recommended that you use mkstemp(3) instead.")))
char *tempnam(const char *__dir, const char *__prefix) __asm("_" "tempnam" );
typedef __darwin_off_t off_t;
int fseeko(FILE * __stream, off_t __offset, int __whence);
off_t ftello(FILE * __stream);
int snprintf(char * __str, size_t __size, const char * __format, ...) __attribute__((__format__ (__printf__, 3, 4)));
int vfscanf(FILE * __stream, const char * __format, va_list) __attribute__((__format__ (__scanf__, 2, 0)));
int vscanf(const char * __format, va_list) __attribute__((__format__ (__scanf__, 1, 0)));
int vsnprintf(char * __str, size_t __size, const char * __format, va_list) __attribute__((__format__ (__printf__, 3, 0)));
int vsscanf(const char * __str, const char * __format, va_list) __attribute__((__format__ (__scanf__, 2, 0)));
typedef __darwin_ssize_t ssize_t;
int dprintf(int, const char * , ...) __attribute__((__format__ (__printf__, 2, 3))) ;
int vdprintf(int, const char * , va_list) __attribute__((__format__ (__printf__, 2, 0))) ;
ssize_t getdelim(char ** __linep, size_t * __linecapp, int __delimiter, FILE * __stream) ;
ssize_t getline(char ** __linep, size_t * __linecapp, FILE * __stream) ;
extern const int sys_nerr;
extern const char *const sys_errlist[];
int asprintf(char ** , const char * , ...) __attribute__((__format__ (__printf__, 2, 3)));
char *ctermid_r(char *);
char *fgetln(FILE *, size_t *);
const char *fmtcheck(const char *, const char *);
int fpurge(FILE *);
void setbuffer(FILE *, char *, int);
int setlinebuf(FILE *);
int vasprintf(char ** , const char * , va_list) __attribute__((__format__ (__printf__, 2, 0)));
FILE *zopen(const char *, const char *, int);
FILE *funopen(const void *,
int (* )(void *, char *, int),
int (* )(void *, const char *, int),
fpos_t (* )(void *, fpos_t, int),
int (* )(void *));
extern int __sprintf_chk (char * , int, size_t,
const char * , ...);
extern int __snprintf_chk (char * , size_t, int, size_t,
const char * , ...);
extern int __vsprintf_chk (char * , int, size_t,
const char * , va_list);
extern int __vsnprintf_chk (char * , size_t, int, size_t,
const char * , va_list);
typedef float float_t;
typedef double double_t;
extern int __math_errhandling(void);
extern int __fpclassifyf(float);
extern int __fpclassifyd(double);
extern int __fpclassifyl(long double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isfinitef(float);
extern __inline __attribute__ ((__always_inline__)) int __inline_isfinited(double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isfinitel(long double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isinff(float);
extern __inline __attribute__ ((__always_inline__)) int __inline_isinfd(double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isinfl(long double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isnanf(float);
extern __inline __attribute__ ((__always_inline__)) int __inline_isnand(double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isnanl(long double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isnormalf(float);
extern __inline __attribute__ ((__always_inline__)) int __inline_isnormald(double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isnormall(long double);
extern __inline __attribute__ ((__always_inline__)) int __inline_signbitf(float);
extern __inline __attribute__ ((__always_inline__)) int __inline_signbitd(double);
extern __inline __attribute__ ((__always_inline__)) int __inline_signbitl(long double);
extern __inline __attribute__ ((__always_inline__)) int __inline_isfinitef(float __x) {
return __x == __x && __builtin_fabsf(__x) != __builtin_inff();
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isfinited(double __x) {
return __x == __x && __builtin_fabs(__x) != __builtin_inf();
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isfinitel(long double __x) {
return __x == __x && __builtin_fabsl(__x) != __builtin_infl();
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isinff(float __x) {
return __builtin_fabsf(__x) == __builtin_inff();
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isinfd(double __x) {
return __builtin_fabs(__x) == __builtin_inf();
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isinfl(long double __x) {
return __builtin_fabsl(__x) == __builtin_infl();
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isnanf(float __x) {
return __x != __x;
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isnand(double __x) {
return __x != __x;
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isnanl(long double __x) {
return __x != __x;
}
extern __inline __attribute__ ((__always_inline__)) int __inline_signbitf(float __x) {
union { float __f; unsigned int __u; } __u;
__u.__f = __x;
return (int)(__u.__u >> 31);
}
extern __inline __attribute__ ((__always_inline__)) int __inline_signbitd(double __x) {
union { double __f; unsigned long long __u; } __u;
__u.__f = __x;
return (int)(__u.__u >> 63);
}
extern __inline __attribute__ ((__always_inline__)) int __inline_signbitl(long double __x) {
union {
long double __ld;
struct{ unsigned long long __m; unsigned short __sexp; } __p;
} __u;
__u.__ld = __x;
return (int)(__u.__p.__sexp >> 15);
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isnormalf(float __x) {
return __inline_isfinitef(__x) && __builtin_fabsf(__x) >= 1.17549435082228750797e-38F;
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isnormald(double __x) {
return __inline_isfinited(__x) && __builtin_fabs(__x) >= ((double)2.22507385850720138309e-308L);
}
extern __inline __attribute__ ((__always_inline__)) int __inline_isnormall(long double __x) {
return __inline_isfinitel(__x) && __builtin_fabsl(__x) >= 3.36210314311209350626e-4932L;
}
extern float acosf(float);
extern double acos(double);
extern long double acosl(long double);
extern float asinf(float);
extern double asin(double);
extern long double asinl(long double);
extern float atanf(float);
extern double atan(double);
extern long double atanl(long double);
extern float atan2f(float, float);
extern double atan2(double, double);
extern long double atan2l(long double, long double);
extern float cosf(float);
extern double cos(double);
extern long double cosl(long double);
extern float sinf(float);
extern double sin(double);
extern long double sinl(long double);
extern float tanf(float);
extern double tan(double);
extern long double tanl(long double);
extern float acoshf(float);
extern double acosh(double);
extern long double acoshl(long double);
extern float asinhf(float);
extern double asinh(double);
extern long double asinhl(long double);
extern float atanhf(float);
extern double atanh(double);
extern long double atanhl(long double);
extern float coshf(float);
extern double cosh(double);
extern long double coshl(long double);
extern float sinhf(float);
extern double sinh(double);
extern long double sinhl(long double);
extern float tanhf(float);
extern double tanh(double);
extern long double tanhl(long double);
extern float expf(float);
extern double exp(double);
extern long double expl(long double);
extern float exp2f(float);
extern double exp2(double);
extern long double exp2l(long double);
extern float expm1f(float);
extern double expm1(double);
extern long double expm1l(long double);
extern float logf(float);
extern double log(double);
extern long double logl(long double);
extern float log10f(float);
extern double log10(double);
extern long double log10l(long double);
extern float log2f(float);
extern double log2(double);
extern long double log2l(long double);
extern float log1pf(float);
extern double log1p(double);
extern long double log1pl(long double);
extern float logbf(float);
extern double logb(double);
extern long double logbl(long double);
extern float modff(float, float *);
extern double modf(double, double *);
extern long double modfl(long double, long double *);
extern float ldexpf(float, int);
extern double ldexp(double, int);
extern long double ldexpl(long double, int);
extern float frexpf(float, int *);
extern double frexp(double, int *);
extern long double frexpl(long double, int *);
extern int ilogbf(float);
extern int ilogb(double);
extern int ilogbl(long double);
extern float scalbnf(float, int);
extern double scalbn(double, int);
extern long double scalbnl(long double, int);
extern float scalblnf(float, long int);
extern double scalbln(double, long int);
extern long double scalblnl(long double, long int);
extern float fabsf(float);
extern double fabs(double);
extern long double fabsl(long double);
extern float cbrtf(float);
extern double cbrt(double);
extern long double cbrtl(long double);
extern float hypotf(float, float);
extern double hypot(double, double);
extern long double hypotl(long double, long double);
extern float powf(float, float);
extern double pow(double, double);
extern long double powl(long double, long double);
extern float sqrtf(float);
extern double sqrt(double);
extern long double sqrtl(long double);
extern float erff(float);
extern double erf(double);
extern long double erfl(long double);
extern float erfcf(float);
extern double erfc(double);
extern long double erfcl(long double);
extern float lgammaf(float);
extern double lgamma(double);
extern long double lgammal(long double);
extern float tgammaf(float);
extern double tgamma(double);
extern long double tgammal(long double);
extern float ceilf(float);
extern double ceil(double);
extern long double ceill(long double);
extern float floorf(float);
extern double floor(double);
extern long double floorl(long double);
extern float nearbyintf(float);
extern double nearbyint(double);
extern long double nearbyintl(long double);
extern float rintf(float);
extern double rint(double);
extern long double rintl(long double);
extern long int lrintf(float);
extern long int lrint(double);
extern long int lrintl(long double);
extern float roundf(float);
extern double round(double);
extern long double roundl(long double);
extern long int lroundf(float);
extern long int lround(double);
extern long int lroundl(long double);
extern long long int llrintf(float);
extern long long int llrint(double);
extern long long int llrintl(long double);
extern long long int llroundf(float);
extern long long int llround(double);
extern long long int llroundl(long double);
extern float truncf(float);
extern double trunc(double);
extern long double truncl(long double);
extern float fmodf(float, float);
extern double fmod(double, double);
extern long double fmodl(long double, long double);
extern float remainderf(float, float);
extern double remainder(double, double);
extern long double remainderl(long double, long double);
extern float remquof(float, float, int *);
extern double remquo(double, double, int *);
extern long double remquol(long double, long double, int *);
extern float copysignf(float, float);
extern double copysign(double, double);
extern long double copysignl(long double, long double);
extern float nanf(const char *);
extern double nan(const char *);
extern long double nanl(const char *);
extern float nextafterf(float, float);
extern double nextafter(double, double);
extern long double nextafterl(long double, long double);
extern double nexttoward(double, long double);
extern float nexttowardf(float, long double);
extern long double nexttowardl(long double, long double);
extern float fdimf(float, float);
extern double fdim(double, double);
extern long double fdiml(long double, long double);
extern float fmaxf(float, float);
extern double fmax(double, double);
extern long double fmaxl(long double, long double);
extern float fminf(float, float);
extern double fmin(double, double);
extern long double fminl(long double, long double);
extern float fmaf(float, float, float);
extern double fma(double, double, double);
extern long double fmal(long double, long double, long double);
extern float __inff(void) __attribute__((deprecated));
extern double __inf(void) __attribute__((deprecated));
extern long double __infl(void) __attribute__((deprecated));
extern float __nan(void) ;
extern float __exp10f(float) ;
extern double __exp10(double) ;
extern __inline __attribute__ ((__always_inline__)) void __sincosf(float __x, float *__sinp, float *__cosp);
extern __inline __attribute__ ((__always_inline__)) void __sincos(double __x, double *__sinp, double *__cosp);
extern float __cospif(float) ;
extern double __cospi(double) ;
extern float __sinpif(float) ;
extern double __sinpi(double) ;
extern float __tanpif(float) ;
extern double __tanpi(double) ;
extern __inline __attribute__ ((__always_inline__)) void __sincospif(float __x, float *__sinp, float *__cosp);
extern __inline __attribute__ ((__always_inline__)) void __sincospi(double __x, double *__sinp, double *__cosp);
struct __float2 { float __sinval; float __cosval; };
struct __double2 { double __sinval; double __cosval; };
extern struct __float2 __sincosf_stret(float);
extern struct __double2 __sincos_stret(double);
extern struct __float2 __sincospif_stret(float);
extern struct __double2 __sincospi_stret(double);
extern __inline __attribute__ ((__always_inline__)) void __sincosf(float __x, float *__sinp, float *__cosp) {
const struct __float2 __stret = __sincosf_stret(__x);
*__sinp = __stret.__sinval; *__cosp = __stret.__cosval;
}
extern __inline __attribute__ ((__always_inline__)) void __sincos(double __x, double *__sinp, double *__cosp) {
const struct __double2 __stret = __sincos_stret(__x);
*__sinp = __stret.__sinval; *__cosp = __stret.__cosval;
}
extern __inline __attribute__ ((__always_inline__)) void __sincospif(float __x, float *__sinp, float *__cosp) {
const struct __float2 __stret = __sincospif_stret(__x);
*__sinp = __stret.__sinval; *__cosp = __stret.__cosval;
}
extern __inline __attribute__ ((__always_inline__)) void __sincospi(double __x, double *__sinp, double *__cosp) {
const struct __double2 __stret = __sincospi_stret(__x);
*__sinp = __stret.__sinval; *__cosp = __stret.__cosval;
}
extern double j0(double) ;
extern double j1(double) ;
extern double jn(int, double) ;
extern double y0(double) ;
extern double y1(double) ;
extern double yn(int, double) ;
extern double scalb(double, double);
extern int signgam;
extern long int rinttol(double) __attribute__((deprecated));
extern long int roundtol(double) __attribute__((deprecated));
extern double drem(double, double) __attribute__((deprecated));
extern int finite(double) __attribute__((deprecated));
extern double gamma(double) __attribute__((deprecated));
extern double significand(double) __attribute__((deprecated));
struct exception {
int type;
char *name;
double arg1;
double arg2;
double retval;
};
extern int matherr(struct exception *) __attribute__((deprecated));
typedef __darwin_clock_t clock_t;
typedef __darwin_time_t time_t;
struct timespec
{
__darwin_time_t tv_sec;
long tv_nsec;
};
struct tm {
int tm_sec;
int tm_min;
int tm_hour;
int tm_mday;
int tm_mon;
int tm_year;
int tm_wday;
int tm_yday;
int tm_isdst;
long tm_gmtoff;
char *tm_zone;
};
extern char *tzname[];
extern int getdate_err;
extern long timezone __asm("_" "timezone" );
extern int daylight;
char *asctime(const struct tm *);
clock_t clock(void) __asm("_" "clock" );
char *ctime(const time_t *);
double difftime(time_t, time_t);
struct tm *getdate(const char *);
struct tm *gmtime(const time_t *);
struct tm *localtime(const time_t *);
time_t mktime(struct tm *) __asm("_" "mktime" );
size_t strftime(char * , size_t, const char * , const struct tm * ) __asm("_" "strftime" );
char *strptime(const char * , const char * , struct tm * ) __asm("_" "strptime" );
time_t time(time_t *);
void tzset(void);
char *asctime_r(const struct tm * , char * );
char *ctime_r(const time_t *, char *);
struct tm *gmtime_r(const time_t * , struct tm * );
struct tm *localtime_r(const time_t * , struct tm * );
time_t posix2time(time_t);
void tzsetwall(void);
time_t time2posix(time_t);
time_t timelocal(struct tm * const);
time_t timegm(struct tm * const);
int nanosleep(const struct timespec *__rqtp, struct timespec *__rmtp) __asm("_" "nanosleep" );
typedef enum {
_CLOCK_REALTIME = 0,
_CLOCK_MONOTONIC = 6,
_CLOCK_MONOTONIC_RAW = 4,
_CLOCK_MONOTONIC_RAW_APPROX = 5,
_CLOCK_UPTIME_RAW = 8,
_CLOCK_UPTIME_RAW_APPROX = 9,
_CLOCK_PROCESS_CPUTIME_ID = 12,
_CLOCK_THREAD_CPUTIME_ID = 16
} clockid_t;
int clock_getres(clockid_t __clock_id, struct timespec *__res);
int clock_gettime(clockid_t __clock_id, struct timespec *__tp);
__uint64_t clock_gettime_nsec_np(clockid_t __clock_id);
int clock_settime(clockid_t __clock_id, const struct timespec *__tp);
typedef struct
{
unsigned char _x[64]
__attribute__((__aligned__(8)));
} omp_lock_t;
typedef struct
{
unsigned char _x[80]
__attribute__((__aligned__(8)));
} omp_nest_lock_t;
typedef enum omp_sched_t
{
omp_sched_static = 1,
omp_sched_dynamic = 2,
omp_sched_guided = 3,
omp_sched_auto = 4
} omp_sched_t;
extern void omp_set_num_threads (int) __attribute__((__nothrow__));
extern int omp_get_num_threads (void) __attribute__((__nothrow__));
extern int omp_get_max_threads (void) __attribute__((__nothrow__));
extern int omp_get_thread_num (void) __attribute__((__nothrow__));
extern int omp_get_num_procs (void) __attribute__((__nothrow__));
extern int omp_in_parallel (void) __attribute__((__nothrow__));
extern void omp_set_dynamic (int) __attribute__((__nothrow__));
extern int omp_get_dynamic (void) __attribute__((__nothrow__));
extern void omp_set_nested (int) __attribute__((__nothrow__));
extern int omp_get_nested (void) __attribute__((__nothrow__));
extern void omp_init_lock (omp_lock_t *) __attribute__((__nothrow__));
extern void omp_destroy_lock (omp_lock_t *) __attribute__((__nothrow__));
extern void omp_set_lock (omp_lock_t *) __attribute__((__nothrow__));
extern void omp_unset_lock (omp_lock_t *) __attribute__((__nothrow__));
extern int omp_test_lock (omp_lock_t *) __attribute__((__nothrow__));
extern void omp_init_nest_lock (omp_nest_lock_t *) __attribute__((__nothrow__));
extern void omp_destroy_nest_lock (omp_nest_lock_t *) __attribute__((__nothrow__));
extern void omp_set_nest_lock (omp_nest_lock_t *) __attribute__((__nothrow__));
extern void omp_unset_nest_lock (omp_nest_lock_t *) __attribute__((__nothrow__));
extern int omp_test_nest_lock (omp_nest_lock_t *) __attribute__((__nothrow__));
extern double omp_get_wtime (void) __attribute__((__nothrow__));
extern double omp_get_wtick (void) __attribute__((__nothrow__));
void omp_set_schedule (omp_sched_t, int) __attribute__((__nothrow__));
void omp_get_schedule (omp_sched_t *, int *) __attribute__((__nothrow__));
int omp_get_thread_limit (void) __attribute__((__nothrow__));
void omp_set_max_active_levels (int) __attribute__((__nothrow__));
int omp_get_max_active_levels (void) __attribute__((__nothrow__));
int omp_get_level (void) __attribute__((__nothrow__));
int omp_get_ancestor_thread_num (int) __attribute__((__nothrow__));
int omp_get_team_size (int) __attribute__((__nothrow__));
int omp_get_active_level (void) __attribute__((__nothrow__));
int omp_in_final (void) __attribute__((__nothrow__));
int main ( );
void test01 ( );
void test02 ( );
void test03 ( );
void test04 ( );
float r4_exp ( uint32_t *jsr, uint32_t ke[256], float fe[256], float we[256] );
void r4_exp_setup ( uint32_t ke[256], float fe[256], float we[256] );
float r4_nor ( uint32_t *jsr, uint32_t kn[128], float fn[128], float wn[128] );
void r4_nor_setup ( uint32_t kn[128], float fn[128], float wn[128] );
float r4_uni ( uint32_t *jsr );
uint32_t shr3_seeded ( uint32_t *jsr );
void timestamp ( );
int main ( )
{
timestamp ( );
printf ( "\n" );
printf ( "ZIGGURAT_OPENMP:\n" );
printf ( "  C version\n" );
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
test01 ( );
test02 ( );
test03 ( );
test04 ( );
printf ( "\n" );
printf ( "ZIGGURAT_OPENMP:\n" );
printf ( "  Normal end of execution.\n" );
printf ( "\n" );
timestamp ( );
return 0;
}
void test01 ( )
{
uint32_t jsr;
uint32_t jsr_value;
double mega_rate_par;
double mega_rate_seq;
int r;
int r_num = 1000;
int *result_par;
int *result_seq;
int s;
int s_num = 10000;
uint32_t *seed;
int thread;
int thread_num;
double wtime_par;
double wtime_seq;
printf ( "\n" );
printf ( "TEST01\n" );
printf ( "  SHR3_SEEDED computes random integers.\n" );
printf ( "  Since the output is completely determined\n" );
printf ( "  by the input value of SEED, we can run in\n" );
printf ( "  parallel as long as we make an array of seeds.\n" );
#pragma omp parallel
{
#pragma omp master
{
thread_num = omp_get_num_threads ( );
printf ( "\n" );
printf ( "  The number of threads is %d\n", thread_num );
}
}
seed = ( uint32_t * ) malloc ( thread_num * sizeof ( uint32_t ) );
result_seq = ( int * ) malloc ( thread_num * sizeof ( int ) );
result_par = ( int * ) malloc ( thread_num * sizeof ( int ) );
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_seq = omp_get_wtime ( );
for ( r = 0; r < r_num; r++ )
{
thread = ( r % thread_num );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
jsr_value = shr3_seeded ( &jsr );
}
result_seq[thread] = jsr_value;
seed[thread] = jsr;
}
wtime_seq = omp_get_wtime ( ) - wtime_seq;
mega_rate_seq = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_seq
/ 1000000.0;
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_par = omp_get_wtime ( );
#pragma omp parallel shared ( result_par, seed ) private ( jsr, jsr_value, r, s, thread )
{
#pragma omp for schedule ( static, 1 )
for ( r = 0; r < r_num; r++ )
{
thread = omp_get_thread_num ( );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
jsr_value = shr3_seeded ( &jsr );
}
result_par[thread] = jsr_value;
seed[thread] = jsr;
}
}
wtime_par = omp_get_wtime ( ) - wtime_par;
mega_rate_par = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_par
/ 1000000.0;
printf ( "\n" );
printf ( "  Correctness check:\n" );
printf ( "\n" );
printf ( "  Computing values sequentially should reach the\n" );
printf ( "  same result as doing it in parallel:\n" );
printf ( "\n" );
printf ( "    THREAD    Sequential      Parallel    Difference\n" );
printf ( "\n" );
for ( thread = 0; thread < thread_num; thread++ )
{
printf ( "  %8d  %12d  %12d  %12d\n", thread, result_seq[thread],
result_par[thread], result_seq[thread] - result_par[thread] );
}
printf ( "\n" );
printf ( "  Efficiency check:\n" );
printf ( "\n" );
printf ( "  Computing values in parallel should be faster:\n" );
printf ( "\n" );
printf ( "              Sequential      Parallel\n" );
printf ( "\n" );
printf ( "      TIME:  %14f  %14f\n", wtime_seq, wtime_par );
printf ( "      RATE:  %14f  %14f\n", mega_rate_seq, mega_rate_par );
free ( result_par );
free ( result_seq );
free ( seed );
return;
}
void test02 ( )
{
uint32_t jsr;
uint32_t jsr_value;
double mega_rate_par;
double mega_rate_seq;
int r;
int r_num = 1000;
float r4_value;
float *result_par;
float *result_seq;
int s;
int s_num = 10000;
uint32_t *seed;
int thread;
int thread_num;
double wtime_par;
double wtime_seq;
printf ( "\n" );
printf ( "TEST02\n" );
printf ( "  R4_UNI computes uniformly random single precision real values.\n" );
printf ( "  Since the output is completely determined\n" );
printf ( "  by the input value of SEED, we can run in\n" );
printf ( "  parallel as long as we make an array of seeds.\n" );
#pragma omp parallel
{
#pragma omp master
{
thread_num = omp_get_num_threads ( );
printf ( "\n" );
printf ( "  The number of threads is %d\n", thread_num );
}
}
seed = ( uint32_t * ) malloc ( thread_num * sizeof ( uint32_t ) );
result_seq = ( float * ) malloc ( thread_num * sizeof ( float ) );
result_par = ( float * ) malloc ( thread_num * sizeof ( float ) );
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_seq = omp_get_wtime ( );
for ( r = 0; r < r_num; r++ )
{
thread = ( r % thread_num );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
r4_value = r4_uni ( &jsr );
}
result_seq[thread] = r4_value;
seed[thread] = jsr;
}
wtime_seq = omp_get_wtime ( ) - wtime_seq;
mega_rate_seq = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_seq / 1000000.0;
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_par = omp_get_wtime ( );
#pragma omp parallel shared ( result_par, seed ) private ( jsr, r, r4_value, s, thread )
{
#pragma omp for schedule ( static, 1 )
for ( r = 0; r < r_num; r++ )
{
thread = omp_get_thread_num ( );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
r4_value = r4_uni ( &jsr );
}
result_par[thread] = r4_value;
seed[thread] = jsr;
}
}
wtime_par = omp_get_wtime ( ) - wtime_par;
mega_rate_par = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_par / 1000000.0;
printf ( "\n" );
printf ( "  Correctness check:\n" );
printf ( "\n" );
printf ( "  Computing values sequentially should reach the\n" );
printf ( "  same result as doing it in parallel:\n" );
printf ( "\n" );
printf ( "    THREAD    Sequential        Parallel      Difference\n" );
printf ( "\n" );
for ( thread = 0; thread < thread_num; thread++ )
{
printf ( "  %8d  %14f  %14f  %14f\n", thread, result_seq[thread],
result_par[thread], result_seq[thread] - result_par[thread] );
}
printf ( "\n" );
printf ( "  Efficiency check:\n" );
printf ( "\n" );
printf ( "  Computing values in parallel should be faster:'\n" );
printf ( "\n" );
printf ( "              Sequential      Parallel\n" );
printf ( "\n" );
printf ( "      TIME:  %14f  %14f\n", wtime_seq, wtime_par );
printf ( "      RATE:  %14f  %14f\n", mega_rate_seq, mega_rate_par );
free ( result_par );
free ( result_seq );
free ( seed );
return;
}
void test03 ( )
{
float fn[128];
uint32_t jsr;
uint32_t jsr_value;
uint32_t kn[128];
double mega_rate_par;
double mega_rate_seq;
int r;
int r_num = 1000;
float r4_value;
float *result_par;
float *result_seq;
int s;
int s_num = 10000;
uint32_t *seed;
int thread;
int thread_num;
float wn[128];
double wtime_par;
double wtime_seq;
printf ( "\n" );
printf ( "TEST03\n" );
printf ( "  R4_NOR computes normal random single precision real values.\n" );
printf ( "  Since the output is completely determined\n" );
printf ( "  by the input value of SEED and the tables, we can run in\n" );
printf ( "  parallel as long as we make an array of seeds and share the tables.\n" );
#pragma omp parallel
{
#pragma omp master
{
thread_num = omp_get_num_threads ( );
printf ( "\n" );
printf ( "  The number of threads is %d\n", thread_num );
}
}
seed = ( uint32_t * ) malloc ( thread_num * sizeof ( uint32_t ) );
result_seq = ( float * ) malloc ( thread_num * sizeof ( float ) );
result_par = ( float * ) malloc ( thread_num * sizeof ( float ) );
r4_nor_setup ( kn, fn, wn );
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_seq = omp_get_wtime ( );
for ( r = 0; r < r_num; r++ )
{
thread = ( r % thread_num );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
r4_value = r4_nor ( &jsr, kn, fn, wn );
}
result_seq[thread] = r4_value;
seed[thread] = jsr;
}
wtime_seq = omp_get_wtime ( ) - wtime_seq;
mega_rate_seq = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_seq / 1000000.0;
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_par = omp_get_wtime ( );
#pragma omp parallel shared ( result_par, seed ) private ( jsr, r, r4_value, s, thread )
{
#pragma omp for schedule ( static, 1 )
for ( r = 0; r < r_num; r++ )
{
thread = omp_get_thread_num ( );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
r4_value = r4_nor ( &jsr, kn, fn, wn );
}
result_par[thread] = r4_value;
seed[thread] = jsr;
}
}
wtime_par = omp_get_wtime ( ) - wtime_par;
mega_rate_par = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_par
/ 1000000.0;
printf ( "\n" );
printf ( "  Correctness check:\n" );
printf ( "\n" );
printf ( "  Computing values sequentially should reach the\n" );
printf ( "  same result as doing it in parallel:\n" );
printf ( "\n" );
printf ( "    THREAD    Sequential        Parallel      Difference\n" );
printf ( "\n" );
for ( thread = 0; thread < thread_num; thread++ )
{
printf ( "  %8d  %14f  %14f  %14f\n", thread, result_seq[thread],
result_par[thread], result_seq[thread] - result_par[thread] );
}
printf ( "\n" );
printf ( "  Efficiency check:\n" );
printf ( "\n" );
printf ( "  Computing values in parallel should be faster:\n" );
wprintf ( "\n" );
printf ( "              Sequential      Parallel\n" );
printf ( "\n" );
printf ( "      TIME:  %14f  %14f\n", wtime_seq, wtime_par );
printf ( "      RATE:  %14f  %14f\n", mega_rate_seq, mega_rate_par );
free ( result_par );
free ( result_seq );
free ( seed );
return;
}
void test04 ( )
{
float fe[256];
uint32_t jsr;
uint32_t jsr_value;
uint32_t ke[256];
double mega_rate_par;
double mega_rate_seq;
int r;
int r_num = 1000;
float r4_value;
float *result_par;
float *result_seq;
int s;
int s_num = 10000;
uint32_t *seed;
int thread;
int thread_num;
float we[256];
double wtime_par;
double wtime_seq;
printf ( "\n" );
printf ( "TEST04\n" );
printf ( "  R4_EXP computes exponential random single precision real values.\n" );
printf ( "  Since the output is completely determined\n" );
printf ( "  by the input value of SEED and the tables, we can run in\n" );
printf ( "  parallel as long as we make an array of seeds and share the tables.\n" );
#pragma omp parallel
{
#pragma omp master
{
thread_num = omp_get_num_threads ( );
printf ( "\n" );
printf ( "  The number of threads is %d\n", thread_num );
}
}
seed = ( uint32_t * ) malloc ( thread_num * sizeof ( uint32_t ) );
result_seq = ( float * ) malloc ( thread_num * sizeof ( float ) );
result_par = ( float * ) malloc ( thread_num * sizeof ( float ) );
r4_exp_setup ( ke, fe, we );
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_seq = omp_get_wtime ( );
for ( r = 0; r < r_num; r++ )
{
thread = ( r % thread_num );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
r4_value = r4_exp ( &jsr, ke, fe, we );
}
result_seq[thread] = r4_value;
seed[thread] = jsr;
}
wtime_seq = omp_get_wtime ( ) - wtime_seq;
mega_rate_seq = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_seq
/ 1000000.0;
jsr = 123456789;
for ( thread = 0; thread < thread_num; thread++ )
{
seed[thread] = shr3_seeded ( &jsr );
}
wtime_par = omp_get_wtime ( );
#pragma omp parallel shared ( result_par, seed ) private ( jsr, r, r4_value, s, thread )
{
#pragma omp for schedule ( static, 1 )
for ( r = 0; r < r_num; r++ )
{
thread = omp_get_thread_num ( );
jsr = seed[thread];
for ( s = 0; s < s_num; s++ )
{
r4_value = r4_exp ( &jsr, ke, fe, we );
}
result_par[thread] = r4_value;
seed[thread] = jsr;
}
}
wtime_par = omp_get_wtime ( ) - wtime_par;
mega_rate_par = ( double ) ( r_num ) * ( double ) ( s_num ) / wtime_par
/ 1000000.0;
printf ( "\n" );
printf ( "  Correctness check:\n" );
printf ( "\n" );
printf ( "  Computing values sequentially should reach the\n" );
printf ( "  same result as doing it in parallel:\n" );
printf ( "\n" );
printf ( "    THREAD    Sequential        Parallel      Difference\n" );
printf ( "\n" );
for ( thread = 0; thread < thread_num; thread++ )
{
printf ( "  %8d  %14f  %14f  %14f\n", thread, result_seq[thread],
result_par[thread], result_seq[thread] - result_par[thread] );
}
printf ( "\n" );
printf ( "  Efficiency check:\n" );
printf ( "\n" );
printf ( "  Computing values in parallel should be faster:\n" );
printf ( "\n" );
printf ( "              Sequential      Parallel\n" );
printf ( "\n" );
printf ( "      TIME:  %14f  %14f\n", wtime_seq, wtime_par );
printf ( "      RATE:  %14f  %14f\n", mega_rate_seq, mega_rate_par );
free ( result_par );
free ( result_seq );
free ( seed );
return;
}
float r4_exp ( uint32_t *jsr, uint32_t ke[256], float fe[256], float we[256] )
{
uint32_t iz;
uint32_t jz;
float value;
float x;
jz = shr3_seeded ( jsr );
iz = ( jz & 255 );
if ( jz < ke[iz] )
{
value = ( float ) ( jz ) * we[iz];
}
else
{
for ( ; ; )
{
if ( iz == 0 )
{
value = 7.69711 - log ( r4_uni ( jsr ) );
break;
}
x = ( float ) ( jz ) * we[iz];
if ( fe[iz] + r4_uni ( jsr ) * ( fe[iz-1] - fe[iz] ) < exp ( - x ) )
{
value = x;
break;
}
jz = shr3_seeded ( jsr );
iz = ( jz & 255 );
if ( jz < ke[iz] )
{
value = ( float ) ( jz ) * we[iz];
break;
}
}
}
return value;
}
void r4_exp_setup ( uint32_t ke[256], float fe[256], float we[256] )
{
double de = 7.697117470131487;
int i;
const double m2 = 2147483648.0;
double q;
double te = 7.697117470131487;
const double ve = 3.949659822581572E-03;
q = ve / exp ( - de );
ke[0] = ( uint32_t ) ( ( de / q ) * m2 );
ke[1] = 0;
we[0] = ( float ) ( q / m2 );
we[255] = ( float ) ( de / m2 );
fe[0] = 1.0;
fe[255] = ( float ) ( exp ( - de ) );
for ( i = 254; 1 <= i; i-- )
{
de = - log ( ve / de + exp ( - de ) );
ke[i+1] = ( uint32_t ) ( ( de / te ) * m2 );
te = de;
fe[i] = ( float ) ( exp ( - de ) );
we[i] = ( float ) ( de / m2 );
}
return;
}
float r4_nor ( uint32_t *jsr, uint32_t kn[128], float fn[128], float wn[128] )
{
int hz;
uint32_t iz;
const float r = 3.442620;
float value;
float x;
float y;
hz = ( int ) shr3_seeded ( jsr );
iz = ( hz & 127 );
if ( fabs ( hz ) < kn[iz] )
{
value = ( float ) ( hz ) * wn[iz];
}
else
{
for ( ; ; )
{
if ( iz == 0 )
{
for ( ; ; )
{
x = - 0.2904764 * log ( r4_uni ( jsr ) );
y = - log ( r4_uni ( jsr ) );
if ( x * x <= y + y );
{
break;
}
}
if ( hz <= 0 )
{
value = - r - x;
}
else
{
value = + r + x;
}
break;
}
x = ( float ) ( hz ) * wn[iz];
if ( fn[iz] + r4_uni ( jsr ) * ( fn[iz-1] - fn[iz] )
< exp ( - 0.5 * x * x ) )
{
value = x;
break;
}
hz = ( int ) shr3_seeded ( jsr );
iz = ( hz & 127 );
if ( fabs ( hz ) < kn[iz] )
{
value = ( float ) ( hz ) * wn[iz];
break;
}
}
}
return value;
}
void r4_nor_setup ( uint32_t kn[128], float fn[128], float wn[128] )
{
double dn = 3.442619855899;
int i;
const double m1 = 2147483648.0;
double q;
double tn = 3.442619855899;
const double vn = 9.91256303526217E-03;
q = vn / exp ( - 0.5 * dn * dn );
kn[0] = ( uint32_t ) ( ( dn / q ) * m1 );
kn[1] = 0;
wn[0] = ( float ) ( q / m1 );
wn[127] = ( float ) ( dn / m1 );
fn[0] = 1.0;
fn[127] = ( float ) ( exp ( - 0.5 * dn * dn ) );
for ( i = 126; 1 <= i; i-- )
{
dn = sqrt ( - 2.0 * log ( vn / dn + exp ( - 0.5 * dn * dn ) ) );
kn[i+1] = ( uint32_t ) ( ( dn / tn ) * m1 );
tn = dn;
fn[i] = ( float ) ( exp ( - 0.5 * dn * dn ) );
wn[i] = ( float ) ( dn / m1 );
}
return;
}
float r4_uni ( uint32_t *jsr )
{
uint32_t jsr_input;
float value;
jsr_input = *jsr;
*jsr = ( *jsr ^ ( *jsr << 13 ) );
*jsr = ( *jsr ^ ( *jsr >> 17 ) );
*jsr = ( *jsr ^ ( *jsr << 5 ) );
value = fmod ( 0.5 + ( float ) ( jsr_input + *jsr ) / 65536.0 / 65536.0, 1.0 );
return value;
}
uint32_t shr3_seeded ( uint32_t *jsr )
{
uint32_t value;
value = *jsr;
*jsr = ( *jsr ^ ( *jsr << 13 ) );
*jsr = ( *jsr ^ ( *jsr >> 17 ) );
*jsr = ( *jsr ^ ( *jsr << 5 ) );
value = value + *jsr;
return value;
}
void timestamp ( void )
{
static char time_buffer[40];
const struct tm *tm;
size_t len;
time_t now;
now = time ( ((void *)0) );
tm = localtime ( &now );
len = strftime ( time_buffer, 40, "%d %B %Y %I:%M:%S %p", tm );
printf ( "%s\n", time_buffer );
return;
}
