#ifdef __vxworks
#include "vxWorks.h"
#include "version.h" 
#endif
#ifdef __ANDROID__
#undef __linux__
#endif
#ifdef IN_RTS
#include "tconfig.h"
#include "tsystem.h"
#include <sys/stat.h>
#define xmalloc(S) malloc (S)
#else
#include "config.h"
#include "system.h"
#endif
#include "adaint.h"
#include "raise.h"
#ifdef __cplusplus
extern "C" {
#endif
extern void __gnat_raise_program_error (const char *, int);
extern struct Exception_Data constraint_error;
extern struct Exception_Data numeric_error;
extern struct Exception_Data program_error;
extern struct Exception_Data storage_error;
#ifdef CERT
#define Raise_From_Signal_Handler \
__gnat_raise_exception
extern void Raise_From_Signal_Handler (struct Exception_Data *, const char *);
#else
#define Raise_From_Signal_Handler \
ada__exceptions__raise_from_signal_handler
extern void Raise_From_Signal_Handler (struct Exception_Data *, const char *);
#endif
int   __gl_main_priority                 = -1;
int   __gl_main_cpu                      = -1;
int   __gl_time_slice_val                = -1;
char  __gl_wc_encoding                   = 'n';
char  __gl_locking_policy                = ' ';
char  __gl_queuing_policy                = ' ';
char  __gl_task_dispatching_policy       = ' ';
char *__gl_priority_specific_dispatching = 0;
int   __gl_num_specific_dispatching      = 0;
char *__gl_interrupt_states              = 0;
int   __gl_num_interrupt_states          = 0;
int   __gl_unreserve_all_interrupts      = 0;
int   __gl_exception_tracebacks          = 0;
int   __gl_exception_tracebacks_symbolic = 0;
int   __gl_detect_blocking               = 0;
int   __gl_default_stack_size            = -1;
int   __gl_leap_seconds_support          = 0;
int   __gl_canonical_streams             = 0;
char *__gl_bind_env_addr                 = NULL;
int   __gl_zero_cost_exceptions          = 0;
int  __gnat_handler_installed      = 0;
#ifndef IN_RTS
int __gnat_inside_elab_final_code = 0;
#endif
#undef HAVE_GNAT_INIT_FLOAT
char __gnat_get_interrupt_state (int);
char
__gnat_get_interrupt_state (int intrup)
{
if (intrup >= __gl_num_interrupt_states)
return 'n';
else
return __gl_interrupt_states [intrup];
}
char __gnat_get_specific_dispatching (int);
char
__gnat_get_specific_dispatching (int priority)
{
if (__gl_num_specific_dispatching == 0)
return ' ';
else if (priority >= __gl_num_specific_dispatching)
return 'F';
else
return __gl_priority_specific_dispatching [priority];
}
#ifndef IN_RTS
void
__gnat_set_globals (void)
{
}
#endif
#if defined (_AIX)
#include <signal.h>
#include <sys/time.h>
#ifndef SA_NODEFER
#define SA_NODEFER 0
#endif 
#ifndef _AIXVERSION_430
extern int nanosleep (struct timestruc_t *, struct timestruc_t *);
int
nanosleep (struct timestruc_t *Rqtp, struct timestruc_t *Rmtp)
{
return nsleep (Rqtp, Rmtp);
}
#endif 
static void
__gnat_error_handler (int sig,
siginfo_t *si ATTRIBUTE_UNUSED,
void *ucontext ATTRIBUTE_UNUSED)
{
struct Exception_Data *exception;
const char *msg;
switch (sig)
{
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &constraint_error;
msg = "SIGBUS";
break;
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
void
__gnat_install_handler (void)
{
struct sigaction act;
act.sa_flags = SA_NODEFER | SA_RESTART | SA_SIGINFO;
act.sa_sigaction = __gnat_error_handler;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGABRT) != 's')
sigaction (SIGABRT, &act, NULL);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGILL) != 's')
sigaction (SIGILL,  &act, NULL);
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
sigaction (SIGSEGV, &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined (__hpux__)
#include <signal.h>
#include <sys/ucontext.h>
#if defined (IN_RTS) && defined (__ia64__)
#include <sys/uc_access.h>
#define HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED, void *ucontext)
{
ucontext_t *uc = (ucontext_t *) ucontext;
uint64_t ip;
__uc_get_ip (uc, &ip);
__uc_set_ip (uc, ip + 1);
}
#endif 
static void
__gnat_error_handler (int sig, siginfo_t *si ATTRIBUTE_UNUSED, void *ucontext)
{
struct Exception_Data *exception;
const char *msg;
__gnat_adjust_context_for_raise (sig, ucontext);
switch (sig)
{
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &constraint_error;
msg = "SIGBUS";
break;
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
#if defined (__hppa__)
char __gnat_alternate_stack[16 * 1024]; 
#else
char __gnat_alternate_stack[128 * 1024]; 
#endif
void
__gnat_install_handler (void)
{
struct sigaction act;
stack_t stack;
stack.ss_sp = __gnat_alternate_stack;
stack.ss_size = sizeof (__gnat_alternate_stack);
stack.ss_flags = 0;
sigaltstack (&stack, NULL);
act.sa_sigaction = __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_RESTART | SA_SIGINFO;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGABRT) != 's')
sigaction (SIGABRT, &act, NULL);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGILL) != 's')
sigaction (SIGILL,  &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
act.sa_flags |= SA_ONSTACK;
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
sigaction (SIGSEGV, &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined (__linux__)
#include <signal.h>
#define __USE_GNU 1 
#include <sys/ucontext.h>
#if !defined (NULL)
#define NULL ((void *) 0)
#endif
#if defined (MaRTE)
#pragma weak linux_sigaction
int linux_sigaction (int signum, const struct sigaction *act,
struct sigaction *oldact)
{
return sigaction (signum, act, oldact);
}
#define sigaction(signum, act, oldact) linux_sigaction (signum, act, oldact)
#pragma weak fake_linux_sigfillset
void fake_linux_sigfillset (sigset_t *set)
{
sigfillset (set);
}
#define sigfillset(set) fake_linux_sigfillset (set)
#pragma weak fake_linux_sigemptyset
void fake_linux_sigemptyset (sigset_t *set)
{
sigemptyset (set);
}
#define sigemptyset(set) fake_linux_sigemptyset (set)
#endif
#if defined (__i386__) || defined (__x86_64__) || defined (__ia64__) \
|| defined (__ARMEL__)
#define HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED, void *ucontext)
{
mcontext_t *mcontext = &((ucontext_t *) ucontext)->uc_mcontext;
#if defined (__i386__)
unsigned long *pc = (unsigned long *)mcontext->gregs[REG_EIP];
if (signo == SIGSEGV && pc && *pc == 0x00240c83)
mcontext->gregs[REG_ESP] += 4096 + 4 * sizeof (unsigned long);
#elif defined (__x86_64__)
unsigned long long *pc = (unsigned long long *)mcontext->gregs[REG_RIP];
if (signo == SIGSEGV && pc
&& ((*pc & 0xffffffffffLL) == 0x00240c8348LL
|| (*pc & 0xffffffffLL) == 0x00240c83LL))
mcontext->gregs[REG_RSP] += 4096 + 4 * sizeof (unsigned long);
#elif defined (__ia64__)
mcontext->sc_ip++;
#elif defined (__ARMEL__)
mcontext->arm_pc+=2;
#ifdef __thumb2__
#define CPSR_THUMB_BIT 5
if (mcontext->arm_cpsr & (1<<CPSR_THUMB_BIT))
mcontext->arm_pc+=1;
#endif
#endif
}
#endif
static void
__gnat_error_handler (int sig, siginfo_t *si ATTRIBUTE_UNUSED, void *ucontext)
{
struct Exception_Data *exception;
const char *msg;
__gnat_adjust_context_for_raise (sig, ucontext);
switch (sig)
{
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &storage_error;
msg = "SIGBUS: possible stack overflow";
break;
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
#ifndef __ia64__
#define HAVE_GNAT_ALTERNATE_STACK 1
# if 16 * 1024 < MINSIGSTKSZ
#  error "__gnat_alternate_stack too small"
# endif
char __gnat_alternate_stack[16 * 1024];
#endif
#ifdef __XENO__
#include <sys/mman.h>
#include <native/task.h>
RT_TASK main_task;
#endif
void
__gnat_install_handler (void)
{
struct sigaction act;
#ifdef __XENO__
int prio;
if (__gl_main_priority == -1)
prio = 49;
else
prio = __gl_main_priority;
mlockall (MCL_CURRENT|MCL_FUTURE);
rt_task_shadow (&main_task, "environment_task", prio, T_FPU);
#endif
act.sa_sigaction = __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_RESTART | SA_SIGINFO;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGABRT) != 's')
sigaction (SIGABRT, &act, NULL);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGILL) != 's')
sigaction (SIGILL,  &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
{
#ifdef HAVE_GNAT_ALTERNATE_STACK
stack_t stack;
stack.ss_sp = __gnat_alternate_stack;
stack.ss_size = sizeof (__gnat_alternate_stack);
stack.ss_flags = 0;
sigaltstack (&stack, NULL);
act.sa_flags |= SA_ONSTACK;
#endif
sigaction (SIGSEGV, &act, NULL);
}
__gnat_handler_installed = 1;
}
#elif defined (__Lynx__)
#include <signal.h>
#include <unistd.h>
static void
__gnat_error_handler (int sig)
{
struct Exception_Data *exception;
const char *msg;
switch(sig)
{
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
case SIGILL:
exception = &constraint_error;
msg = "SIGILL";
break;
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &constraint_error;
msg = "SIGBUS";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
void
__gnat_install_handler (void)
{
struct sigaction act;
act.sa_handler = __gnat_error_handler;
act.sa_flags = 0x0;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGILL) != 's')
sigaction (SIGILL,  &act, NULL);
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
sigaction (SIGSEGV, &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined (__sun__) && !defined (__vxworks)
#include <signal.h>
#include <siginfo.h>
#include <sys/ucontext.h>
#include <sys/regset.h>
static void
__gnat_error_handler (int sig, siginfo_t *si, void *ucontext ATTRIBUTE_UNUSED)
{
struct Exception_Data *exception;
static int recurse = 0;
const char *msg;
switch (sig)
{
case SIGSEGV:
if (si->si_code == SEGV_ACCERR
|| (long) si->si_addr == 0
|| (((long) si->si_addr) & 3) != 0
|| recurse)
{
exception = &constraint_error;
msg = "SIGSEGV";
}
else
{
recurse++;
((volatile char *)
((long) si->si_addr & - getpagesize ()))[getpagesize ()];
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
}
break;
case SIGBUS:
exception = &program_error;
msg = "SIGBUS";
break;
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
recurse = 0;
Raise_From_Signal_Handler (exception, msg);
}
void
__gnat_install_handler (void)
{
struct sigaction act;
act.sa_sigaction = __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_RESTART | SA_SIGINFO;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGABRT) != 's')
sigaction (SIGABRT, &act, NULL);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
sigaction (SIGSEGV, &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined (VMS)
void __gnat_set_features (void);
int __gnat_features_set = 0;
void (*__gnat_ctrl_c_handler) (void) = 0;
#ifdef __IA64
#define lib_get_curr_invo_context LIB$I64_GET_CURR_INVO_CONTEXT
#define lib_get_prev_invo_context LIB$I64_GET_PREV_INVO_CONTEXT
#define lib_get_invo_handle LIB$I64_GET_INVO_HANDLE
#else
#define lib_get_curr_invo_context LIB$GET_CURR_INVO_CONTEXT
#define lib_get_prev_invo_context LIB$GET_PREV_INVO_CONTEXT
#define lib_get_invo_handle LIB$GET_INVO_HANDLE
#endif
#define FAC_MASK  		0x0fff0000
#define DECADA_M_FACILITY	0x00310000
#define SS$_CONTINUE           1
#define SS$_ACCVIO            12
#define SS$_HPARITH         1284
#define SS$_INTDIV          1156
#define SS$_STKOVF          1364
#define SS$_CONTROLC        1617
#define SS$_RESIGNAL        2328
#define MTH$_FLOOVEMAT   1475268       
extern int C$_SIGKILL;
extern int C$_SIGINT;
extern int SS$_DEBUG;
extern int LIB$_KEYNOTFOU;
extern int LIB$_ACTIMAGE;
#define RDB$_STREAM_EOF 20480426
#define FDL$_UNPRIKW 11829410
#define CMA$_EXIT_THREAD 4227492
struct cond_sigargs
{
unsigned int sigarg;
unsigned int sigargval;
};
struct cond_subtests
{
unsigned int num;
const struct cond_sigargs sigargs[];
};
struct cond_except
{
unsigned int cond;
const struct Exception_Data *except;
unsigned int needs_adjust;  
const struct cond_subtests *subtests;
};
struct descriptor_s
{
unsigned short len, mbz;
__char_ptr32 adr;
};
#ifdef IN_RTS
#define Status_Error ada__io_exceptions__status_error
extern struct Exception_Data Status_Error;
#define Mode_Error ada__io_exceptions__mode_error
extern struct Exception_Data Mode_Error;
#define Name_Error ada__io_exceptions__name_error
extern struct Exception_Data Name_Error;
#define Use_Error ada__io_exceptions__use_error
extern struct Exception_Data Use_Error;
#define Device_Error ada__io_exceptions__device_error
extern struct Exception_Data Device_Error;
#define End_Error ada__io_exceptions__end_error
extern struct Exception_Data End_Error;
#define Data_Error ada__io_exceptions__data_error
extern struct Exception_Data Data_Error;
#define Layout_Error ada__io_exceptions__layout_error
extern struct Exception_Data Layout_Error;
#define Non_Ada_Error system__aux_dec__non_ada_error
extern struct Exception_Data Non_Ada_Error;
#define Coded_Exception system__vms_exception_table__coded_exception
extern struct Exception_Data *Coded_Exception (void *);
#define Base_Code_In system__vms_exception_table__base_code_in
extern void *Base_Code_In (void *);
#define ADA$_ALREADY_OPEN	0x0031a594
#define ADA$_CONSTRAINT_ERRO	0x00318324
#define ADA$_DATA_ERROR		0x003192c4
#define ADA$_DEVICE_ERROR	0x003195e4
#define ADA$_END_ERROR		0x00319904
#define ADA$_FAC_MODE_MISMAT	0x0031a8b3
#define ADA$_IOSYSFAILED	0x0031af04
#define ADA$_KEYSIZERR		0x0031aa3c
#define ADA$_KEY_MISMATCH	0x0031a8e3
#define ADA$_LAYOUT_ERROR	0x00319c24
#define ADA$_LINEXCMRS		0x0031a8f3
#define ADA$_MAXLINEXC		0x0031a8eb
#define ADA$_MODE_ERROR		0x00319f44
#define ADA$_MRN_MISMATCH	0x0031a8db
#define ADA$_MRS_MISMATCH	0x0031a8d3
#define ADA$_NAME_ERROR		0x0031a264
#define ADA$_NOT_OPEN		0x0031a58c
#define ADA$_ORG_MISMATCH	0x0031a8bb
#define ADA$_PROGRAM_ERROR	0x00318964
#define ADA$_RAT_MISMATCH	0x0031a8cb
#define ADA$_RFM_MISMATCH	0x0031a8c3
#define ADA$_STAOVF		0x00318cac
#define ADA$_STATUS_ERROR	0x0031a584
#define ADA$_STORAGE_ERROR	0x00318c84
#define ADA$_UNSUPPORTED	0x0031a8ab
#define ADA$_USE_ERROR		0x0031a8a4
static const struct cond_except dec_ada_cond_except_table [] =
{
{ADA$_PROGRAM_ERROR,   &program_error, 0, 0},
{ADA$_USE_ERROR,       &Use_Error, 0, 0},
{ADA$_KEYSIZERR,       &program_error, 0, 0},
{ADA$_STAOVF,          &storage_error, 0, 0},
{ADA$_CONSTRAINT_ERRO, &constraint_error, 0, 0},
{ADA$_IOSYSFAILED,     &Device_Error, 0, 0},
{ADA$_LAYOUT_ERROR,    &Layout_Error, 0, 0},
{ADA$_STORAGE_ERROR,   &storage_error, 0, 0},
{ADA$_DATA_ERROR,      &Data_Error, 0, 0},
{ADA$_DEVICE_ERROR,    &Device_Error, 0, 0},
{ADA$_END_ERROR,       &End_Error, 0, 0},
{ADA$_MODE_ERROR,      &Mode_Error, 0, 0},
{ADA$_NAME_ERROR,      &Name_Error, 0, 0},
{ADA$_STATUS_ERROR,    &Status_Error, 0, 0},
{ADA$_NOT_OPEN,        &Use_Error, 0, 0},
{ADA$_ALREADY_OPEN,    &Use_Error, 0, 0},
{ADA$_USE_ERROR,       &Use_Error, 0, 0},
{ADA$_UNSUPPORTED,     &Use_Error, 0, 0},
{ADA$_FAC_MODE_MISMAT, &Use_Error, 0, 0},
{ADA$_ORG_MISMATCH,    &Use_Error, 0, 0},
{ADA$_RFM_MISMATCH,    &Use_Error, 0, 0},
{ADA$_RAT_MISMATCH,    &Use_Error, 0, 0},
{ADA$_MRS_MISMATCH,    &Use_Error, 0, 0},
{ADA$_MRN_MISMATCH,    &Use_Error, 0, 0},
{ADA$_KEY_MISMATCH,    &Use_Error, 0, 0},
{ADA$_MAXLINEXC,       &constraint_error, 0, 0},
{ADA$_LINEXCMRS,       &constraint_error, 0, 0},
#if 0
{ADA$_LOCK_ERROR,      &Lock_Error, 0, 0},
{ADA$_EXISTENCE_ERROR, &Existence_Error, 0, 0},
{ADA$_KEY_ERROR,       &Key_Error, 0, 0},
#endif
{0,                    0, 0, 0}
};
#endif 
#define ACCVIO_VIRTUAL_ADDR 3
static const struct cond_subtests accvio_c_e =
{1,  
{
{ ACCVIO_VIRTUAL_ADDR, 0 }
}
};
#define NEEDS_ADJUST 1
static const struct cond_except system_cond_except_table [] =
{
{MTH$_FLOOVEMAT, &constraint_error, 0, 0},
{SS$_INTDIV,     &constraint_error, 0, 0},
{SS$_HPARITH,    &constraint_error, NEEDS_ADJUST, 0},
{SS$_ACCVIO,     &constraint_error, NEEDS_ADJUST, &accvio_c_e},
{SS$_ACCVIO,     &storage_error,    NEEDS_ADJUST, 0},
{SS$_STKOVF,     &storage_error,    NEEDS_ADJUST, 0},
{0,               0, 0, 0}
};
typedef int resignal_predicate (int code);
static const int * const cond_resignal_table [] =
{
&C$_SIGKILL,
(int *)CMA$_EXIT_THREAD,
&SS$_DEBUG,
&LIB$_KEYNOTFOU,
&LIB$_ACTIMAGE,
(int *) RDB$_STREAM_EOF,
(int *) FDL$_UNPRIKW,
0
};
static const int facility_resignal_table [] =
{
0x1380000, 
0x2220000, 
0
};
static int
__gnat_default_resignal_p (int code)
{
int i, iexcept;
for (i = 0; facility_resignal_table [i]; i++)
if ((code & FAC_MASK) == facility_resignal_table [i])
return 1;
for (i = 0, iexcept = 0;
cond_resignal_table [i]
&& !(iexcept = LIB$MATCH_COND (&code, &cond_resignal_table [i]));
i++);
return iexcept;
}
static resignal_predicate *__gnat_resignal_p = __gnat_default_resignal_p;
void
__gnat_set_resignal_predicate (resignal_predicate *predicate)
{
if (predicate == NULL)
__gnat_resignal_p = __gnat_default_resignal_p;
else
__gnat_resignal_p = predicate;
}
#define Default_Exception_Msg_Max_Length 512
static int
copy_msg (struct descriptor_s *msgdesc, char *message)
{
int len = strlen (message);
int copy_len;
if (len > 0 && len <= Default_Exception_Msg_Max_Length - 3)
{
strcat (message, "\r\n");
len += 2;
}
copy_len = (len + msgdesc->len <= Default_Exception_Msg_Max_Length - 1 ?
msgdesc->len :
Default_Exception_Msg_Max_Length - 1 - len);
strncpy (&message [len], msgdesc->adr, copy_len);
message [len + copy_len] = 0;
return 0;
}
static const struct cond_except *
scan_conditions ( int *sigargs, const struct cond_except *table [])
{
int i;
struct cond_except entry;
for (i = 0; (*table) [i].cond; i++)
{
unsigned int match = LIB$MATCH_COND (&sigargs [1], &(*table) [i].cond);
const struct cond_subtests *subtests  = (*table) [i].subtests;
if (match)
{
if (!subtests)
{
return &(*table) [i];
}
else
{
unsigned int ii;
int num = (*subtests).num;
for (ii = 0; ii < num; ii++)
{
unsigned int arg = (*subtests).sigargs [ii].sigarg;
unsigned int argval = (*subtests).sigargs [ii].sigargval;
if (sigargs [arg] != argval)
{
num = 0;
break;
}
}
if (num == (*subtests).num)
return &(*table) [i];
}
}
}
return &(*table) [i];
}
long
__gnat_handle_vms_condition (int *sigargs, void *mechargs)
{
struct Exception_Data *exception = 0;
unsigned int needs_adjust = 0;
void *base_code;
struct descriptor_s gnat_facility = {4, 0, "GNAT"};
char message [Default_Exception_Msg_Max_Length];
const char *msg = "";
if (__gnat_resignal_p (sigargs [1]))
return SS$_RESIGNAL;
#ifndef IN_RTS
if (sigargs [1] == SS$_HPARITH)
return SS$_RESIGNAL;
#endif
#ifdef IN_RTS
base_code = Base_Code_In ((void *) sigargs[1]);
exception = Coded_Exception (base_code);
#endif
if (exception == 0)
#ifdef IN_RTS
{
int i;
struct cond_except cond;
const struct cond_except *cond_table;
const struct cond_except *cond_tables [] = {dec_ada_cond_except_table,
system_cond_except_table,
0};
unsigned int ctrlc = SS$_CONTROLC;
unsigned int *sigint = &C$_SIGINT;
int ctrlc_match = LIB$MATCH_COND (&sigargs [1], &ctrlc);
int sigint_match = LIB$MATCH_COND (&sigargs [1], &sigint);
extern int SYS$DCLAST (void (*astadr)(), unsigned long long astprm,
unsigned int acmode);
if ((ctrlc_match || sigint_match) && __gnat_ctrl_c_handler)
{
SYS$DCLAST (__gnat_ctrl_c_handler, 0, 0);
return SS$_CONTINUE;
}
i = 0;
while ((cond_table = cond_tables[i++]) && !exception)
{
cond = *scan_conditions (sigargs, &cond_table);
exception = (struct Exception_Data *) cond.except;
}
if (exception)
needs_adjust = cond.needs_adjust;
else
exception = &Non_Ada_Error;
}
#else
{
exception = &program_error;
}
#endif
message[0] = 0;
sigargs[0] -= 2;
extern int SYS$PUTMSG (void *, int (*)(), void *, unsigned long long);
if ((sigargs [1] & FAC_MASK) == DECADA_M_FACILITY)
SYS$PUTMSG (sigargs, copy_msg, &gnat_facility,
(unsigned long long ) message);
else
SYS$PUTMSG (sigargs, copy_msg, 0,
(unsigned long long ) message);
sigargs[0] += 2;
msg = message;
if (needs_adjust)
__gnat_adjust_context_for_raise (sigargs [1], (void *)mechargs);
Raise_From_Signal_Handler (exception, msg);
}
#if defined (IN_RTS) && defined (__IA64)
void
GNAT$STOP (int *sigargs)
{
sigargs [0] += 2;
__gnat_handle_vms_condition (sigargs, 0);
}
#endif
void
__gnat_install_handler (void)
{
long prvhnd ATTRIBUTE_UNUSED;
#if !defined (IN_RTS)
extern int SYS$SETEXV (unsigned int vector, int (*addres)(),
unsigned int accmode, void *(*(prvhnd)));
SYS$SETEXV (1, __gnat_handle_vms_condition, 3, &prvhnd);
#endif
__gnat_handler_installed = 1;
}
#if defined (IN_RTS) && defined (__alpha__)
#include <vms/chfctxdef.h>
#include <vms/chfdef.h>
#define HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED, void *ucontext)
{
if (signo == SS$_HPARITH)
{
CHF$MECH_ARRAY * mechargs = (CHF$MECH_ARRAY *) ucontext;
CHF$SIGNAL_ARRAY * sigargs
= (CHF$SIGNAL_ARRAY *) mechargs->chf$q_mch_sig_addr;
int vcount = sigargs->chf$is_sig_args;
int * pc_slot = & (&sigargs->chf$l_sig_name)[vcount-2];
(*pc_slot)--;
}
}
#endif
#if defined (IN_RTS) && defined (__IA64)
#include <vms/chfctxdef.h>
#include <vms/chfdef.h>
#define HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
typedef unsigned long long u64;
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED, void *ucontext)
{
CHF$MECH_ARRAY * mechargs = (CHF$MECH_ARRAY *) ucontext;
CHF64$SIGNAL_ARRAY *chfsig64
= (CHF64$SIGNAL_ARRAY *) mechargs->chf$ph_mch_sig64_addr;
u64 * post_sigarray
= (u64 *)chfsig64 + 1 + chfsig64->chf64$l_sig_args;
u64 * ih_pc_loc = post_sigarray - 2;
(*ih_pc_loc) ++;
}
#endif
static void
__gnat_vms_get_logical (const char *name, char *buf, int len)
{
struct descriptor_s name_desc, result_desc;
int status;
unsigned short rlen;
name_desc.len = strlen (name);
name_desc.mbz = 0;
name_desc.adr = (char *)name;
result_desc.len = len;
result_desc.mbz = 0;
result_desc.adr = buf;
status = LIB$GET_LOGICAL (&name_desc, &result_desc, &rlen);
if ((status & 1) == 1 && rlen < len)
buf[rlen] = 0;
else
buf[0] = 0;
}
#define VMS_PAGESIZE 8192
#define PSL__C_USER 3
#define PRT__C_NA 0
#define VA__M_DESCEND 1
#define VA___REGSUM_BY_VA 1
struct regsum
{
unsigned long long q_region_id;
unsigned int l_flags;
unsigned int l_region_protection;
void *pq_start_va;
unsigned long long q_region_size;
void *pq_first_free_va;
};
extern int SYS$GET_REGION_INFO (unsigned int, unsigned long long *,
void *, void *, unsigned int,
void *, unsigned int *);
extern int SYS$EXPREG_64 (unsigned long long *, unsigned long long,
unsigned int, unsigned int, void **,
unsigned long long *);
extern int SYS$SETPRT_64 (void *, unsigned long long, unsigned int,
unsigned int, void **, unsigned long long *,
unsigned int *);
static int
__gnat_set_stack_guard_page (void *addr, unsigned long size)
{
int status;
void *ret_va;
unsigned long long ret_len;
unsigned int ret_prot;
void *start_va;
unsigned long long length;
unsigned int retlen;
struct regsum buffer;
status = SYS$GET_REGION_INFO
(VA___REGSUM_BY_VA, NULL, addr, NULL, sizeof (buffer), &buffer, &retlen);
if ((status & 1) != 1)
return -1;
status = SYS$EXPREG_64 (&buffer.q_region_id,
size, 0, 0, &start_va, &length);
if ((status & 1) != 1)
return -1;
if (!(buffer.l_flags & VA__M_DESCEND))
start_va = (void *)((unsigned long long)start_va + length - VMS_PAGESIZE);
status = SYS$SETPRT_64 (start_va, VMS_PAGESIZE, PSL__C_USER, PRT__C_NA,
&ret_va, &ret_len, &ret_prot);
if ((status & 1) != 1)
return -1;
return 0;
}
static void
__gnat_set_stack_limit (void)
{
#ifdef __ia64__
void *sp;
unsigned long size;
char value[16];
char *e;
__gnat_vms_get_logical ("GNAT_STACK_SIZE", value, sizeof (value));
size = strtoul (value, &e, 0);
if (e > value && *e == 0)
{
asm ("mov %0=sp" : "=r" (sp));
__gnat_set_stack_guard_page (sp, size * 1024);
}
__gnat_vms_get_logical ("GNAT_RBS_SIZE", value, sizeof (value));
size = strtoul (value, &e, 0);
if (e > value && *e == 0)
{
asm ("mov %0=ar.bsp" : "=r" (sp));
__gnat_set_stack_guard_page (sp, size * 1024);
}
#endif
}
#ifdef IN_RTS
extern int SYS$IEEE_SET_FP_CONTROL (void *, void *, void *);
#define K_TRUE 1
#define __int64 long long
#define __NEW_STARLET
#include <vms/ieeedef.h>
#endif
struct feature {
const char *name;
int *gl_addr;
};
int __gl_heap_size = 64;
char __gl_float_format = 'I';
static const struct feature features[] =
{
{"GNAT$NO_MALLOC_64", &__gl_heap_size},
{0, 0}
};
void
__gnat_set_features (void)
{
int i;
char buff[16];
#ifdef IN_RTS
IEEE clrmsk, setmsk, prvmsk;
clrmsk.ieee$q_flags = 0LL;
setmsk.ieee$q_flags = 0LL;
#endif
for (i = 0; features[i].name; i++)
{
__gnat_vms_get_logical (features[i].name, buff, sizeof (buff));
if (strcmp (buff, "ENABLE") == 0
|| strcmp (buff, "TRUE") == 0
|| strcmp (buff, "1") == 0)
*features[i].gl_addr = 32;
else if (strcmp (buff, "DISABLE") == 0
|| strcmp (buff, "FALSE") == 0
|| strcmp (buff, "0") == 0)
*features[i].gl_addr = 64;
}
__gnat_set_stack_limit ();
#ifdef IN_RTS
if (__gl_float_format == 'V')
{
setmsk.ieee$v_trap_enable_inv = K_TRUE;
setmsk.ieee$v_trap_enable_dze = K_TRUE;
setmsk.ieee$v_trap_enable_ovf = K_TRUE;
SYS$IEEE_SET_FP_CONTROL (&clrmsk, &setmsk, &prvmsk);
}
#endif
__gnat_features_set = 1;
}
extern unsigned int LIB$GETSYI (int *, ...);
#define SYI$_VERSION 0x1000
int
__gnat_is_vms_v7 (void)
{
struct descriptor_s desc;
char version[8];
int status;
int code = SYI$_VERSION;
desc.len = sizeof (version);
desc.mbz = 0;
desc.adr = version;
status = LIB$GETSYI (&code, 0, &desc);
if ((status & 1) == 1 && version[1] == '7' && version[2] == '.')
return 1;
else
return 0;
}
#elif defined (__FreeBSD__) || defined (__DragonFly__)
#include <signal.h>
#include <sys/ucontext.h>
#include <unistd.h>
static void
__gnat_error_handler (int sig,
siginfo_t *si ATTRIBUTE_UNUSED,
void *ucontext ATTRIBUTE_UNUSED)
{
struct Exception_Data *exception;
const char *msg;
switch (sig)
{
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
case SIGILL:
exception = &constraint_error;
msg = "SIGILL";
break;
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &storage_error;
msg = "SIGBUS: possible stack overflow";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
void
__gnat_install_handler (void)
{
struct sigaction act;
act.sa_sigaction
= (void (*)(int, struct __siginfo *, void*)) __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_RESTART | SA_SIGINFO;
(void) sigemptyset (&act.sa_mask);
(void) sigaction (SIGILL,  &act, NULL);
(void) sigaction (SIGFPE,  &act, NULL);
(void) sigaction (SIGSEGV, &act, NULL);
(void) sigaction (SIGBUS,  &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined(__vxworks)
#include <signal.h>
#include <taskLib.h>
#if (defined (__i386__) || defined (__x86_64__)) && !defined (VTHREADS)
#include <sysLib.h>
#endif
#include "sigtramp.h"
#ifndef __RTP__
#include <intLib.h>
#include <iv.h>
#endif
#if ((defined (ARMEL) && (_WRS_VXWORKS_MAJOR == 6)) || defined (__x86_64__)) && !defined(__RTP__)
#define VXWORKS_FORCE_GUARD_PAGE 1
#include <vmLib.h>
extern size_t vxIntStackOverflowSize;
#define INT_OVERFLOW_SIZE vxIntStackOverflowSize
#endif
#ifdef VTHREADS
#include "private/vThreadsP.h"
#endif
#ifndef __RTP__
extern void * __gnat_inum_to_ivec (int);
void *
__gnat_inum_to_ivec (int num)
{
return (void *) INUM_TO_IVEC (num);
}
#endif
#if !defined(__alpha_vxworks) && ((_WRS_VXWORKS_MAJOR != 6) && (_WRS_VXWORKS_MAJOR != 7)) && !defined(__RTP__)
extern long getpid (void);
long
getpid (void)
{
return taskIdSelf ();
}
#endif
static int
__gnat_reset_guard_page (int sig)
{
#if defined (VXWORKS_FORCE_GUARD_PAGE)
if (sig != SIGSEGV && sig != SIGBUS && sig != SIGILL) return FALSE;
if (INT_OVERFLOW_SIZE == 0) return FALSE;
TASK_ID tid           = taskIdSelf ();
WIND_TCB *pTcb        = taskTcb (tid);
VIRT_ADDR guardPage   = (VIRT_ADDR) pTcb->pStackEnd - INT_OVERFLOW_SIZE;
UINT stateMask        = VM_STATE_MASK_VALID;
UINT guardState       = VM_STATE_VALID_NOT;
#if (_WRS_VXWORKS_MAJOR >= 7)
stateMask  |= MMU_ATTR_SPL_MSK;
guardState |= MMU_ATTR_NO_BLOCK;
#endif
UINT nState;
vmStateGet (NULL, guardPage, &nState);
if ((nState & VM_STATE_MASK_VALID) != VM_STATE_VALID_NOT)
{
vmStateSet (NULL, guardPage, INT_OVERFLOW_SIZE, stateMask, guardState);
return TRUE;
}
#endif 
return FALSE;
}
void
__gnat_clear_exception_count (void)
{
#ifdef VTHREADS
WIND_TCB *currentTask = (WIND_TCB *) taskIdSelf();
currentTask->vThreads.excCnt = 0;
#endif
}
void
__gnat_map_signal (int sig,
siginfo_t *si ATTRIBUTE_UNUSED,
void *sc ATTRIBUTE_UNUSED)
{
struct Exception_Data *exception;
const char *msg;
switch (sig)
{
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
#ifdef VTHREADS
#ifdef __VXWORKSMILS__
case SIGILL:
exception = &storage_error;
msg = "SIGILL: possible stack overflow";
break;
case SIGSEGV:
exception = &storage_error;
msg = "SIGSEGV";
break;
case SIGBUS:
exception = &program_error;
msg = "SIGBUS";
break;
#else
case SIGILL:
exception = &constraint_error;
msg = "Floating point exception or SIGILL";
break;
case SIGSEGV:
exception = &storage_error;
msg = "SIGSEGV";
break;
case SIGBUS:
exception = &storage_error;
msg = "SIGBUS: possible stack overflow";
break;
#endif
#elif (_WRS_VXWORKS_MAJOR >= 6)
case SIGILL:
exception = &constraint_error;
msg = "SIGILL";
break;
#ifdef __RTP__
case SIGSEGV:
exception = &storage_error;
msg = "SIGSEGV: possible stack overflow";
break;
case SIGBUS:
exception = &program_error;
msg = "SIGBUS";
break;
#else
case SIGSEGV:
exception = &storage_error;
msg = "SIGSEGV";
break;
case SIGBUS:
exception = &storage_error;
msg = "SIGBUS: possible stack overflow";
break;
#endif
#else
case SIGILL:
exception = &storage_error;
msg = "SIGILL: possible stack overflow";
break;
case SIGSEGV:
exception = &storage_error;
msg = "SIGSEGV";
break;
case SIGBUS:
exception = &program_error;
msg = "SIGBUS";
break;
#endif
default:
exception = &program_error;
msg = "unhandled signal";
}
if (__gnat_reset_guard_page (sig))
{
exception = &storage_error;
switch (sig)
{
case SIGSEGV:
msg = "SIGSEGV: stack overflow";
break;
case SIGBUS:
msg = "SIGBUS: stack overflow";
break;
case SIGILL:
msg = "SIGILL: stack overflow";
break;
}
}
__gnat_clear_exception_count ();
Raise_From_Signal_Handler (exception, msg);
}
#if defined (ARMEL) && (_WRS_VXWORKS_MAJOR >= 7) || defined (__aarch64__)
#define HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
#include <arch/../regs.h>
#ifndef __RTP__
#include <sigLib.h>
#else
#include <signal.h>
#include <regs.h>
#include <ucontext.h>
#endif 
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED,
void *sc ATTRIBUTE_UNUSED)
{
uintptr_t *pc_addr;
#ifdef __RTP__
mcontext_t *mcontext = &((ucontext_t *) sc)->uc_mcontext;
pc_addr = (uintptr_t*)&mcontext->regs.pc;
#else
struct sigcontext * sctx = (struct sigcontext *) sc;
pc_addr = (uintptr_t*)&sctx->sc_pregs->pc;
#endif
*pc_addr += 2;
}
#endif 
static void
__gnat_error_handler (int sig, siginfo_t *si, void *sc)
{
sigset_t mask;
#if !(defined (__RTP__) || defined (VTHREADS)) && ((CPU == PPCE500V2) || (CPU == PPC85XX))
register unsigned msr;
asm volatile ("mfmsr %0" : "=r" (msr));
if ((msr & 0x02000000) == 0)
{
msr |= 0x02000000;
asm volatile ("mtmsr %0" : : "r" (msr));
}
#endif
sigprocmask (SIG_SETMASK, NULL, &mask);
sigdelset (&mask, sig);
sigprocmask (SIG_SETMASK, &mask, NULL);
#if defined (__ARMEL__) || defined (__PPC__) || defined (__i386__) || defined (__x86_64__) || defined (__aarch64__)
#ifdef HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
__gnat_adjust_context_for_raise (sig, sc);
#endif
__gnat_sigtramp (sig, (void *)si, (void *)sc,
(__sigtramphandler_t *)&__gnat_map_signal);
#else
__gnat_map_signal (sig, si, sc);
#endif
}
#if defined(__leon__) && defined(_WRS_KERNEL)
extern void excEnt (void);
struct trap_entry {
unsigned long inst_first;
unsigned long inst_second;
unsigned long inst_third;
unsigned long inst_fourth;
};
struct trap_entry *trap_0_entry;
#endif
void
__gnat_install_handler (void)
{
struct sigaction act;
act.sa_sigaction = __gnat_error_handler;
act.sa_flags = SA_SIGINFO | SA_ONSTACK;
sigemptyset (&act.sa_mask);
sigaction (SIGFPE,  &act, NULL);
sigaction (SIGILL,  &act, NULL);
sigaction (SIGSEGV, &act, NULL);
sigaction (SIGBUS,  &act, NULL);
#if defined(__leon__) && defined(_WRS_KERNEL)
trap_0_entry = (struct trap_entry *)(intVecBaseGet () + 0x80 * 4);
trap_0_entry->inst_first = 0xae102000 + 9;
trap_0_entry->inst_second = 0x2d000000 + ((unsigned long)excEnt >> 10);
trap_0_entry->inst_third = 0x81c5a000 + ((unsigned long)excEnt & 0x3ff);
trap_0_entry->inst_fourth = 0xa1480000;
#endif
#ifdef __HANDLE_VXSIM_SC
{
char *model = sysModel ();
if ((strncmp (model, "Linux", 5) == 0)
|| (strncmp (model, "Windows", 7) == 0)
|| (strncmp (model, "SIMLINUX", 8) == 0) 
|| (strncmp (model, "SIMNT", 5) == 0)) 
__gnat_set_is_vxsim (TRUE);
}
#endif
__gnat_handler_installed = 1;
}
#define HAVE_GNAT_INIT_FLOAT
void
__gnat_init_float (void)
{
#if defined (_ARCH_PPC) && !defined (_SOFT_FLOAT) && (!defined (VTHREADS) || defined (__VXWORKSMILS__))
#if defined (__SPE__)
{
}
#else
asm ("mtfsb0 25");
asm ("mtfsb0 26");
#endif
#endif
#if (defined (__i386__) || defined (__x86_64__)) && !defined (VTHREADS)
asm ("finit");
#endif
#if defined (sparc64)
#define FSR_TEM_NVM (1 << 27)  
#define FSR_TEM_OFM (1 << 26)  
#define FSR_TEM_UFM (1 << 25)  
#define FSR_TEM_DZM (1 << 24)  
#define FSR_TEM_NXM (1 << 23)  
{
unsigned int fsr;
__asm__("st %%fsr, %0" : "=m" (fsr));
fsr &= ~(FSR_TEM_OFM | FSR_TEM_UFM);
__asm__("ld %0, %%fsr" : : "m" (fsr));
}
#endif
}
void (*__gnat_set_stack_limit_hook)(void) = (void (*)(void))0;
#elif defined(__NetBSD__)
#include <signal.h>
#include <unistd.h>
static void
__gnat_error_handler (int sig)
{
struct Exception_Data *exception;
const char *msg;
switch(sig)
{
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
case SIGILL:
exception = &constraint_error;
msg = "SIGILL";
break;
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &constraint_error;
msg = "SIGBUS";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
void
__gnat_install_handler (void)
{
struct sigaction act;
act.sa_handler = __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_RESTART;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGILL) != 's')
sigaction (SIGILL,  &act, NULL);
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
sigaction (SIGSEGV, &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined(__OpenBSD__)
#include <signal.h>
#include <unistd.h>
static void
__gnat_error_handler (int sig)
{
struct Exception_Data *exception;
const char *msg;
switch(sig)
{
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
case SIGILL:
exception = &constraint_error;
msg = "SIGILL";
break;
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &constraint_error;
msg = "SIGBUS";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
void
__gnat_install_handler (void)
{
struct sigaction act;
act.sa_handler = __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_RESTART;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGILL) != 's')
sigaction (SIGILL,  &act, NULL);
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
sigaction (SIGSEGV, &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/sysctl.h>
char __gnat_alternate_stack[32 * 1024]; 
#define	UC_RESET_ALT_STACK	0x80000000
#if !(defined (__arm__) || defined (__arm64__) || TARGET_IPHONE_SIMULATOR)
#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_statistics.h>
#endif
#ifdef __arm64__
#include <sys/ucontext.h>
#include "sigtramp.h"
#endif
static int
__gnat_is_stack_guard (mach_vm_address_t addr)
{
#if !(defined (__arm__) || defined (__arm64__) || TARGET_IPHONE_SIMULATOR)
kern_return_t kret;
vm_region_submap_info_data_64_t info;
mach_vm_address_t start;
mach_vm_size_t size;
natural_t depth;
mach_msg_type_number_t count;
count = VM_REGION_SUBMAP_INFO_COUNT_64;
start = addr;
size = -1;
depth = 9999;
kret = mach_vm_region_recurse (mach_task_self (), &start, &size, &depth,
(vm_region_recurse_info_t) &info, &count);
if (kret == KERN_SUCCESS
&& addr >= start && addr < (start + size)
&& info.protection == VM_PROT_NONE
&& info.user_tag == VM_MEMORY_STACK)
return 1;
return 0;
#else
return addr >= 4096;
#endif
}
#define HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
#if defined (__x86_64__)
static int
__darwin_major_version (void)
{
static int cache = -1;
if (cache < 0)
{
int mib[2] = {CTL_KERN, KERN_OSRELEASE};
size_t len;
if (sysctl (mib, 2, NULL, &len, NULL, 0) == 0)
{
char release[len];
sysctl (mib, 2, release, &len, NULL, 0);
cache = (int) strtol (release, NULL, 10);
}
else
{
cache = 0;
}
}
return cache;
}
#endif
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED,
void *ucontext ATTRIBUTE_UNUSED)
{
#if defined (__x86_64__)
if (__darwin_major_version () < 12)
{
ucontext_t *uc = (ucontext_t *)ucontext;
unsigned long t = uc->uc_mcontext->__ss.__rbx;
uc->uc_mcontext->__ss.__rbx = uc->uc_mcontext->__ss.__rdx;
uc->uc_mcontext->__ss.__rdx = t;
}
#elif defined(__arm64__)
ucontext_t *uc = (ucontext_t *)ucontext;
uc->uc_mcontext->__ss.__pc++;
#endif
}
static void
__gnat_map_signal (int sig, siginfo_t *si, void *mcontext ATTRIBUTE_UNUSED)
{
struct Exception_Data *exception;
const char *msg;
switch (sig)
{
case SIGSEGV:
case SIGBUS:
if (__gnat_is_stack_guard ((unsigned long)si->si_addr))
{
#ifdef __arm64__
mcontext_t mc = (mcontext_t)mcontext;
if (!(*(unsigned int *)(mc->__ss.__pc-1) & ((unsigned int)1 << 30)))
mc->__ss.__pc = mc->__ss.__lr;
#endif
exception = &storage_error;
msg = "stack overflow";
}
else
{
exception = &constraint_error;
msg = "erroneous memory access";
}
syscall (SYS_sigreturn, NULL, UC_RESET_ALT_STACK);
break;
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
static void
__gnat_error_handler (int sig, siginfo_t *si, void *ucontext)
{
__gnat_adjust_context_for_raise (sig, ucontext);
#ifdef __arm64__
__gnat_sigtramp (sig, (void *)si, ucontext,
(__sigtramphandler_t *)&__gnat_map_signal);
#else
__gnat_map_signal (sig, si, ucontext);
#endif
}
void
__gnat_install_handler (void)
{
struct sigaction act;
stack_t stack;
stack.ss_sp = __gnat_alternate_stack;
stack.ss_size = sizeof (__gnat_alternate_stack);
stack.ss_flags = 0;
sigaltstack (&stack, NULL);
act.sa_flags = SA_NODEFER | SA_RESTART | SA_SIGINFO;
act.sa_sigaction = __gnat_error_handler;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGABRT) != 's')
sigaction (SIGABRT, &act, NULL);
if (__gnat_get_interrupt_state (SIGFPE) != 's')
sigaction (SIGFPE,  &act, NULL);
if (__gnat_get_interrupt_state (SIGILL) != 's')
sigaction (SIGILL,  &act, NULL);
act.sa_flags |= SA_ONSTACK;
if (__gnat_get_interrupt_state (SIGSEGV) != 's')
sigaction (SIGSEGV, &act, NULL);
if (__gnat_get_interrupt_state (SIGBUS) != 's')
sigaction (SIGBUS,  &act, NULL);
__gnat_handler_installed = 1;
}
#elif defined(__QNX__)
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include "sigtramp.h"
void
__gnat_map_signal (int sig,
siginfo_t *si ATTRIBUTE_UNUSED,
void *mcontext ATTRIBUTE_UNUSED)
{
struct Exception_Data *exception;
const char *msg;
switch(sig)
{
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
case SIGILL:
exception = &constraint_error;
msg = "SIGILL";
break;
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &constraint_error;
msg = "SIGBUS";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
static void
__gnat_error_handler (int sig, siginfo_t *si, void *ucontext)
{
__gnat_sigtramp (sig, (void *) si, (void *) ucontext,
(__sigtramphandler_t *)&__gnat_map_signal);
}
char __gnat_alternate_stack[0];
void
__gnat_install_handler (void)
{
struct sigaction act;
int err;
act.sa_handler = __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_SIGINFO;
sigemptyset (&act.sa_mask);
if (__gnat_get_interrupt_state (SIGFPE) != 's') {
err = sigaction (SIGFPE,  &act, NULL);
if (err == -1) {
err = errno;
perror ("error while attaching SIGFPE");
perror (strerror (err));
}
}
if (__gnat_get_interrupt_state (SIGILL) != 's') {
sigaction (SIGILL,  &act, NULL);
if (err == -1) {
err = errno;
perror ("error while attaching SIGFPE");
perror (strerror (err));
}
}
if (__gnat_get_interrupt_state (SIGSEGV) != 's') {
sigaction (SIGSEGV, &act, NULL);
if (err == -1) {
err = errno;
perror ("error while attaching SIGFPE");
perror (strerror (err));
}
}
if (__gnat_get_interrupt_state (SIGBUS) != 's') {
sigaction (SIGBUS,  &act, NULL);
if (err == -1) {
err = errno;
perror ("error while attaching SIGFPE");
perror (strerror (err));
}
}
__gnat_handler_installed = 1;
}
#elif defined (__DJGPP__)
void
__gnat_install_handler ()
{
__gnat_handler_installed = 1;
}
#elif defined(__ANDROID__)
#include <signal.h>
#include <sys/ucontext.h>
#include "sigtramp.h"
#define HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED, void *ucontext)
{
mcontext_t *mcontext = &((ucontext_t *) ucontext)->uc_mcontext;
((mcontext_t *) mcontext)->arm_pc += 2;
}
static void
__gnat_map_signal (int sig,
siginfo_t *si ATTRIBUTE_UNUSED,
void *mcontext ATTRIBUTE_UNUSED)
{
struct Exception_Data *exception;
const char *msg;
switch (sig)
{
case SIGSEGV:
exception = &storage_error;
msg = "stack overflow or erroneous memory access";
break;
case SIGBUS:
exception = &constraint_error;
msg = "SIGBUS";
break;
case SIGFPE:
exception = &constraint_error;
msg = "SIGFPE";
break;
default:
exception = &program_error;
msg = "unhandled signal";
}
Raise_From_Signal_Handler (exception, msg);
}
static void
__gnat_error_handler (int sig, siginfo_t *si, void *ucontext)
{
__gnat_adjust_context_for_raise (sig, ucontext);
__gnat_sigtramp (sig, (void *) si, (void *) ucontext,
(__sigtramphandler_t *)&__gnat_map_signal);
}
char __gnat_alternate_stack[16 * 1024];
void
__gnat_install_handler (void)
{
struct sigaction act;
stack_t stack;
stack.ss_sp = __gnat_alternate_stack;
stack.ss_size = sizeof (__gnat_alternate_stack);
stack.ss_flags = 0;
sigaltstack (&stack, NULL);
act.sa_sigaction = __gnat_error_handler;
act.sa_flags = SA_NODEFER | SA_RESTART | SA_SIGINFO;
sigemptyset (&act.sa_mask);
sigaction (SIGABRT, &act, NULL);
sigaction (SIGFPE,  &act, NULL);
sigaction (SIGILL,  &act, NULL);
sigaction (SIGBUS,  &act, NULL);
act.sa_flags |= SA_ONSTACK;
sigaction (SIGSEGV, &act, NULL);
__gnat_handler_installed = 1;
}
#else
void
__gnat_install_handler (void)
{
__gnat_handler_installed = 1;
}
#endif
#if defined (_WIN32) || defined (__INTERIX) \
|| defined (__Lynx__) || defined(__NetBSD__) || defined(__FreeBSD__) \
|| defined (__OpenBSD__) || defined (__DragonFly__) || defined(__QNX__)
#define HAVE_GNAT_INIT_FLOAT
void
__gnat_init_float (void)
{
#if defined (__i386__) || defined (__x86_64__)
asm ("finit");
#endif  
}
#endif
#ifndef HAVE_GNAT_INIT_FLOAT
void
__gnat_init_float (void)
{
}
#endif
#ifndef HAVE_GNAT_ADJUST_CONTEXT_FOR_RAISE
void
__gnat_adjust_context_for_raise (int signo ATTRIBUTE_UNUSED,
void *ucontext ATTRIBUTE_UNUSED)
{
}
#endif
#ifdef __cplusplus
}
#endif
