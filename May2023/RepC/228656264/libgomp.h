#ifndef LIBGOMP_H 
#define LIBGOMP_H 1
#ifndef _LIBGOMP_CHECKING_
#define _LIBGOMP_CHECKING_ 0
#endif
#ifndef _LIBGOMP_TEAM_TIMING_
#define _LIBGOMP_TEAM_TIMING_ 1
#endif
#ifndef _LIBGOMP_TASK_TIMING_
#define _LIBGOMP_TASK_TIMING_ 1
#endif
#ifndef _LIBGOMP_TASK_GRANULARITY_
#define _LIBGOMP_TASK_GRANULARITY_ 0
#endif
#ifndef _LIBGOMP_TASK_SWITCH_AUDITING_
#define _LIBGOMP_TASK_SWITCH_AUDITING_ 1
#define _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_ 0
#endif
#ifndef _LIBGOMP_LIBGOMP_TIMING_
#define _LIBGOMP_LIBGOMP_TIMING_ 1
#endif
#ifndef _LIBGOMP_TEAM_LOCK_TIMING_
#define _LIBGOMP_TEAM_LOCK_TIMING_ 1
#endif
#define DEFAULT_NUM_PRIORITIES 16
#include "config.h"
#include "gstdint.h"
#include "libgomp-plugin.h"
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif
#include <stdbool.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#if _LIBGOMP_TEAM_TIMING_ || _LIBGOMP_TASK_TIMING_ || _LIBGOMP_TASK_GRANULARITY_ || \
_LIBGOMP_TASK_SWITCH_AUDITING_ || _LIBGOMP_LIBGOMP_TIMING_ || _LIBGOMP_TEAM_LOCK_TIMING_
#include <stdio.h>
#include <time.h>
#endif
#if _LIBGOMP_CHECKING_
# ifdef STRING_WITH_STRINGS
#  include <string.h>
#  include <strings.h>
# else
#  ifdef HAVE_STRING_H
#   include <string.h>
#  else
#   ifdef HAVE_STRINGS_H
#    include <strings.h>
#   endif
#  endif
# endif
#endif
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility push(hidden)
#endif
enum memmodel
{
MEMMODEL_RELAXED = 0,
MEMMODEL_CONSUME = 1,
MEMMODEL_ACQUIRE = 2,
MEMMODEL_RELEASE = 3,
MEMMODEL_ACQ_REL = 4,
MEMMODEL_SEQ_CST = 5
};
extern void *gomp_malloc (size_t) __attribute__((malloc));
extern void *gomp_malloc_cleared (size_t) __attribute__((malloc));
extern void *gomp_aligned_alloc (size_t, size_t) __attribute__((malloc));
extern void *gomp_realloc (void *, size_t);
#define gomp_alloca(x)  __builtin_alloca(x)
extern void gomp_vdebug (int, const char *, va_list);
extern void gomp_debug (int, const char *, ...)
__attribute__ ((format (printf, 2, 3)));
#define gomp_vdebug(KIND, FMT, VALIST) \
do { \
if (__builtin_expect (gomp_debug_var, 0)) \
(gomp_vdebug) ((KIND), (FMT), (VALIST)); \
} while (0)
#define gomp_debug(KIND, ...) \
do { \
if (__builtin_expect (gomp_debug_var, 0)) \
(gomp_debug) ((KIND), __VA_ARGS__); \
} while (0)
extern void gomp_verror (const char *, va_list);
extern void gomp_error (const char *, ...)
__attribute__ ((format (printf, 1, 2)));
extern void gomp_vfatal (const char *, va_list)
__attribute__ ((noreturn));
extern void gomp_fatal (const char *, ...)
__attribute__ ((noreturn, format (printf, 1, 2)));
enum gomp_task_type
{
GOMP_TASK_TYPE_TIED,
GOMP_TASK_TYPE_UNTIED,
GOMP_TASK_TYPE_ENUM_SIZE
};
enum gomp_task_kind
{
GOMP_TASK_IMPLICIT,
GOMP_TASK_UNDEFERRED,
GOMP_TASK_WAITING,
GOMP_TASK_TIED,
GOMP_TASK_TIED_SUSPENDED,
GOMP_TASK_ASYNC_RUNNING,
GOMP_TASK_KIND_ENUM_SIZE
#if defined HAVE_TLS || defined USE_EMUTLS
, GOMP_TASK_KIND_ENUM_PAD = 0x7fffffff
#endif
};
struct gomp_thread;
struct gomp_task_icv;
struct gomp_task;
struct gomp_taskgroup;
struct htab;
#include "priority_queue.h"
#include "sem.h"
#include "mutex.h"
#include "bar.h"
#include "simple-bar.h"
#include "ptrlock.h"
#include "spinlock.h"
#include "context.h"
#include "state.h"
enum gomp_schedule_type
{
GFS_RUNTIME,
GFS_STATIC,
GFS_DYNAMIC,
GFS_GUIDED,
GFS_AUTO
};
struct gomp_doacross_work_share
{
union {
long chunk_size;
unsigned long long chunk_size_ull;
long q;
unsigned long long q_ull;
};
unsigned long elt_sz;
unsigned int ncounts;
bool flattened;
unsigned char *array;
long t;
union {
long boundary;
unsigned long long boundary_ull;
};
unsigned int shift_counts[];
};
struct gomp_work_share
{
enum gomp_schedule_type sched;
int mode;
union {
struct {
long chunk_size;
long end;
long incr;
};
struct {
unsigned long long chunk_size_ull;
unsigned long long end_ull;
unsigned long long incr_ull;
};
};
union {
unsigned *ordered_team_ids;
struct gomp_doacross_work_share *doacross;
};
unsigned ordered_num_used;
unsigned ordered_owner;
unsigned ordered_cur;
struct gomp_work_share *next_alloc;
gomp_mutex_t lock __attribute__((aligned (64)));
unsigned threads_completed;
union {
long next;
unsigned long long next_ull;
void *copyprivate;
};
union {
gomp_ptrlock_t next_ws;
struct gomp_work_share *next_free;
};
unsigned inline_ordered_team_ids[0];
};
struct gomp_team_state
{
struct gomp_team *team;
struct gomp_work_share *work_share;
struct gomp_work_share *last_work_share;
unsigned team_id;
#if defined HAVE_TLS || defined USE_EMUTLS
int core_id;
void * alt_stack;
unsigned long alt_stack_size;
int ibs_fd;
int ipi_fd;
#endif
unsigned level;
unsigned active_level;
unsigned place_partition_off;
unsigned place_partition_len;
#ifdef HAVE_SYNC_BUILTINS
unsigned long single_count;
#endif
unsigned long static_trip;
};
struct target_mem_desc;
struct gomp_task_icv
{
unsigned long nthreads_var;
enum gomp_schedule_type run_sched_var;
int run_sched_chunk_size;
int default_device_var;
unsigned int thread_limit_var;
bool dyn_var;
bool nest_var;
bool wf_sched_var;
bool untied_block_var;
bool ult_var;
unsigned long ult_stack_size;
#if defined HAVE_TLS || defined USE_EMUTLS
void *text_section_start;
void *text_section_end;
#endif
char bind_var;
struct target_mem_desc *target_data;
};
#if defined HAVE_TLS || defined USE_EMUTLS
struct gomp_struct_fields_offset
{
size_t in_thread_team_offset;
size_t in_thread_preemptable_offset;
size_t in_thread_task_offset;
size_t in_task_kind_offset;
size_t in_task_state_offset;
size_t in_state_context_offset;
} __attribute__((packed,aligned(8)));
extern struct gomp_struct_fields_offset gomp_global_sfo;
#endif
extern struct gomp_task_icv gomp_global_icv;
#ifndef HAVE_SYNC_BUILTINS
extern gomp_mutex_t gomp_managed_threads_lock;
#endif
extern unsigned long gomp_max_active_levels_var;
extern bool gomp_cancel_var;
extern bool gomp_auto_cutoff_var;
extern bool gomp_signal_unblock;
extern bool gomp_ipi_var;
extern double gomp_ipi_decision_model;
extern unsigned long gomp_ipi_priority_gap;
extern unsigned long gomp_ipi_sending_cap;
extern unsigned long gomp_ibs_rate_var;
extern unsigned long gomp_queue_policy_var;
extern int gomp_max_task_priority_var;
extern unsigned long long gomp_spin_count_var, gomp_throttled_spin_count_var;
extern unsigned long gomp_available_cpus, gomp_managed_threads;
extern unsigned long *gomp_nthreads_var_list, gomp_nthreads_var_list_len;
extern char *gomp_bind_var_list;
extern unsigned long gomp_bind_var_list_len;
extern void **gomp_places_list;
extern unsigned long gomp_places_list_len;
extern unsigned int gomp_num_teams_var;
extern int gomp_debug_var;
extern int goacc_device_num;
extern char *goacc_device_type;
#if _LIBGOMP_TEAM_TIMING_
#include "team-timing.h"
#endif
#include "task-timing.h"
#if _LIBGOMP_TASK_GRANULARITY_
#include "task-granularity.h"
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
#include "libgomp-timing.h"
#endif
#if _LIBGOMP_TEAM_LOCK_TIMING_
#include "team-lock-timing.h"
#endif
struct gomp_task_depend_entry
{
void *addr;
struct gomp_task_depend_entry *next;
struct gomp_task_depend_entry *prev;
struct gomp_task *task;
bool is_in;
bool redundant;
bool redundant_out;
};
struct gomp_dependers_vec
{
size_t n_elem;
size_t allocated;
struct gomp_task *elem[];
};
struct gomp_taskwait
{
bool in_taskwait;
bool in_depend_wait;
size_t n_depend;
gomp_sem_t taskwait_sem;
};
struct gomp_task
{
struct gomp_task *parent;
struct gomp_task *next_tied_task;
struct gomp_task *previous_tied_task;
struct gomp_task *ascending_tied_task;
struct gomp_task *undeferred_ancestor;
struct priority_queue tied_children_queue;
struct priority_queue untied_children_queue;
struct gomp_taskgroup *taskgroup;
struct gomp_dependers_vec *dependers;
struct htab *depend_hash;
struct gomp_taskwait *taskwait;
size_t depend_count;
size_t num_dependees;
int priority;
uint64_t creation_time;
uint64_t completion_time;
#if _LIBGOMP_TASK_GRANULARITY_
uint64_t stage_start;
uint64_t stage_end;
uint64_t sum_stages;
#endif
struct gomp_thread *suspending_thread;
struct priority_node pnode[9];
struct gomp_task_state *state;
struct gomp_task_icv icv;
void (*fn) (void *);
void *fn_data;
enum gomp_task_type type;
enum gomp_task_kind kind;
bool in_tied_task;
bool is_blocked;
bool final_task;
bool copy_ctors_done;
bool parent_depends_on;
struct gomp_task_depend_entry depend[];
};
#if _LIBGOMP_TASK_SWITCH_AUDITING_
#include "task-switch-auditing.h"
#endif
struct gomp_taskgroup
{
struct gomp_taskgroup *prev;
struct priority_queue tied_taskgroup_queue;
struct priority_queue untied_taskgroup_queue;
bool in_taskgroup_wait;
bool cancelled;
gomp_sem_t taskgroup_sem;
size_t num_children;
};
enum gomp_target_task_state
{
GOMP_TARGET_TASK_DATA,
GOMP_TARGET_TASK_BEFORE_MAP,
GOMP_TARGET_TASK_FALLBACK,
GOMP_TARGET_TASK_READY_TO_RUN,
GOMP_TARGET_TASK_RUNNING,
GOMP_TARGET_TASK_FINISHED
};
struct gomp_target_task
{
struct gomp_device_descr *devicep;
void (*fn) (void *);
size_t mapnum;
size_t *sizes;
unsigned short *kinds;
unsigned int flags;
enum gomp_target_task_state state;
struct target_mem_desc *tgt;
struct gomp_task *task;
struct gomp_team *team;
void **args;
void *hostaddrs[];
};
struct gomp_team
{
unsigned nthreads;
unsigned work_share_chunk;
struct gomp_team_state prev_ts;
gomp_sem_t master_release;
gomp_sem_t **ordered_release;
#if _LIBGOMP_TEAM_TIMING_
struct gomp_team_time team_time;
#endif
struct gomp_task_time_table **prio_task_time;
#if _LIBGOMP_TASK_GRANULARITY_
struct gomp_task_granularity_table **task_granularity_table;
#endif
#if _LIBGOMP_TASK_SWITCH_AUDITING_
struct gomp_task_switch_audit **task_switch_audit;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
struct gomp_libgomp_time **libgomp_time;
#endif
#if _LIBGOMP_TEAM_LOCK_TIMING_
struct gomp_team_lock_time **team_lock_time;
#endif
struct gomp_work_share *work_shares_to_free;
struct gomp_work_share *work_share_list_alloc;
struct gomp_work_share *work_share_list_free;
#ifdef HAVE_SYNC_BUILTINS
unsigned long single_count;
#else
gomp_mutex_t work_share_list_free_lock;
#endif
gomp_barrier_t barrier;
struct gomp_work_share work_shares[8];
gomp_mutex_t task_lock;
struct priority_queue tied_task_queue;
struct priority_queue untied_task_queue;
unsigned int task_count;
unsigned int task_queued_count;
unsigned int task_running_count;
int work_share_cancelled;
int team_cancelled;
struct gomp_task_state_list_group task_state_pool;
struct gomp_task implicit_task[];
};
struct gomp_thread
{
void (*fn) (void *data);
void *data;
struct gomp_team_state ts;
struct gomp_task *task;
struct gomp_task *last_tied_task;
struct gomp_task_state *cached_state;
struct priority_queue tied_suspended;
struct priority_queue untied_suspended;
bool hold_team_lock;
#if defined HAVE_TLS || defined USE_EMUTLS
bool in_libgomp;
#endif
unsigned int non_preemptable;
gomp_sem_t release;
unsigned int place;
struct gomp_task_time_table *prio_task_time;
#if _LIBGOMP_TASK_GRANULARITY_
struct gomp_task_granularity_table *task_granularity_table;
#endif
#if _LIBGOMP_TASK_SWITCH_AUDITING_
struct gomp_task_switch_audit *task_switch_audit;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
struct gomp_libgomp_time *libgomp_time;
#endif
#if _LIBGOMP_TEAM_LOCK_TIMING_
struct gomp_team_lock_time *team_lock_time;
#endif
struct gomp_task_state_list *local_task_state_list;
struct gomp_task_state_list_group *global_task_state_group;
struct gomp_thread_pool *thread_pool;
};
struct gomp_thread_pool
{
struct gomp_thread **threads;
unsigned threads_size;
unsigned threads_used;
struct gomp_team *last_team;
unsigned long threads_busy;
gomp_simple_barrier_t threads_dock;
};
enum gomp_cancel_kind
{
GOMP_CANCEL_PARALLEL = 1,
GOMP_CANCEL_LOOP = 2,
GOMP_CANCEL_FOR = GOMP_CANCEL_LOOP,
GOMP_CANCEL_DO = GOMP_CANCEL_LOOP,
GOMP_CANCEL_SECTIONS = 4,
GOMP_CANCEL_TASKGROUP = 8
};
#if defined __nvptx__
extern struct gomp_thread *nvptx_thrs __attribute__((shared));
static inline struct gomp_thread *gomp_thread (void)
{
int tid;
asm ("mov.u32 %0, %%tid.y;" : "=r" (tid));
return nvptx_thrs + tid;
}
#elif defined HAVE_TLS || defined USE_EMUTLS
extern __thread struct gomp_thread *gomp_tls_ptr;
extern __thread struct gomp_thread gomp_tls_data;
static inline struct gomp_thread *gomp_thread (void)
{
return &gomp_tls_data;
}
#else
extern pthread_key_t gomp_tls_key;
static inline struct gomp_thread *gomp_thread (void)
{
return pthread_getspecific (gomp_tls_key);
}
#endif
extern struct gomp_task_icv *gomp_new_icv (void);
static inline struct gomp_task_icv *gomp_icv (bool write)
{
struct gomp_task *task = gomp_thread ()->task;
if (task)
return &task->icv;
else if (write)
return gomp_new_icv ();
else
return &gomp_global_icv;
}
#ifdef LIBGOMP_USE_PTHREADS
extern pthread_attr_t gomp_thread_attr;
extern pthread_key_t gomp_thread_destructor;
#endif
static inline __attribute__((always_inline)) unsigned int
gomp_count_1_bits(unsigned long long int i)
{
i = i - ((i >> 1) & 0x5555555555555555ULL);
i = (i & 0x3333333333333333ULL) + ((i >> 2) & 0x3333333333333333ULL);
i = (i + (i >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
i = i + (i >> 8);
i = i + (i >> 16);
i = i + (i >> 32);
return (unsigned int) (i & 0x7f);
}
#if defined HAVE_TLS || defined USE_EMUTLS
#ifndef RDTSC_32_LSB
#define RDTSC_32_LSB() ({ \
unsigned int cycles_low; \
asm volatile ( \
"RDTSC\n\t" \
"mov %%eax, %0\n\t" \
: \
"=r" (cycles_low) \
: \
: \
"%rax", "%rdx" \
); \
cycles_low; \
})
#endif
#ifndef RDTSC
#define RDTSC() ({ \
unsigned int cycles_low; \
unsigned int cycles_high; \
asm volatile ( \
"RDTSC\n\t" \
"mov %%edx, %0\n\t" \
"mov %%eax, %1\n\t" \
: \
"=r" (cycles_high), "=r" (cycles_low) \
: \
: \
"%rax", "%rdx" \
); \
(((uint64_t) cycles_high << 32) | cycles_low); \
})
#endif
#define RESIDUAL_TIME_THRESHOLD   44000
extern __thread unsigned long long ipi_mask;
static inline __attribute__((always_inline))
int ipi_syscall(unsigned long long cpus_mask)
{
int ret = 0;
asm volatile
(
"syscall"
: "=a" (ret)
: "0"(134), "D"(cpus_mask)
: "rcx", "r11", "memory"
);
return ret;
}
static inline __attribute__((always_inline))
int gomp_send_ipi(void)
{
int ret = 0;
if (ipi_mask)
{
ret = ipi_syscall(ipi_mask);
ipi_mask = 0ULL;
}
return ret;
}
static inline unsigned long long
get_mask_of_threads_with_lower_priority_tasks(struct gomp_thread *thr, struct gomp_team *team,
struct gomp_thread_pool *pool, int priority, bool unblocked_task)
{
unsigned int i, i_start, i_circ;
unsigned int set_cnt = gomp_count_1_bits(ipi_mask);
unsigned long long mask = ipi_mask;
if (pool->threads != NULL && set_cnt < gomp_ipi_sending_cap)
{
i_start = RDTSC_32_LSB() % pool->threads_used;
for (i=0; i<pool->threads_used; i++)
{
i_circ = (i + i_start) % pool->threads_used;
if (pool->threads[i_circ] == NULL)
continue;
if (pool->threads[i_circ] == thr)
continue;
if (pool->threads[i_circ]->ts.team != team)
continue;
if ((1ULL << pool->threads[i_circ]->ts.core_id) & mask)
continue;
if (pool->threads[i_circ]->in_libgomp)
continue;
if (pool->threads[i_circ]->task == NULL)
continue;
if (pool->threads[i_circ]->task->kind == GOMP_TASK_IMPLICIT)
continue;
if (gomp_signal_unblock && unblocked_task)
{
if ((pool->threads[i_circ]->task->priority + gomp_ipi_priority_gap) > priority)
continue;
}
else
{
if ((pool->threads[i_circ]->task->priority + gomp_ipi_priority_gap) >= priority)
continue;
}
if (gomp_ipi_decision_model > 0.0)
{
if (((signed long long) gomp_get_task_time(pool->threads[i_circ]->prio_task_time, pool->threads[i_circ]->task->fn, \
pool->threads[i_circ]->task->kind, pool->threads[i_circ]->task->type, pool->threads[i_circ]->task->priority) - \
(signed long long) (RDTSC() - pool->threads[i_circ]->task->creation_time)) < RESIDUAL_TIME_THRESHOLD)
continue;
}
set_cnt = set_cnt + 1;
mask = mask | (1ULL << pool->threads[i_circ]->ts.core_id);
if (set_cnt >= gomp_ipi_sending_cap)
break;
}
}
return mask;
}
static inline unsigned long long
get_mask_single_thread_with_lower_priority_tasks(struct gomp_thread *thr, int priority, bool unblocked_task)
{
unsigned int set_cnt = gomp_count_1_bits(ipi_mask);
unsigned long long mask = ipi_mask;
if (set_cnt < gomp_ipi_sending_cap)
{
do
{
if (thr == NULL)
break;
if ((1ULL << thr->ts.core_id) & mask)
break;
if (thr->in_libgomp)
break;
if (thr->task == NULL)
break;
if (thr->task->kind == GOMP_TASK_IMPLICIT)
break;
if (gomp_signal_unblock && unblocked_task)
{
if ((thr->task->priority + gomp_ipi_priority_gap) > priority)
break;
}
else
{
if ((thr->task->priority + gomp_ipi_priority_gap) >= priority)
break;
}
if (gomp_ipi_decision_model > 0.0)
{
if (((signed long long) gomp_get_task_time(thr->prio_task_time, thr->task->fn, thr->task->kind, thr->task->type, thr->task->priority) - \
(signed long long) (RDTSC() - thr->task->creation_time)) < RESIDUAL_TIME_THRESHOLD)
break;
}
set_cnt = set_cnt + 1;
mask = mask | (1ULL << thr->ts.core_id);
}
while (0);
}
return mask;
}
#endif
extern void gomp_init_affinity (void);
#ifdef LIBGOMP_USE_PTHREADS
extern void gomp_init_thread_affinity (pthread_attr_t *, unsigned int);
#endif
extern void **gomp_affinity_alloc (unsigned long, bool);
extern void gomp_affinity_init_place (void *);
extern bool gomp_affinity_add_cpus (void *, unsigned long, unsigned long,
long, bool);
extern bool gomp_affinity_remove_cpu (void *, unsigned long);
extern bool gomp_affinity_copy_place (void *, void *, long);
extern bool gomp_affinity_same_place (void *, void *);
extern bool gomp_affinity_finalize_place_list (bool);
extern bool gomp_affinity_init_level (int, unsigned long, bool);
extern void gomp_affinity_print_place (void *);
extern void gomp_get_place_proc_ids_8 (int, int64_t *);
extern int gomp_iter_static_next (long *, long *);
extern bool gomp_iter_dynamic_next_locked (long *, long *);
extern bool gomp_iter_guided_next_locked (long *, long *);
#ifdef HAVE_SYNC_BUILTINS
extern bool gomp_iter_dynamic_next (long *, long *);
extern bool gomp_iter_guided_next (long *, long *);
#endif
extern int gomp_iter_ull_static_next (unsigned long long *,
unsigned long long *);
extern bool gomp_iter_ull_dynamic_next_locked (unsigned long long *,
unsigned long long *);
extern bool gomp_iter_ull_guided_next_locked (unsigned long long *,
unsigned long long *);
#if defined HAVE_SYNC_BUILTINS && defined __LP64__
extern bool gomp_iter_ull_dynamic_next (unsigned long long *,
unsigned long long *);
extern bool gomp_iter_ull_guided_next (unsigned long long *,
unsigned long long *);
#endif
extern void gomp_ordered_first (void);
extern void gomp_ordered_last (void);
extern void gomp_ordered_next (void);
extern void gomp_ordered_static_init (void);
extern void gomp_ordered_static_next (void);
extern void gomp_ordered_sync (void);
extern void gomp_doacross_init (unsigned, long *, long);
extern void gomp_doacross_ull_init (unsigned, unsigned long long *,
unsigned long long);
extern unsigned gomp_resolve_num_threads (unsigned, unsigned);
extern void gomp_init_num_threads (void);
extern unsigned gomp_dynamic_max_threads (void);
extern void gomp_interrupt_trampoline (void);
extern int gomp_thread_interrupt_registration (void);
extern void gomp_thread_interrupt_cancellation (void);
extern void gomp_init_task (struct gomp_task *, struct gomp_task *,
struct gomp_task_icv *);
extern void gomp_end_task (void);
extern void gomp_barrier_handle_tasks (gomp_barrier_state_t);
extern void gomp_task_handle_locking (struct priority_queue *);
extern int gomp_locked_task_switch (struct priority_queue *, gomp_mutex_t *);
extern int gomp_undeferred_task_switch (void);
extern int gomp_blocked_task_switch (void);
#if defined HAVE_TLS || defined USE_EMUTLS
extern void gomp_interrupt_task_scheduling_pre (void);
extern void gomp_interrupt_task_scheduling_post (void);
#endif
extern void gomp_task_maybe_wait_for_dependencies (void **);
extern bool gomp_create_target_task (struct gomp_device_descr *,
void (*) (void *), size_t, void **,
size_t *, unsigned short *, unsigned int,
void **, void **,
enum gomp_target_task_state);
static void inline
gomp_finish_task (struct gomp_task *task)
{
if (__builtin_expect (task->depend_hash != NULL, 0))
free (task->depend_hash);
}
extern struct gomp_team *gomp_new_team (unsigned);
extern void gomp_team_start (void (*) (void *), void *, unsigned,
unsigned, struct gomp_team *);
extern void gomp_team_end (void);
extern void gomp_free_thread (void *);
extern void gomp_init_targets_once (void);
extern int gomp_get_num_devices (void);
extern bool gomp_target_task_fn (void *);
typedef struct splay_tree_node_s *splay_tree_node;
typedef struct splay_tree_s *splay_tree;
typedef struct splay_tree_key_s *splay_tree_key;
struct target_var_desc {
splay_tree_key key;
bool copy_from;
bool always_copy_from;
uintptr_t offset;
uintptr_t length;
};
struct target_mem_desc {
uintptr_t refcount;
splay_tree_node array;
uintptr_t tgt_start;
uintptr_t tgt_end;
void *to_free;
struct target_mem_desc *prev;
size_t list_count;
struct gomp_device_descr *device_descr;
struct target_var_desc list[];
};
#define REFCOUNT_INFINITY (~(uintptr_t) 0)
#define REFCOUNT_LINK (~(uintptr_t) 1)
struct splay_tree_key_s {
uintptr_t host_start;
uintptr_t host_end;
struct target_mem_desc *tgt;
uintptr_t tgt_offset;
uintptr_t refcount;
splay_tree_key link_key;
};
static inline int
splay_compare (splay_tree_key x, splay_tree_key y)
{
if (x->host_start == x->host_end
&& y->host_start == y->host_end)
return 0;
if (x->host_end <= y->host_start)
return -1;
if (x->host_start >= y->host_end)
return 1;
return 0;
}
#include "splay-tree.h"
typedef struct acc_dispatch_t
{
struct target_mem_desc *data_environ;
__typeof (GOMP_OFFLOAD_openacc_exec) *exec_func;
__typeof (GOMP_OFFLOAD_openacc_register_async_cleanup)
*register_async_cleanup_func;
__typeof (GOMP_OFFLOAD_openacc_async_test) *async_test_func;
__typeof (GOMP_OFFLOAD_openacc_async_test_all) *async_test_all_func;
__typeof (GOMP_OFFLOAD_openacc_async_wait) *async_wait_func;
__typeof (GOMP_OFFLOAD_openacc_async_wait_async) *async_wait_async_func;
__typeof (GOMP_OFFLOAD_openacc_async_wait_all) *async_wait_all_func;
__typeof (GOMP_OFFLOAD_openacc_async_wait_all_async)
*async_wait_all_async_func;
__typeof (GOMP_OFFLOAD_openacc_async_set_async) *async_set_async_func;
__typeof (GOMP_OFFLOAD_openacc_create_thread_data) *create_thread_data_func;
__typeof (GOMP_OFFLOAD_openacc_destroy_thread_data)
*destroy_thread_data_func;
struct {
__typeof (GOMP_OFFLOAD_openacc_cuda_get_current_device)
*get_current_device_func;
__typeof (GOMP_OFFLOAD_openacc_cuda_get_current_context)
*get_current_context_func;
__typeof (GOMP_OFFLOAD_openacc_cuda_get_stream) *get_stream_func;
__typeof (GOMP_OFFLOAD_openacc_cuda_set_stream) *set_stream_func;
} cuda;
} acc_dispatch_t;
enum gomp_device_state
{
GOMP_DEVICE_UNINITIALIZED,
GOMP_DEVICE_INITIALIZED,
GOMP_DEVICE_FINALIZED
};
struct gomp_device_descr
{
const char *name;
unsigned int capabilities;
int target_id;
enum offload_target_type type;
__typeof (GOMP_OFFLOAD_get_name) *get_name_func;
__typeof (GOMP_OFFLOAD_get_caps) *get_caps_func;
__typeof (GOMP_OFFLOAD_get_type) *get_type_func;
__typeof (GOMP_OFFLOAD_get_num_devices) *get_num_devices_func;
__typeof (GOMP_OFFLOAD_init_device) *init_device_func;
__typeof (GOMP_OFFLOAD_fini_device) *fini_device_func;
__typeof (GOMP_OFFLOAD_version) *version_func;
__typeof (GOMP_OFFLOAD_load_image) *load_image_func;
__typeof (GOMP_OFFLOAD_unload_image) *unload_image_func;
__typeof (GOMP_OFFLOAD_alloc) *alloc_func;
__typeof (GOMP_OFFLOAD_free) *free_func;
__typeof (GOMP_OFFLOAD_dev2host) *dev2host_func;
__typeof (GOMP_OFFLOAD_host2dev) *host2dev_func;
__typeof (GOMP_OFFLOAD_dev2dev) *dev2dev_func;
__typeof (GOMP_OFFLOAD_can_run) *can_run_func;
__typeof (GOMP_OFFLOAD_run) *run_func;
__typeof (GOMP_OFFLOAD_async_run) *async_run_func;
struct splay_tree_s mem_map;
gomp_mutex_t lock;
enum gomp_device_state state;
acc_dispatch_t openacc;
};
enum gomp_map_vars_kind
{
GOMP_MAP_VARS_OPENACC,
GOMP_MAP_VARS_TARGET,
GOMP_MAP_VARS_DATA,
GOMP_MAP_VARS_ENTER_DATA
};
extern void gomp_acc_insert_pointer (size_t, void **, size_t *, void *);
extern void gomp_acc_remove_pointer (void *, bool, int, int);
extern struct target_mem_desc *gomp_map_vars (struct gomp_device_descr *,
size_t, void **, void **,
size_t *, void *, bool,
enum gomp_map_vars_kind);
extern void gomp_unmap_vars (struct target_mem_desc *, bool);
extern void gomp_init_device (struct gomp_device_descr *);
extern void gomp_free_memmap (struct splay_tree_s *);
extern void gomp_unload_device (struct gomp_device_descr *);
extern void gomp_init_work_share (struct gomp_work_share *, bool, unsigned);
extern void gomp_fini_work_share (struct gomp_work_share *);
extern bool gomp_work_share_start (bool);
extern void gomp_work_share_end (void);
extern bool gomp_work_share_end_cancel (void);
extern void gomp_work_share_end_nowait (void);
static inline void
gomp_work_share_init_done (void)
{
struct gomp_thread *thr = gomp_thread ();
if (__builtin_expect (thr->ts.last_work_share != NULL, 1))
gomp_ptrlock_set (&thr->ts.last_work_share->next_ws, thr->ts.work_share);
}
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility pop
#endif
#include "libgomp_g.h"
#include "omp-lock.h"
#define _LIBGOMP_OMP_LOCK_DEFINED 1
#include "omp.h.in"
#if !defined (HAVE_ATTRIBUTE_VISIBILITY) \
|| !defined (HAVE_ATTRIBUTE_ALIAS) \
|| !defined (HAVE_AS_SYMVER_DIRECTIVE) \
|| !defined (PIC) \
|| !defined (HAVE_SYMVER_SYMBOL_RENAMING_RUNTIME_SUPPORT)
# undef LIBGOMP_GNU_SYMBOL_VERSIONING
#endif
#ifdef LIBGOMP_GNU_SYMBOL_VERSIONING
extern void gomp_init_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_destroy_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_set_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_unset_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern int gomp_test_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_init_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void gomp_destroy_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void gomp_set_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void gomp_unset_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern int gomp_test_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void gomp_init_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_destroy_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_set_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_unset_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern int gomp_test_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_init_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_destroy_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_set_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_unset_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern int gomp_test_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
# define strong_alias(fn, al) \
extern __typeof (fn) al __attribute__ ((alias (#fn)));
# define omp_lock_symver(fn) \
__asm (".symver g" #fn "_30, " #fn "@@OMP_3.0"); \
__asm (".symver g" #fn "_25, " #fn "@OMP_1.0");
#else
# define gomp_init_lock_30 omp_init_lock
# define gomp_destroy_lock_30 omp_destroy_lock
# define gomp_set_lock_30 omp_set_lock
# define gomp_unset_lock_30 omp_unset_lock
# define gomp_test_lock_30 omp_test_lock
# define gomp_init_nest_lock_30 omp_init_nest_lock
# define gomp_destroy_nest_lock_30 omp_destroy_nest_lock
# define gomp_set_nest_lock_30 omp_set_nest_lock
# define gomp_unset_nest_lock_30 omp_unset_nest_lock
# define gomp_test_nest_lock_30 omp_test_nest_lock
#endif
#ifdef HAVE_ATTRIBUTE_VISIBILITY
# define attribute_hidden __attribute__ ((visibility ("hidden")))
#else
# define attribute_hidden
#endif
#ifdef HAVE_ATTRIBUTE_ALIAS
# define ialias_ulp	ialias_str1(__USER_LABEL_PREFIX__)
# define ialias_str1(x)	ialias_str2(x)
# define ialias_str2(x)	#x
# define ialias(fn) \
extern __typeof (fn) gomp_ialias_##fn \
__attribute__ ((alias (#fn))) attribute_hidden;
# define ialias_redirect(fn) \
extern __typeof (fn) fn __asm__ (ialias_ulp "gomp_ialias_" #fn) attribute_hidden;
# define ialias_call(fn) gomp_ialias_ ## fn
#else
# define ialias(fn)
# define ialias_redirect(fn)
# define ialias_call(fn) fn
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
static inline void
gomp_init_sfo (void)
{
gomp_global_sfo.in_thread_team_offset = offsetof (struct gomp_thread, ts);
gomp_global_sfo.in_thread_preemptable_offset = offsetof (struct gomp_thread, non_preemptable);
gomp_global_sfo.in_thread_task_offset = offsetof (struct gomp_thread, task);
gomp_global_sfo.in_task_kind_offset = offsetof (struct gomp_task, kind);
gomp_global_sfo.in_task_state_offset = offsetof (struct gomp_task, state);
gomp_global_sfo.in_state_context_offset = offsetof (struct gomp_task_state, context);
}
#endif
static inline size_t
priority_queue_offset (enum priority_queue_type type)
{
return offsetof (struct gomp_task, pnode[(int) type]);
}
static inline struct gomp_task *
priority_node_to_task (enum priority_queue_type type,
struct priority_node *node)
{
return (struct gomp_task *) ((char *) node - priority_queue_offset (type));
}
static inline struct priority_node *
task_to_priority_node (enum priority_queue_type type,
struct gomp_task *task)
{
return (struct priority_node *) ((char *) task
+ priority_queue_offset (type));
}
#endif 
