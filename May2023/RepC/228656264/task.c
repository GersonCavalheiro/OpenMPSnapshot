#include "libgomp.h"
#include <stdlib.h>
#include <string.h>
#include "gomp-constants.h"
#include "task.h"
typedef struct gomp_task_depend_entry *hash_entry_type;
static inline void *
htab_alloc (size_t size)
{
return gomp_malloc (size);
}
static inline void
htab_free (void *ptr)
{
free (ptr);
}
#include "hashtab.h"
static inline hashval_t
htab_hash (hash_entry_type element)
{
return hash_pointer (element->addr);
}
static inline bool
htab_eq (hash_entry_type x, hash_entry_type y)
{
return x->addr == y->addr;
}
void
gomp_init_task (struct gomp_task *task, struct gomp_task *parent_task,
struct gomp_task_icv *prev_icv)
{
task->parent = parent_task;
task->next_tied_task = NULL;
task->previous_tied_task = NULL;
task->ascending_tied_task = NULL;
task->undeferred_ancestor = NULL;
task->icv = *prev_icv;
task->type = GOMP_TASK_TYPE_TIED;
task->kind = GOMP_TASK_IMPLICIT;
task->taskwait = NULL;
task->creation_time = 0ULL;
task->completion_time = 0ULL;
task->state = NULL;
task->suspending_thread = NULL;
task->in_tied_task = false;
task->is_blocked = false;
task->final_task = false;
task->copy_ctors_done = false;
task->parent_depends_on = false;
priority_queue_init (&task->tied_children_queue);
priority_queue_init (&task->untied_children_queue);
task->taskgroup = NULL;
task->dependers = NULL;
task->depend_hash = NULL;
task->depend_count = 0;
}
void
gomp_end_task (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_task *task = thr->task;
gomp_finish_task (task);
thr->task = task->parent;
}
static inline void
gomp_clear_parent_in_list (enum priority_queue_type type, struct priority_list *list)
{
struct gomp_task *task;
struct priority_node *p;
if ((p = list->tasks) != NULL)
{
do
{
task = priority_node_to_task (type, p);
if (task->icv.ult_var && task->parent == task->ascending_tied_task)
task->ascending_tied_task = task->parent->ascending_tied_task;
task->parent = NULL;
p = p->next;
}
while (p != list->tasks);
}
if ((p = list->blocked_tasks) != NULL)
{
do
{
task = priority_node_to_task (type, p);
if (task->icv.ult_var && task->parent == task->ascending_tied_task)
task->ascending_tied_task = task->parent->ascending_tied_task;
task->parent = NULL;
p = p->next;
}
while (p != list->blocked_tasks);
}
}
static void
gomp_clear_parent_in_tree (enum priority_queue_type type, prio_splay_tree sp, prio_splay_tree_node node)
{
if (!node)
return;
prio_splay_tree_node left = node->left, right = node->right;
gomp_clear_parent_in_list (type, &node->key.l);
#if _LIBGOMP_CHECKING_
memset (node, 0xaf, sizeof (*node));
#endif
free (node);
gomp_clear_parent_in_tree (type, sp, left);
gomp_clear_parent_in_tree (type, sp, right);
}
static inline void
gomp_clear_parent (enum priority_queue_type type, struct priority_queue *q)
{
if (priority_queue_multi_p (q))
{
gomp_clear_parent_in_tree (type, &q->t, q->t.root);
q->t.root = NULL;
}
else
gomp_clear_parent_in_list (type, &q->l);
}
static void
gomp_task_handle_depend (struct gomp_task *task, struct gomp_task *parent,
void **depend)
{
size_t ndepend = (uintptr_t) depend[0];
size_t nout = (uintptr_t) depend[1];
size_t i;
hash_entry_type ent;
task->depend_count = ndepend;
task->num_dependees = 0;
if (parent->depend_hash == NULL)
parent->depend_hash = htab_create (2 * ndepend > 12 ? 2 * ndepend : 12);
for (i = 0; i < ndepend; i++)
{
task->depend[i].addr = depend[2 + i];
task->depend[i].next = NULL;
task->depend[i].prev = NULL;
task->depend[i].task = task;
task->depend[i].is_in = i >= nout;
task->depend[i].redundant = false;
task->depend[i].redundant_out = false;
hash_entry_type *slot = htab_find_slot (&parent->depend_hash,
&task->depend[i], INSERT);
hash_entry_type out = NULL, last = NULL;
if (*slot)
{
if ((*slot)->task == task)
{
task->depend[i].redundant = true;
continue;
}
for (ent = *slot; ent; ent = ent->next)
{
if (ent->redundant_out)
break;
last = ent;
if (i >= nout && ent->is_in)
continue;
if (!ent->is_in)
out = ent;
struct gomp_task *tsk = ent->task;
if (tsk->dependers == NULL)
{
tsk->dependers
= gomp_malloc (sizeof (struct gomp_dependers_vec)
+ 6 * sizeof (struct gomp_task *));
tsk->dependers->n_elem = 1;
tsk->dependers->allocated = 6;
tsk->dependers->elem[0] = task;
task->num_dependees++;
continue;
}
else if (tsk->dependers->n_elem
&& (tsk->dependers->elem[tsk->dependers->n_elem - 1]
== task))
continue;
else if (tsk->dependers->n_elem == tsk->dependers->allocated)
{
tsk->dependers->allocated
= tsk->dependers->allocated * 2 + 2;
tsk->dependers
= gomp_realloc (tsk->dependers,
sizeof (struct gomp_dependers_vec)
+ (tsk->dependers->allocated
* sizeof (struct gomp_task *)));
}
tsk->dependers->elem[tsk->dependers->n_elem++] = task;
task->num_dependees++;
}
task->depend[i].next = *slot;
(*slot)->prev = &task->depend[i];
}
*slot = &task->depend[i];
if (!task->depend[i].is_in && out)
{
if (out != last)
{
out->next->prev = out->prev;
out->prev->next = out->next;
out->next = last->next;
out->prev = last;
last->next = out;
if (out->next)
out->next->prev = out;
}
out->redundant_out = true;
}
}
}
void
GOMP_task (void (*fn) (void *), void *data, void (*cpyfn) (void *, void *),
long arg_size, long arg_align, bool if_clause, unsigned flags,
void **depend, int priority)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
enum gomp_task_type task_type;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_TASK_GRANULARITY_
if (thr->task && !thr->task->icv.ult_var)
{
thr->task->stage_end = RDTSC();
thr->task->sum_stages += (thr->task->stage_end - thr->task->stage_start);
}
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#ifdef HAVE_BROKEN_POSIX_SEMAPHORES
if (cpyfn)
if_clause = false;
flags &= ~GOMP_TASK_FLAG_UNTIED;
#endif
task_type = ((flags & GOMP_TASK_FLAG_UNTIED)) ? GOMP_TASK_TYPE_UNTIED : GOMP_TASK_TYPE_TIED;
if (team && (gomp_team_barrier_cancelled (&team->barrier) ||
(thr->task && thr->task->taskgroup && thr->task->taskgroup->cancelled)))
{
#if _LIBGOMP_TASK_GRANULARITY_
if (thr->task && !thr->task->icv.ult_var)
thr->task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if ((flags & GOMP_TASK_FLAG_PRIORITY) == 0)
priority = 0;
else if (priority > gomp_max_task_priority_var)
priority = gomp_max_task_priority_var;
undeferred_task_condition_check:
if (!if_clause || team == NULL || (thr->task && thr->task->final_task) ||
(gomp_auto_cutoff_var && team->task_count > 64 * team->nthreads))
{
struct gomp_task task;
struct gomp_task_state state;
if (team && thr->task && thr->task->icv.ult_var)
{
if (thr->task->kind == GOMP_TASK_IMPLICIT)
{
gomp_suspend_implicit_for_undeferred (fn, data, cpyfn, arg_size, arg_align, flags, depend, priority);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if (thr->last_tied_task && ((thr->task->type == GOMP_TASK_TYPE_TIED && thr->last_tied_task != thr->task) ||
(thr->task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task != thr->task->ascending_tied_task)))
{
if (gomp_undeferred_task_switch () == 0)
{
thr = gomp_thread ();
team = thr->ts.team;
}
goto undeferred_task_condition_check;
}
}
gomp_init_task (&task, thr->task, gomp_icv (false));
#if _LIBGOMP_TASK_TIMING_
task.creation_time = RDTSC();
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
task.creation_time = RDTSC();
#endif
if ((flags & GOMP_TASK_FLAG_DEPEND) &&
thr->task && thr->task->depend_hash)
gomp_task_maybe_wait_for_dependencies (depend);
task.fn = fn;
task.final_task = ((team == NULL) || (thr->task && thr->task->final_task) || (flags & GOMP_TASK_FLAG_FINAL));
if (task.final_task || (thr->task && thr->task->kind != GOMP_TASK_IMPLICIT && thr->task->type == GOMP_TASK_TYPE_TIED))
task.type = GOMP_TASK_TYPE_TIED;
else
task.type = task_type;
task.kind = GOMP_TASK_UNDEFERRED;
task.state = NULL;
if (!thr->task || priority > thr->task->priority)
task.priority = priority;
else
task.priority = thr->task->priority;
if (thr->task)
{
task.in_tied_task = thr->task->in_tied_task;
task.taskgroup = thr->task->taskgroup;
if (task.icv.ult_var)
{
if (thr->task->kind != GOMP_TASK_IMPLICIT)
{
if (thr->task->type == GOMP_TASK_TYPE_TIED)
task.ascending_tied_task = thr->task;
else
task.ascending_tied_task = thr->task->ascending_tied_task;
if (thr->task->undeferred_ancestor)
task.undeferred_ancestor = thr->task->undeferred_ancestor;
else
task.undeferred_ancestor = thr->task;
}
task.state = thr->task->state;
}
}
else if (task.icv.ult_var)
{
task.kind = GOMP_TASK_IMPLICIT;
state.switch_task = NULL;
state.switch_from_task = NULL;
task.state = &state;
}
if (task.state)
{
if (task.state->switch_task)
task.state->switch_task->state->switch_from_task = &task;
if (task.state->switch_from_task)
task.state->switch_from_task->state->switch_task = &task;
}
if (thr->task && task.type == GOMP_TASK_TYPE_TIED)
gomp_insert_task_in_tied_list (thr, thr->task, &task);
thr->task = &task;
#if _LIBGOMP_TASK_GRANULARITY_
if (!task.icv.ult_var)
{
task.sum_stages = 0ULL;
task.stage_start = RDTSC();
}
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
if (__builtin_expect (cpyfn != NULL, 0))
{
char buf[arg_size + arg_align - 1];
char *arg = (char *) (((uintptr_t) buf + arg_align - 1) &
~(uintptr_t) (arg_align - 1));
if (!thr->non_preemptable)
{
__atomic_store_n (&thr->non_preemptable, thr->non_preemptable+1, MEMMODEL_SEQ_CST);
cpyfn (arg, data);
__atomic_store_n (&thr->non_preemptable, thr->non_preemptable-1, MEMMODEL_SEQ_CST);
}
else
cpyfn (arg, data);
fn (arg);
}
else
fn (data);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_GRANULARITY_
if (!task.icv.ult_var)
{
task.stage_end = RDTSC();
task.sum_stages += (task.stage_end - task.stage_start);
gomp_save_task_granularity (thr->task_granularity_table, fn, task.sum_stages);
}
#endif
#if _LIBGOMP_TASK_TIMING_
task.completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, fn, task.kind, task.type, task.priority, (task.completion_time-task.creation_time));
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
{
task.completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, fn, task.kind, task.type, task.priority, (task.completion_time-task.creation_time));
}
#endif
if (!priority_queue_empty_p (&task.tied_children_queue, MEMMODEL_RELAXED) ||
!priority_queue_empty_p (&task.untied_children_queue, MEMMODEL_RELAXED))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
gomp_clear_parent (PQ_CHILDREN_TIED, &task.tied_children_queue);
gomp_clear_parent (PQ_CHILDREN_UNTIED, &task.untied_children_queue);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
}
gomp_end_task ();
if (thr->task && task.type == GOMP_TASK_TYPE_TIED)
gomp_remove_task_from_tied_list (thr, thr->task, &task);
if (task.state)
{
if (task.state->switch_task)
task.state->switch_task->state->switch_from_task = thr->task;
if (task.state->switch_from_task)
task.state->switch_from_task->state->switch_task = thr->task;
}
}
else
{
struct gomp_task *task;
struct gomp_task *parent = thr->task;
struct gomp_taskgroup *taskgroup = parent->taskgroup;
char *arg;
bool do_wake;
size_t depend_size = 0;
if (flags & GOMP_TASK_FLAG_DEPEND)
depend_size = ((uintptr_t) depend[0] *
sizeof (struct gomp_task_depend_entry));
task = gomp_malloc (sizeof (*task) + depend_size +
arg_size + arg_align - 1);
arg = (char *) (((uintptr_t) (task + 1) + depend_size + arg_align - 1) &
~(uintptr_t) (arg_align - 1));
gomp_init_task (task, parent, gomp_icv (false));
if (parent->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TASK_TIMING_
task->creation_time = RDTSC();
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
task->creation_time = RDTSC();
#endif
if (task->icv.ult_var)
{
if (parent->kind != GOMP_TASK_IMPLICIT)
{
if (parent->type == GOMP_TASK_TYPE_TIED)
task->ascending_tied_task = parent;
else
task->ascending_tied_task = parent->ascending_tied_task;
}
}
task->priority = priority;
task->type = task_type;
task->kind = GOMP_TASK_UNDEFERRED;
task->in_tied_task = parent->in_tied_task;
task->taskgroup = taskgroup;
thr->task = task;
if (cpyfn)
{
if (!thr->non_preemptable)
{
__atomic_store_n (&thr->non_preemptable, thr->non_preemptable+1, MEMMODEL_SEQ_CST);
cpyfn (arg, data);
__atomic_store_n (&thr->non_preemptable, thr->non_preemptable-1, MEMMODEL_SEQ_CST);
}
else
cpyfn (arg, data);
task->copy_ctors_done = true;
}
else
memcpy (arg, data, arg_size);
thr->task = parent;
task->kind = GOMP_TASK_WAITING;
task->fn = fn;
task->fn_data = arg;
task->final_task = (flags & GOMP_TASK_FLAG_FINAL) >> 1;
task->state = NULL;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (__builtin_expect ((gomp_team_barrier_cancelled (&team->barrier) ||
(taskgroup && taskgroup->cancelled)) && !task->copy_ctors_done, 0))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_finish_task (task);
free (task);
#if _LIBGOMP_TASK_GRANULARITY_
if (!parent->icv.ult_var)
parent->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if (taskgroup)
taskgroup->num_children++;
if (depend_size)
{
gomp_task_handle_depend (task, parent, depend);
if (task->num_dependees)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_TASK_GRANULARITY_
if (!parent->icv.ult_var)
parent->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
}
if (parent->icv.ult_var && parent->icv.wf_sched_var && parent->type == GOMP_TASK_TYPE_UNTIED &&
(task->type == GOMP_TASK_TYPE_UNTIED || thr->last_tied_task == NULL || thr->last_tied_task == task->ascending_tied_task))
{
if (task->type == GOMP_TASK_TYPE_TIED)
{
priority_queue_insert_running (PQ_CHILDREN_TIED, &parent->tied_children_queue, task, task->priority);
if (taskgroup)
priority_queue_insert_running (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue, task, task->priority);
}
else
{
priority_queue_insert_running (PQ_CHILDREN_UNTIED, &parent->untied_children_queue, task, task->priority);
if (taskgroup)
priority_queue_insert_running (PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue, task, task->priority);
}
task->kind = GOMP_TASK_TIED;
gomp_get_task_state_from_cache (thr, task);
++team->task_count;
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, parent, task);
#endif
gomp_suspend_untied_task_for_successor(thr, team, parent, task, NULL);
}
else
{
if (task->type == GOMP_TASK_TYPE_TIED)
{
priority_queue_insert (PQ_CHILDREN_TIED, &parent->tied_children_queue,
task, task->priority, INS_NEW_CHILD_POLICY(gomp_queue_policy_var), false, false);
if (taskgroup)
priority_queue_insert (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue,
task, task->priority, INS_NEW_GROUP_POLICY(gomp_queue_policy_var), false, false);
priority_queue_insert (PQ_TEAM_TIED, &team->tied_task_queue,
task, task->priority, INS_NEW_TEAM_POLICY(gomp_queue_policy_var), false, false);
}
else
{
priority_queue_insert (PQ_CHILDREN_UNTIED, &parent->untied_children_queue,
task, task->priority, INS_NEW_CHILD_POLICY(gomp_queue_policy_var), false, false);
if (taskgroup)
priority_queue_insert (PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue,
task, task->priority, INS_NEW_GROUP_POLICY(gomp_queue_policy_var), false, false);
priority_queue_insert (PQ_TEAM_UNTIED, &team->untied_task_queue,
task, task->priority, INS_NEW_TEAM_POLICY(gomp_queue_policy_var), false, false);
}
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && task->priority)
ipi_mask = get_mask_of_threads_with_lower_priority_tasks(thr, team, thr->thread_pool, task->priority, false);
#endif
++team->task_count;
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
do_wake = team->task_running_count +
!parent->in_tied_task < team->nthreads;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
gomp_team_barrier_wake (&team->barrier, 1);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
}
}
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if _LIBGOMP_TASK_GRANULARITY_
if (thr->task && !thr->task->icv.ult_var)
thr->task->stage_start = RDTSC();
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
}
ialias (GOMP_taskgroup_start)
ialias (GOMP_taskgroup_end)
#define TYPE long
#define UTYPE unsigned long
#define TYPE_is_long 1
#include "taskloop.c"
#undef TYPE
#undef UTYPE
#undef TYPE_is_long
#define TYPE unsigned long long
#define UTYPE TYPE
#define GOMP_taskloop GOMP_taskloop_ull
#include "taskloop.c"
#undef TYPE
#undef UTYPE
#undef GOMP_taskloop
static void inline
priority_queue_move_task_first (enum priority_queue_type type,
struct priority_queue *head,
struct gomp_task *task)
{
#if _LIBGOMP_CHECKING_
if (!priority_queue_task_in_queue_p (type, head, task))
gomp_fatal ("Attempt to move first missing task %p", task);
#endif
struct priority_list *list;
if (priority_queue_multi_p (head))
{
list = priority_queue_lookup_priority (head, task->priority);
#if _LIBGOMP_CHECKING_
if (!list)
gomp_fatal ("Unable to find priority %d", task->priority);
#endif
}
else
list = &head->l;
if (!task->is_blocked)
{
priority_list_remove (type, list, task_to_priority_node (type, task),
false, MEMMODEL_RELAXED);
priority_list_insert (type, list, task_to_priority_node (type, task), PRIORITY_INSERT_BEGIN,
task->parent_depends_on, false);
}
}
static void
gomp_target_task_completion (struct gomp_team *team, struct gomp_task *task)
{
bool tied_task = (task->type == GOMP_TASK_TYPE_TIED);
struct gomp_task *parent = task->parent;
if (parent)
{
if (tied_task)
priority_queue_move_task_first (PQ_CHILDREN_TIED, &parent->tied_children_queue, task);
else
priority_queue_move_task_first (PQ_CHILDREN_UNTIED, &parent->untied_children_queue, task);
}
struct gomp_taskgroup *taskgroup = task->taskgroup;
if (taskgroup)
{
if (tied_task)
priority_queue_move_task_first (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue, task);
else
priority_queue_move_task_first (PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue, task);
}
if (tied_task)
priority_queue_insert (PQ_TEAM_TIED, &team->tied_task_queue, task, task->priority,
INS_NEW_TEAM_POLICY(gomp_queue_policy_var), task->parent_depends_on, task->is_blocked);
else
priority_queue_insert (PQ_TEAM_UNTIED, &team->untied_task_queue, task, task->priority,
INS_NEW_TEAM_POLICY(gomp_queue_policy_var), task->parent_depends_on, task->is_blocked);
task->kind = GOMP_TASK_WAITING;
if (parent && parent->taskwait)
{
if (parent->taskwait->in_taskwait)
{
if (parent->icv.ult_var)
gomp_task_handle_blocking (team, parent);
else
{
parent->taskwait->in_taskwait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
else if (parent->taskwait->in_depend_wait)
{
if (parent->icv.ult_var)
gomp_task_handle_blocking (team, parent);
else
{
parent->taskwait->in_depend_wait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
}
if (taskgroup && taskgroup->in_taskgroup_wait)
{
taskgroup->in_taskgroup_wait = false;
if (!parent->icv.ult_var)
gomp_sem_post (&taskgroup->taskgroup_sem);
}
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && task->priority)
{
struct gomp_thread *thr = gomp_thread ();
ipi_mask = get_mask_of_threads_with_lower_priority_tasks(thr, team, thr->thread_pool, task->priority, false);
}
#endif
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
if (team->nthreads > team->task_running_count)
gomp_team_barrier_wake (&team->barrier, 1);
}
void
GOMP_PLUGIN_target_task_completion (void *data)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
struct gomp_thread *thr = gomp_thread ();
#endif
struct gomp_target_task *ttask = (struct gomp_target_task *) data;
struct gomp_task *task = ttask->task;
struct gomp_team *team = ttask->team;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (ttask->state == GOMP_TARGET_TASK_READY_TO_RUN)
{
ttask->state = GOMP_TARGET_TASK_FINISHED;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
return;
}
ttask->state = GOMP_TARGET_TASK_FINISHED;
gomp_target_task_completion (team, task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
}
static void inline priority_queue_downgrade_task (enum priority_queue_type,
struct priority_queue *, struct gomp_task *);
static void gomp_task_run_post_handle_depend_hash (struct gomp_task *);
bool
gomp_create_target_task (struct gomp_device_descr *devicep,
void (*fn) (void *), size_t mapnum, void **hostaddrs,
size_t *sizes, unsigned short *kinds,
unsigned int flags, void **depend, void **args,
enum gomp_target_task_state state)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
if (team && (gomp_team_barrier_cancelled (&team->barrier) ||
(thr->task->taskgroup && thr->task->taskgroup->cancelled)))
return true;
struct gomp_target_task *ttask;
struct gomp_task *task;
struct gomp_task *parent = thr->task;
struct gomp_taskgroup *taskgroup = parent->taskgroup;
bool do_wake;
size_t depend_size = 0;
uintptr_t depend_cnt = 0;
size_t tgt_align = 0, tgt_size = 0;
if (depend != NULL)
{
depend_cnt = (uintptr_t) depend[0];
depend_size = depend_cnt * sizeof (struct gomp_task_depend_entry);
}
if (fn)
{
size_t i;
for (i = 0; i < mapnum; i++)
if ((kinds[i] & 0xff) == GOMP_MAP_FIRSTPRIVATE)
{
size_t align = (size_t) 1 << (kinds[i] >> 8);
if (tgt_align < align)
tgt_align = align;
tgt_size = (tgt_size + align - 1) & ~(align - 1);
tgt_size += sizes[i];
}
if (tgt_align)
tgt_size += tgt_align - 1;
else
tgt_size = 0;
}
task = gomp_malloc (sizeof (*task) + depend_size
+ sizeof (*ttask)
+ mapnum * (sizeof (void *) + sizeof (size_t)
+ sizeof (unsigned short))
+ tgt_size);
gomp_init_task (task, parent, gomp_icv (false));
if (task->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TASK_TIMING_
task->creation_time = RDTSC();
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
task->creation_time = RDTSC();
#endif
if (task->icv.ult_var)
{
if (parent->kind != GOMP_TASK_IMPLICIT)
{
if (parent->type == GOMP_TASK_TYPE_TIED)
task->ascending_tied_task = parent;
else if (parent->ascending_tied_task)
task->ascending_tied_task = parent->ascending_tied_task;
}
}
task->priority = 0;
task->type = GOMP_TASK_TYPE_TIED;
task->kind = GOMP_TASK_WAITING;
task->in_tied_task = parent->in_tied_task;
task->taskgroup = taskgroup;
ttask = (struct gomp_target_task *) &task->depend[depend_cnt];
ttask->devicep = devicep;
ttask->fn = fn;
ttask->mapnum = mapnum;
ttask->args = args;
memcpy (ttask->hostaddrs, hostaddrs, mapnum * sizeof (void *));
ttask->sizes = (size_t *) &ttask->hostaddrs[mapnum];
memcpy (ttask->sizes, sizes, mapnum * sizeof (size_t));
ttask->kinds = (unsigned short *) &ttask->sizes[mapnum];
memcpy (ttask->kinds, kinds, mapnum * sizeof (unsigned short));
if (tgt_align)
{
char *tgt = (char *) &ttask->kinds[mapnum];
size_t i;
uintptr_t al = (uintptr_t) tgt & (tgt_align - 1);
if (al)
tgt += tgt_align - al;
tgt_size = 0;
for (i = 0; i < mapnum; i++)
if ((kinds[i] & 0xff) == GOMP_MAP_FIRSTPRIVATE)
{
size_t align = (size_t) 1 << (kinds[i] >> 8);
tgt_size = (tgt_size + align - 1) & ~(align - 1);
memcpy (tgt + tgt_size, hostaddrs[i], sizes[i]);
ttask->hostaddrs[i] = tgt + tgt_size;
tgt_size = tgt_size + sizes[i];
}
}
ttask->flags = flags;
ttask->state = state;
ttask->task = task;
ttask->team = team;
task->fn = NULL;
task->fn_data = ttask;
task->final_task = 0;
task->state = NULL;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (__builtin_expect (gomp_team_barrier_cancelled (&team->barrier)
|| (taskgroup && taskgroup->cancelled), 0))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_finish_task (task);
free (task);
return true;
}
if (depend_size)
{
gomp_task_handle_depend (task, parent, depend);
if (task->num_dependees)
{
if (taskgroup)
taskgroup->num_children++;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
return true;
}
}
if (state == GOMP_TARGET_TASK_DATA)
{
gomp_task_run_post_handle_depend_hash (task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_finish_task (task);
free (task);
return false;
}
if (taskgroup)
taskgroup->num_children++;
if (devicep != NULL && (devicep->capabilities & GOMP_OFFLOAD_CAP_OPENMP_400) &&
(!task->icv.ult_var || task->ascending_tied_task == thr->last_tied_task))
{
priority_queue_insert_running (PQ_CHILDREN_TIED, &parent->tied_children_queue, task, 0);
if (taskgroup)
priority_queue_insert_running (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue, task, 0);
task->pnode[PQ_TEAM_TIED].next = NULL;
task->pnode[PQ_TEAM_TIED].prev = NULL;
task->kind = GOMP_TASK_TIED;
++team->task_count;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_insert_task_in_tied_list (thr, parent, task);
thr->task = task;
gomp_target_task_fn (task->fn_data);
thr->task = parent;
gomp_remove_task_from_tied_list (thr, parent, task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
task->kind = GOMP_TASK_ASYNC_RUNNING;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
return true;
}
priority_queue_insert (PQ_CHILDREN_TIED, &parent->tied_children_queue,
task, 0, INS_NEW_CHILD_POLICY(gomp_queue_policy_var), false, false);
if (taskgroup)
priority_queue_insert (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue,
task, 0, INS_NEW_GROUP_POLICY(gomp_queue_policy_var), false, false);
priority_queue_insert (PQ_TEAM_TIED, &team->tied_task_queue,
task, 0, INS_NEW_TEAM_POLICY(gomp_queue_policy_var), false, false);
++team->task_count;
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
do_wake = team->task_running_count + !parent->in_tied_task
< team->nthreads;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
gomp_team_barrier_wake (&team->barrier, 1);
return true;
}
static struct gomp_task *
priority_tree_next_task_1 (enum priority_queue_type type, prio_splay_tree_node node)
{
again:
if (!node)
return NULL;
struct gomp_task *ret = priority_tree_next_task_1 (type, node->right);
if (ret)
return ret;
if (node->key.l.tasks != NULL)
{
ret = priority_node_to_task (type, node->key.l.tasks);
if (ret->kind == GOMP_TASK_WAITING)
return ret;
}
node = node->left;
goto again;
}
static inline struct gomp_task *
priority_tree_next_task (enum priority_queue_type type1,
struct priority_queue *q1,
enum priority_queue_type type2,
struct priority_queue *q2,
bool *q1_chosen_p)
{
struct gomp_task *t1 = priority_tree_next_task_1 (type1, q1->t.root);
struct gomp_task *t2 = NULL;
if (q2)
t2 = priority_tree_next_task_1 (type2, q2->t.root);
if (t1)
{
if (t2 && (t2->priority > t1->priority || (t2->priority == t1->priority
&& t2->parent_depends_on && !t1->parent_depends_on)))
{
*q1_chosen_p = false;
return t2;
}
*q1_chosen_p = true;
return t1;
}
else if (t2)
{
*q1_chosen_p = false;
return t2;
}
*q1_chosen_p = true;
return NULL;
}
static inline struct gomp_task *
priority_queue_next_task (enum priority_queue_type t1,
struct priority_queue *q1,
enum priority_queue_type t2,
struct priority_queue *q2,
bool *q1_chosen_p)
{
struct gomp_task *task1, *task2;
struct priority_node *node;
bool b1 = priority_queue_multi_p (q1);
bool b2 = (q2) ? priority_queue_multi_p (q2) : false;
if (b1 && b2)
return priority_tree_next_task (t1, q1, t2, q2, q1_chosen_p);
else
{
if (b1)
task1 = priority_tree_next_task (t1, q1, PQ_IGNORED, NULL, q1_chosen_p);
else if (q1)
{
node = q1->l.tasks;
task1 = (node) ? priority_node_to_task (t1, node) : NULL;
task1 = (task1 && task1->kind == GOMP_TASK_WAITING) ? task1 : NULL;
}
else
task1 = NULL;
if (b2)
task2 = priority_tree_next_task (t2, q2, PQ_IGNORED, NULL, q1_chosen_p);
else if (q2)
{
node = q2->l.tasks;
task2 = (node) ? priority_node_to_task (t2, node) : NULL;
task2 = (task2 && task2->kind == GOMP_TASK_WAITING) ? task2 : NULL;
}
else
task2 = NULL;
}
if (task1)
{
if (task2 && (task2->priority > task1->priority || (task2->priority == task1->priority
&& task2->parent_depends_on && !task1->parent_depends_on)))
{
*q1_chosen_p = false;
return task2;
}
*q1_chosen_p = true;
return task1;
}
else if (task2)
{
*q1_chosen_p = false;
return task2;
}
*q1_chosen_p = true;
return NULL;
}
static struct gomp_task *
priority_tree_next_suspended_task (enum priority_queue_type type, prio_splay_tree_node node)
{
again:
if (!node)
return NULL;
struct gomp_task *ret = priority_tree_next_suspended_task (type, node->right);
if (ret)
return ret;
if (node->key.l.tasks != NULL)
return priority_node_to_task (type, node->key.l.tasks);
node = node->left;
goto again;
}
static inline struct gomp_task *
priority_queue_next_suspended_task (enum priority_queue_type t, struct priority_queue *q)
{
struct gomp_task *task;
if (priority_queue_multi_p (q))
task = priority_tree_next_suspended_task (t, q->t.root);
else
task = (q->l.tasks) ? priority_node_to_task (t, q->l.tasks) : NULL;
return task;
}
static void inline
priority_list_upgrade_task (struct priority_list *list,
struct priority_node *node)
{
if (list->tasks == node)
{
if (list->last_parent_depends_on == NULL)
list->last_parent_depends_on = node;
return;
}
else if (list->last_parent_depends_on == NULL)
{
if (node->next == list->tasks)
{
list->tasks = node;
list->last_parent_depends_on = node;
return;
}
else
{
node->next->prev = node->prev;
node->prev->next = node->next;
node->next = list->tasks;
node->prev = list->tasks->prev;
list->tasks = node;
list->last_parent_depends_on = node;
node->next->prev = node;
node->prev->next = node;
return;
}
}
else
{
if (node->prev == list->last_parent_depends_on)
{
list->last_parent_depends_on = node;
return;
}
else if (node->next == list->tasks)
{
list->tasks = node;
return;
}
else
{
node->next->prev = node->prev;
node->prev->next = node->next;
node->next = list->last_parent_depends_on->next;
node->prev = list->last_parent_depends_on;
list->last_parent_depends_on = node;
node->next->prev = node;
node->prev->next = node;
return;
}
}
}
static void inline
priority_queue_upgrade_task (struct gomp_task *task,
struct gomp_task *parent)
{
enum priority_queue_type queue_type = (task->type == GOMP_TASK_TYPE_TIED) ? PQ_CHILDREN_TIED : PQ_CHILDREN_UNTIED;
struct priority_queue *head = (task->type == GOMP_TASK_TYPE_TIED) ? &parent->tied_children_queue : &parent->untied_children_queue;
struct priority_node *node = &task->pnode[queue_type];
#if _LIBGOMP_CHECKING_
if (!task->parent_depends_on)
gomp_fatal ("priority_queue_upgrade_task: task must be a parent_depends_on task");
if (!priority_queue_task_in_queue_p (queue_type, head, task))
gomp_fatal ("priority_queue_upgrade_task: cannot find task=%p", task);
#endif
if (priority_queue_multi_p (head))
{
struct priority_list *list = priority_queue_lookup_priority (head, task->priority);
priority_list_upgrade_task (list, node);
}
else
priority_list_upgrade_task (&head->l, node);
}
static void inline
priority_list_downgrade_task (enum priority_queue_type type,
struct priority_list *list,
struct gomp_task *child_task)
{
struct priority_node *node = task_to_priority_node (type, child_task);
if (list->tasks == node)
{
if (node->next != node)
list->tasks = node->next;
if (list->last_parent_depends_on == node)
list->last_parent_depends_on = NULL;
if (list->first_running_task == NULL)
list->first_running_task = node;
return;
}
else
{
if (list->last_parent_depends_on == node)
list->last_parent_depends_on = node->prev;
if (node->next == list->tasks || node->next == list->first_running_task)
{
list->first_running_task = node;
return;
}
else
{
node->prev->next = node->next;
node->next->prev = node->prev;
node->next = list->tasks;
node->prev = list->tasks->prev;
if (list->first_running_task == NULL)
list->first_running_task = node;
node->next->prev = node;
node->prev->next = node;
return;
}
}
}
static void inline
priority_queue_downgrade_task (enum priority_queue_type type,
struct priority_queue *head,
struct gomp_task *task)
{
#if _LIBGOMP_CHECKING_
if (!priority_queue_task_in_queue_p (type, head, task))
gomp_fatal ("Attempt to downgrade missing task %p", task);
#endif
if (priority_queue_multi_p (head))
{
struct priority_list *list = priority_queue_lookup_priority (head, task->priority);
priority_list_downgrade_task (type, list, task);
if (task->icv.ult_var)
{
struct gomp_task *first_task;
if (list->tasks == NULL || (first_task = priority_node_to_task (type, list->tasks)) == task || first_task->kind != GOMP_TASK_WAITING)
{
if (prio_splay_tree_hp_list_mark_node (&head->t, (prio_splay_tree_node) list, false))
{
if (head->t.highest_marked != NULL)
__atomic_store_n(&head->highest_priority, head->t.highest_marked->key.l.priority, MEMMODEL_RELEASE);
else
__atomic_store_n(&head->highest_priority, 0, MEMMODEL_RELEASE);
}
}
}
}
else
priority_list_downgrade_task (type, &head->l, task);
__atomic_store_n(&head->num_waiting_priority_node, head->num_waiting_priority_node-1, MEMMODEL_RELEASE);
}
static inline bool
gomp_task_run_pre (struct gomp_task *child_task, struct gomp_task *parent,
struct gomp_team *team)
{
bool tied_task = (child_task->type == GOMP_TASK_TYPE_TIED);
#if _LIBGOMP_CHECKING_
if (child_task->parent)
{
if (tied_task)
priority_queue_verify (PQ_CHILDREN_TIED,
&child_task->parent->tied_children_queue, true);
else
priority_queue_verify (PQ_CHILDREN_UNTIED,
&child_task->parent->untied_children_queue, true);
}
if (child_task->taskgroup)
{
if (tied_task)
priority_queue_verify (PQ_TASKGROUP_TIED,
&child_task->taskgroup->tied_taskgroup_queue, false);
else
priority_queue_verify (PQ_TASKGROUP_UNTIED,
&child_task->taskgroup->untied_taskgroup_queue, false);
}
if (child_task->suspending_thread)
{
if (child_task->kind == GOMP_TASK_TIED_SUSPENDED)
priority_queue_verify (PQ_SUSPENDED_TIED, &child_task->suspending_thread->tied_suspended, false);
else
priority_queue_verify (PQ_SUSPENDED_UNTIED, &child_task->suspending_thread->untied_suspended, false);
}
else if (child_task->kind == GOMP_TASK_WAITING)
{
if (tied_task)
priority_queue_verify (PQ_TEAM_TIED, &team->tied_task_queue, false);
else
priority_queue_verify (PQ_TEAM_UNTIED, &team->untied_task_queue, false);
}
#endif
struct gomp_taskgroup *taskgroup = child_task->taskgroup;
if (child_task->kind == GOMP_TASK_TIED_SUSPENDED)
{
priority_queue_remove (PQ_SUSPENDED_TIED, &child_task->suspending_thread->tied_suspended, child_task, child_task->is_blocked, MEMMODEL_RELAXED);
child_task->pnode[PQ_SUSPENDED_TIED].next = NULL;
child_task->pnode[PQ_SUSPENDED_TIED].prev = NULL;
if (child_task->state->switch_from_task != NULL)
child_task->state->switch_from_task->state->switch_task = child_task->state->switch_task;
if (child_task->state->switch_task != NULL)
child_task->state->switch_task->state->switch_from_task = child_task->state->switch_from_task;
if (child_task->undeferred_ancestor)
child_task->kind = GOMP_TASK_UNDEFERRED;
else
child_task->kind = GOMP_TASK_TIED;
}
else
{
if (child_task->undeferred_ancestor == NULL)
{
if (parent)
{
if (tied_task)
priority_queue_downgrade_task (PQ_CHILDREN_TIED, &parent->tied_children_queue, child_task);
else
priority_queue_downgrade_task (PQ_CHILDREN_UNTIED, &parent->untied_children_queue, child_task);
}
if (taskgroup)
{
if (tied_task)
priority_queue_downgrade_task (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue, child_task);
else
priority_queue_downgrade_task (PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue, child_task);
}
}
if (child_task->suspending_thread)
{
if (tied_task)
{
gomp_fatal ("Tasks marked as GOMP_TASK_TIED_SUSPENDED should not be handled at this point");
}
else
{
priority_queue_remove (PQ_SUSPENDED_UNTIED, &child_task->suspending_thread->untied_suspended, child_task, child_task->is_blocked, MEMMODEL_RELAXED);
child_task->pnode[PQ_SUSPENDED_UNTIED].next = NULL;
child_task->pnode[PQ_SUSPENDED_UNTIED].prev = NULL;
}
}
else if (child_task->icv.ult_var)
gomp_get_task_state_from_cache (gomp_thread (), child_task);
if (tied_task)
{
priority_queue_remove (PQ_TEAM_TIED, &team->tied_task_queue, child_task, child_task->is_blocked, MEMMODEL_RELAXED);
child_task->pnode[PQ_TEAM_TIED].next = NULL;
child_task->pnode[PQ_TEAM_TIED].prev = NULL;
}
else
{
priority_queue_remove (PQ_TEAM_UNTIED, &team->untied_task_queue, child_task, child_task->is_blocked, MEMMODEL_RELAXED);
child_task->pnode[PQ_TEAM_UNTIED].next = NULL;
child_task->pnode[PQ_TEAM_UNTIED].prev = NULL;
}
if (child_task->undeferred_ancestor)
child_task->kind = GOMP_TASK_UNDEFERRED;
else
child_task->kind = GOMP_TASK_TIED;
if (--team->task_queued_count == 0)
gomp_team_barrier_clear_task_pending (&team->barrier);
if (child_task->kind != GOMP_TASK_UNDEFERRED && parent && parent->icv.ult_var && parent->taskwait && parent->taskwait->in_taskwait)
gomp_task_handle_blocking (team, parent);
}
if ((gomp_team_barrier_cancelled (&team->barrier) || (taskgroup && taskgroup->cancelled)) && !child_task->copy_ctors_done)
return true;
return false;
}
static void
gomp_task_run_post_handle_depend_hash (struct gomp_task *child_task)
{
struct gomp_task *parent = child_task->parent;
size_t i;
for (i = 0; i < child_task->depend_count; i++)
if (!child_task->depend[i].redundant)
{
if (child_task->depend[i].next)
child_task->depend[i].next->prev = child_task->depend[i].prev;
if (child_task->depend[i].prev)
child_task->depend[i].prev->next = child_task->depend[i].next;
else
{
hash_entry_type *slot
= htab_find_slot (&parent->depend_hash, &child_task->depend[i],
NO_INSERT);
if (*slot != &child_task->depend[i])
abort ();
if (child_task->depend[i].next)
*slot = child_task->depend[i].next;
else
htab_clear_slot (parent->depend_hash, slot);
}
}
}
static size_t
gomp_task_run_post_handle_dependers (struct gomp_task *child_task,
struct gomp_team *team)
{
struct gomp_task *parent = child_task->parent;
size_t i, count = child_task->dependers->n_elem, ret = 0;
bool tied_task = (child_task->type == GOMP_TASK_TYPE_TIED);
#if defined HAVE_TLS || defined USE_EMUTLS
int max_priority = 0;
#endif
for (i = 0; i < count; i++)
{
struct gomp_task *task = child_task->dependers->elem[i];
if (--task->num_dependees != 0)
continue;
struct gomp_taskgroup *taskgroup = task->taskgroup;
if (parent)
{
if (tied_task)
priority_queue_insert (PQ_CHILDREN_TIED, &parent->tied_children_queue,
task, task->priority, INS_NEW_CHILD_POLICY(gomp_queue_policy_var),
task->parent_depends_on, false);
else
priority_queue_insert (PQ_CHILDREN_UNTIED, &parent->untied_children_queue,
task, task->priority, INS_NEW_CHILD_POLICY(gomp_queue_policy_var),
task->parent_depends_on, false);
if (parent->taskwait)
{
if (parent->taskwait->in_taskwait)
{
if (parent->icv.ult_var)
gomp_task_handle_blocking (team, parent);
else
{
parent->taskwait->in_taskwait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
else if (parent->taskwait->in_depend_wait)
{
if (parent->icv.ult_var)
gomp_task_handle_blocking (team, parent);
else
{
parent->taskwait->in_depend_wait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
}
}
if (taskgroup)
{
if (tied_task)
priority_queue_insert (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue,
task, task->priority, INS_NEW_GROUP_POLICY(gomp_queue_policy_var),
task->parent_depends_on, false);
else
priority_queue_insert (PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue,
task, task->priority, INS_NEW_GROUP_POLICY(gomp_queue_policy_var),
task->parent_depends_on, false);
if (taskgroup->in_taskgroup_wait)
{
taskgroup->in_taskgroup_wait = false;
if (!task->icv.ult_var)
gomp_sem_post (&taskgroup->taskgroup_sem);
}
}
if (tied_task)
priority_queue_insert (PQ_TEAM_TIED, &team->tied_task_queue,
task, task->priority, INS_NEW_TEAM_POLICY(gomp_queue_policy_var),
task->parent_depends_on, false);
else
priority_queue_insert (PQ_TEAM_UNTIED, &team->untied_task_queue,
task, task->priority, INS_NEW_TEAM_POLICY(gomp_queue_policy_var),
task->parent_depends_on, false);
#if defined HAVE_TLS || defined USE_EMUTLS
if (task->priority > max_priority)
max_priority = task->priority;
#endif
++team->task_count;
++team->task_queued_count;
++ret;
}
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && max_priority)
{
struct gomp_thread *thr = gomp_thread ();
ipi_mask = get_mask_of_threads_with_lower_priority_tasks(thr, team, thr->thread_pool, max_priority, false);
}
#endif
free (child_task->dependers);
child_task->dependers = NULL;
if (ret > 1)
gomp_team_barrier_set_task_pending (&team->barrier);
return ret;
}
static inline size_t
gomp_task_run_post_handle_depend (struct gomp_task *child_task,
struct gomp_team *team)
{
if (child_task->depend_count == 0)
return 0;
if (child_task->parent != NULL)
gomp_task_run_post_handle_depend_hash (child_task);
if (child_task->dependers == NULL)
return 0;
return gomp_task_run_post_handle_dependers (child_task, team);
}
static inline void
gomp_task_run_post_remove_parent (struct gomp_task *child_task)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_task *parent = child_task->parent;
if (parent == NULL)
return;
if (__builtin_expect (child_task->parent_depends_on, 0)
&& --parent->taskwait->n_depend == 0
&& parent->taskwait->in_depend_wait)
{
if (parent->icv.ult_var)
gomp_task_handle_blocking (thr->ts.team, parent);
else
{
parent->taskwait->in_depend_wait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
if (child_task->type == GOMP_TASK_TYPE_TIED)
{
if (priority_queue_remove (PQ_CHILDREN_TIED, &parent->tied_children_queue, child_task, false, MEMMODEL_RELEASE) &&
priority_queue_empty_p (&parent->untied_children_queue, MEMMODEL_RELAXED) &&
parent->taskwait && parent->taskwait->in_taskwait)
{
if (parent->icv.ult_var)
gomp_task_handle_blocking (thr->ts.team, parent);
else
{
parent->taskwait->in_taskwait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
child_task->pnode[PQ_CHILDREN_TIED].next = NULL;
child_task->pnode[PQ_CHILDREN_TIED].prev = NULL;
}
else
{
if (priority_queue_remove (PQ_CHILDREN_UNTIED, &parent->untied_children_queue, child_task, false, MEMMODEL_RELEASE) &&
priority_queue_empty_p (&parent->tied_children_queue, MEMMODEL_RELAXED) &&
parent->taskwait && parent->taskwait->in_taskwait)
{
if (parent->icv.ult_var)
gomp_task_handle_blocking (thr->ts.team, parent);
else
{
parent->taskwait->in_taskwait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
child_task->pnode[PQ_CHILDREN_UNTIED].next = NULL;
child_task->pnode[PQ_CHILDREN_UNTIED].prev = NULL;
}
}
static inline void
gomp_task_run_post_remove_taskgroup (struct gomp_task *child_task)
{
struct gomp_taskgroup *taskgroup = child_task->taskgroup;
if (taskgroup == NULL)
return;
bool empty;
if (child_task->type == GOMP_TASK_TYPE_TIED)
{
empty = priority_queue_remove (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue, child_task, false, MEMMODEL_RELAXED)
&& priority_queue_empty_p (&taskgroup->untied_taskgroup_queue, MEMMODEL_RELAXED);
child_task->pnode[PQ_TASKGROUP_TIED].next = NULL;
child_task->pnode[PQ_TASKGROUP_TIED].prev = NULL;
}
else
{
empty = priority_queue_remove (PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue, child_task, false, MEMMODEL_RELAXED)
&& priority_queue_empty_p (&taskgroup->tied_taskgroup_queue, MEMMODEL_RELAXED);
child_task->pnode[PQ_TASKGROUP_UNTIED].next = NULL;
child_task->pnode[PQ_TASKGROUP_UNTIED].prev = NULL;
}
if (taskgroup->num_children > 1)
--taskgroup->num_children;
else
{
__atomic_store_n (&taskgroup->num_children, 0, MEMMODEL_RELEASE);
}
if (empty && taskgroup->in_taskgroup_wait)
{
taskgroup->in_taskgroup_wait = false;
if (!child_task->icv.ult_var)
gomp_sem_post (&taskgroup->taskgroup_sem);
}
}
void
gomp_barrier_handle_tasks (gomp_barrier_state_t state)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *child_task = NULL;
struct gomp_task *to_free = NULL;
int do_wake = 0;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
if (gomp_icv (false)->ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (gomp_barrier_last_thread (state))
{
if (team->task_count == 0)
{
gomp_team_barrier_done (&team->barrier, state);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_team_barrier_wake (&team->barrier, 0);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
gomp_team_barrier_set_waiting_for_tasks (&team->barrier);
}
while (1)
{
bool cancelled = false;
if (!priority_queue_empty_p (&team->tied_task_queue, MEMMODEL_RELAXED) ||
!priority_queue_empty_p (&team->untied_task_queue, MEMMODEL_RELAXED))
{
bool ignored;
if (gomp_icv (false)->ult_var)
{
if (__atomic_load_n(&thr->tied_suspended.num_waiting_priority_node, MEMMODEL_RELAXED) > 0 ||
__atomic_load_n (&thr->untied_suspended.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
child_task = priority_queue_next_task (PQ_SUSPENDED_TIED, &thr->tied_suspended, PQ_SUSPENDED_UNTIED, &thr->untied_suspended, &ignored);
else
child_task = priority_queue_next_task (PQ_TEAM_TIED, &team->tied_task_queue, PQ_TEAM_UNTIED, &team->untied_task_queue, &ignored);
}
else
{
child_task = priority_queue_next_task (PQ_TEAM_TIED, &team->tied_task_queue, PQ_TEAM_UNTIED, &team->untied_task_queue, &ignored);
}
if (child_task)
{
cancelled = gomp_task_run_pre (child_task, child_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
team->task_running_count++;
child_task->in_tied_task = true;
}
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->hold_team_lock = false;
thr->task = child_task;
if (child_task->type == GOMP_TASK_TYPE_TIED)
gomp_insert_task_in_tied_list (thr, task, child_task);
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
team->task_running_count--;
struct gomp_target_task *ttask = (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
{
if (gomp_icv (false)->ult_var)
{
child_task->state->switch_task = task;
child_task->state->switch_from_task = NULL;
task->state->switch_from_task = child_task;
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, task, child_task);
#endif
context_switch(&child_task->state->switch_task->state->context,
&child_task->state->context);
if ((child_task = task->state->switch_from_task) != NULL)
task->state->switch_from_task = NULL;
}
else
{
#if _LIBGOMP_TASK_GRANULARITY_
child_task->sum_stages = 0ULL;
child_task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
child_task->fn (child_task->fn_data);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_TASK_TIMING_
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
{
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
}
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_GRANULARITY_
child_task->stage_end = RDTSC();
child_task->sum_stages += (child_task->stage_end - child_task->stage_start);
gomp_save_task_granularity (thr->task_granularity_table, child_task->fn, child_task->sum_stages);
#endif
}
}
if (child_task && child_task->type == GOMP_TASK_TYPE_TIED)
gomp_remove_task_from_tied_list (thr, task, child_task);
thr->task = task;
if (thr->hold_team_lock)
{
do_wake = team->task_running_count +
!task->in_tied_task < team->nthreads;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
thr->hold_team_lock = false;
if (do_wake)
gomp_team_barrier_wake (&team->barrier, 1);
}
}
else
{
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if (gomp_icv (false)->ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (child_task)
{
finish_cancelled:;
size_t new_tasks = gomp_task_run_post_handle_depend (child_task, team);
gomp_task_run_post_remove_parent (child_task);
gomp_clear_parent (PQ_CHILDREN_TIED, &child_task->tied_children_queue);
gomp_clear_parent (PQ_CHILDREN_UNTIED, &child_task->untied_children_queue);
gomp_task_run_post_remove_taskgroup (child_task);
to_free = child_task;
child_task = NULL;
if (!cancelled)
team->task_running_count--;
if (new_tasks > 1)
{
do_wake = team->nthreads - team->task_running_count;
if (do_wake > new_tasks)
do_wake = new_tasks;
}
if (--team->task_count == 0 && gomp_team_barrier_waiting_for_tasks (&team->barrier))
{
gomp_team_barrier_done (&team->barrier, state);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_team_barrier_wake (&team->barrier, 0);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
}
}
}
}
static inline int
gomp_check_queues_while_locked (struct gomp_thread *thr, struct gomp_team *team, struct gomp_task *task)
{
int ret = 0;
if (__atomic_load_n(&thr->tied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&thr->untied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (thr->last_tied_task != NULL) {
if (__atomic_load_n (&thr->last_tied_task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task == task->ascending_tied_task &&
__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&thr->last_tied_task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
} else {
if (__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&team->tied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
}
if (__atomic_load_n (&task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&team->untied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
return ret;
}
static inline int
gomp_check_queues_while_undeferred (struct gomp_thread *thr, struct gomp_team *team, struct gomp_task *task)
{
int ret = 0;
if (__atomic_load_n(&thr->tied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&thr->untied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (thr->last_tied_task != NULL) {
if (__atomic_load_n (&thr->last_tied_task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task == task->ascending_tied_task &&
__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&thr->last_tied_task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
} else {
if (__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&team->tied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
}
if (__atomic_load_n (&task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&team->untied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
return ret;
}
static inline int
gomp_check_queues_while_blocked (struct gomp_thread *thr, struct gomp_team *team, struct gomp_task *task)
{
int ret = 0;
if ((__atomic_load_n(&((task)->tied_children_queue.num_priority_node), MEMMODEL_ACQUIRE) +
__atomic_load_n(&((task)->untied_children_queue.num_priority_node), MEMMODEL_ACQUIRE)) == 0)
return ret;
if (__atomic_load_n(&thr->tied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&thr->untied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (thr->last_tied_task != NULL) {
if (__atomic_load_n (&thr->last_tied_task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task == task->ascending_tied_task &&
__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&thr->last_tied_task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
} else {
if (__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&team->tied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
}
if (__atomic_load_n (&task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
if (__atomic_load_n (&team->untied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
ret++;
return (ret) ? ret : -1;
}
static inline unsigned
gomp_get_runnable_tasks (struct gomp_thread *thr, struct gomp_team *team, struct gomp_task *task, struct gomp_task *next[])
{
unsigned t = 0;
bool ignored;
if (!priority_queue_empty_p (&thr->tied_suspended, MEMMODEL_RELAXED))
next[t++] = priority_queue_next_suspended_task (PQ_SUSPENDED_TIED, &thr->tied_suspended);
if (!priority_queue_empty_p (&thr->untied_suspended, MEMMODEL_RELAXED))
next[t++] = priority_queue_next_suspended_task (PQ_SUSPENDED_UNTIED, &thr->untied_suspended);
if (thr->last_tied_task != NULL)
{
if (!priority_queue_empty_p (&thr->last_tied_task->tied_children_queue, MEMMODEL_RELAXED) ||
!priority_queue_empty_p (&thr->last_tied_task->untied_children_queue, MEMMODEL_RELAXED))
next[t++] = priority_queue_next_task (PQ_CHILDREN_TIED, &thr->last_tied_task->tied_children_queue,
PQ_CHILDREN_UNTIED, &thr->last_tied_task->untied_children_queue, &ignored);
if (task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task == task->ascending_tied_task &&
!priority_queue_empty_p (&task->tied_children_queue, MEMMODEL_RELAXED))
next[t++] = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue,
PQ_IGNORED, NULL, &ignored);
if (!priority_queue_empty_p (&task->untied_children_queue, MEMMODEL_RELAXED) ||
!priority_queue_empty_p (&team->untied_task_queue, MEMMODEL_RELAXED))
next[t++] = priority_queue_next_task (PQ_CHILDREN_UNTIED, &task->untied_children_queue,
PQ_TEAM_UNTIED, &team->untied_task_queue, &ignored);
}
else
{
if (!priority_queue_empty_p (&task->tied_children_queue, MEMMODEL_RELAXED) ||
!priority_queue_empty_p (&task->untied_children_queue, MEMMODEL_RELAXED))
next[t++] = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue,
PQ_CHILDREN_UNTIED, &task->untied_children_queue, &ignored);
if (!priority_queue_empty_p (&team->tied_task_queue, MEMMODEL_RELAXED) ||
!priority_queue_empty_p (&team->untied_task_queue, MEMMODEL_RELAXED))
next[t++] = priority_queue_next_task (PQ_TEAM_TIED, &team->tied_task_queue,
PQ_TEAM_UNTIED, &team->untied_task_queue, &ignored);
}
return t;
}
void
gomp_task_handle_locking (struct priority_queue *locked_tasks)
{
#if defined HAVE_TLS || defined USE_EMUTLS || _LIBGOMP_TEAM_LOCK_TIMING_ || _LIBGOMP_LIBGOMP_TIMING_
struct gomp_thread *thr = gomp_thread ();
#endif
struct gomp_team *team = gomp_thread ()->ts.team;
struct gomp_task *task;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (priority_queue_empty_p (locked_tasks, MEMMODEL_RELAXED))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if ((task = priority_queue_next_suspended_task (PQ_LOCKED, locked_tasks)) == NULL)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
priority_queue_remove (PQ_LOCKED, locked_tasks, task, false, MEMMODEL_RELEASE);
task->pnode[PQ_LOCKED].next = NULL;
task->pnode[PQ_LOCKED].prev = NULL;
if (task->type == GOMP_TASK_TYPE_TIED)
{
task->is_blocked = false;
if (task->suspending_thread != NULL)
{
priority_queue_unblock_task (PQ_SUSPENDED_TIED, &task->suspending_thread->tied_suspended, task);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && task->suspending_thread != thr)
ipi_mask = get_mask_single_thread_with_lower_priority_tasks(task->suspending_thread, task->priority, true);
#endif
}
}
else
{
if (task->undeferred_ancestor == NULL)
{
if (task->parent)
priority_queue_unblock_task (PQ_CHILDREN_UNTIED, &task->parent->untied_children_queue, task);
if (task->taskgroup)
priority_queue_unblock_task (PQ_TASKGROUP_UNTIED, &task->taskgroup->untied_taskgroup_queue, task);
}
priority_queue_unblock_task (PQ_TEAM_UNTIED, &team->untied_task_queue, task);
task->is_blocked = false;
if (task->suspending_thread != NULL)
priority_queue_unblock_task (PQ_SUSPENDED_UNTIED, &task->suspending_thread->untied_suspended, task);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && (gomp_signal_unblock || task->priority))
ipi_mask = get_mask_of_threads_with_lower_priority_tasks(thr, team, thr->thread_pool, task->priority, true);
#endif
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
if (task->parent && task->parent->taskwait && task->parent->taskwait->in_taskwait)
gomp_task_handle_blocking (team, task->parent);
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
}
int
gomp_locked_task_switch (struct priority_queue *locked_tasks, gomp_mutex_t *lock)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *next_task;
struct gomp_task *next[5];
bool cancelled = false;
int do_wake;
unsigned t;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->critical_task_switch.number_of_invocations += 1;
#endif
if (gomp_check_queues_while_locked (thr, team, task) == 0)
{
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return 1;
}
#if GOMP_ATOMIC_READ_MUTEX
if (__atomic_load_n (&team->task_lock, MEMMODEL_ACQUIRE))
{
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return 1;
}
#endif
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
if (gomp_mutex_trylock (&team->task_lock))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_ && _LIBGOMP_LIBGOMP_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->libgomp_time->gomp_time += (tmp_time - thr->libgomp_time->entry_time);
#elif _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_acquisition_time += (RDTSC() - thr->team_lock_time->entry_acquisition_time);
#elif _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return 1;
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
#if GOMP_ATOMIC_READ_MUTEX
if (__atomic_load_n(lock, MEMMODEL_ACQUIRE) == 0)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
return 2;
}
#else
__atomic_store_n (&thr->non_preemptable, thr->non_preemptable+1, MEMMODEL_SEQ_CST);
if (gomp_mutex_trylock (lock) == 0)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return 2;
}
__atomic_store_n (&thr->non_preemptable, thr->non_preemptable-1, MEMMODEL_SEQ_CST);
#endif
t = gomp_get_runnable_tasks (thr, team, task, next);
if ((next_task = get_higher_priority_task (next, t)) == NULL)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return 1;
}
cancelled = gomp_task_run_pre (next_task, next_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (next_task->undeferred_ancestor)
next_task = next_task->undeferred_ancestor;
do_wake = gomp_terminate_task_pre (team, task, next_task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_terminate_task_post (thr, team, next_task, do_wake);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return 1;
}
if (task->type == GOMP_TASK_TYPE_TIED)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->critical_task_switch.number_of_tied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->critical_task_switch.number_of_tied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->critical_task_switch.number_of_tied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->critical_task_switch.number_of_tied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_tied_task_for_successor(thr, team, task, next_task, locked_tasks);
}
else
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->critical_task_switch.number_of_untied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->critical_task_switch.number_of_untied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->critical_task_switch.number_of_untied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->critical_task_switch.number_of_untied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_untied_task_for_successor(thr, team, task, next_task, locked_tasks);
}
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return 0;
}
int
gomp_undeferred_task_switch (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *next_task;
struct gomp_task *next[5];
bool cancelled = false;
int do_wake;
unsigned t;
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->undeferred_task_switch.number_of_invocations += 1;
#endif
if (gomp_check_queues_while_undeferred (thr, team, task) == 0)
return 1;
#if GOMP_ATOMIC_READ_MUTEX
if (__atomic_load_n (&team->task_lock, MEMMODEL_ACQUIRE))
return 1;
#endif
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
if (gomp_mutex_trylock (&team->task_lock))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_acquisition_time += (RDTSC() - thr->team_lock_time->entry_acquisition_time);
#endif
return 1;
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
t = gomp_get_runnable_tasks (thr, team, task, next);
if ((next_task = get_higher_priority_task (next, t)) == NULL)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
return 1;
}
cancelled = gomp_task_run_pre (next_task, next_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (next_task->undeferred_ancestor)
next_task = next_task->undeferred_ancestor;
do_wake = gomp_terminate_task_pre (team, task, next_task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_terminate_task_post (thr, team, next_task, do_wake);
return 1;
}
if (task->type == GOMP_TASK_TYPE_TIED)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->undeferred_task_switch.number_of_tied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->undeferred_task_switch.number_of_tied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->undeferred_task_switch.number_of_tied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->undeferred_task_switch.number_of_tied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_tied_task_for_successor(thr, team, task, next_task, NULL);
}
else
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->undeferred_task_switch.number_of_untied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->undeferred_task_switch.number_of_untied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->undeferred_task_switch.number_of_untied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->undeferred_task_switch.number_of_untied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_untied_task_for_successor(thr, team, task, next_task, NULL);
}
return 0;
}
int
gomp_blocked_task_switch (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *next_task;
struct gomp_task *next[5];
bool cancelled = false;
int do_wake;
unsigned t;
int ret;
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->blocked_task_switch.number_of_invocations += 1;
#endif
if (__atomic_load_n(&thr->tied_suspended.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
if (__atomic_load_n (&thr->untied_suspended.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
if (thr->last_tied_task != NULL) {
if (__atomic_load_n (&thr->last_tied_task->tied_children_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
if (task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task == task->ascending_tied_task &&
__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
if (__atomic_load_n (&thr->last_tied_task->untied_children_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
} else {
if (__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
if (__atomic_load_n (&team->tied_task_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
}
if (__atomic_load_n (&task->untied_children_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
if (__atomic_load_n (&team->untied_task_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto blocked_task_try_to_schedule;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
while (1)
{
if ((ret = gomp_check_queues_while_blocked (thr, team, task)) == -1)
continue;
else if (ret == 0)
return 1;
#if GOMP_ATOMIC_READ_MUTEX
else if (__atomic_load_n (&team->task_lock, MEMMODEL_ACQUIRE))
continue;
#endif
else
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
if (gomp_mutex_trylock (&team->task_lock) == 0)
break;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_acquisition_time += (RDTSC() - thr->team_lock_time->entry_acquisition_time);
#endif
}
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
blocked_task_try_to_schedule:
t = gomp_get_runnable_tasks (thr, team, task, next);
if ((next_task = get_higher_priority_task (next, t)) == NULL)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
return 1;
}
cancelled = gomp_task_run_pre (next_task, next_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (next_task->undeferred_ancestor)
next_task = next_task->undeferred_ancestor;
do_wake = gomp_terminate_task_pre (team, task, next_task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_terminate_task_post (thr, team, next_task, do_wake);
return 1;
}
if (task->type == GOMP_TASK_TYPE_TIED)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->blocked_task_switch.number_of_tied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->blocked_task_switch.number_of_tied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->blocked_task_switch.number_of_tied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->blocked_task_switch.number_of_tied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_tied_task_for_successor(thr, team, task, next_task, NULL);
}
else
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->blocked_task_switch.number_of_untied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->blocked_task_switch.number_of_untied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->blocked_task_switch.number_of_untied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->blocked_task_switch.number_of_untied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_untied_task_for_successor(thr, team, task, next_task, NULL);
}
return 0;
}
void
GOMP_taskwait (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *child_task = NULL;
struct gomp_task *to_free = NULL;
struct gomp_taskwait taskwait;
int do_wake = 0;
if (task == NULL || (priority_queue_empty_p (&task->tied_children_queue, MEMMODEL_ACQUIRE) &&
priority_queue_empty_p (&task->untied_children_queue, MEMMODEL_ACQUIRE)))
return;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_TASK_GRANULARITY_
if (!task->icv.ult_var)
{
task->stage_end = RDTSC();
task->sum_stages += (task->stage_end - task->stage_start);
}
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
memset (&taskwait, 0, sizeof (taskwait));
bool child_q = false;
if (task->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
while (1)
{
bool cancelled = false;
if (priority_queue_empty_p (&task->tied_children_queue, MEMMODEL_RELAXED) &&
priority_queue_empty_p (&task->untied_children_queue, MEMMODEL_RELAXED))
{
bool destroy_taskwait = task->taskwait != NULL;
task->taskwait = NULL;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
}
if (destroy_taskwait)
{
gomp_sem_destroy (&taskwait.taskwait_sem);
if (task->icv.ult_var)
taskwait.in_taskwait = false;
}
#if _LIBGOMP_TASK_GRANULARITY_
if (!task->icv.ult_var)
task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
struct gomp_task *next_task;
if (task->icv.ult_var)
{
if (thr->last_tied_task != NULL &&
((task->type == GOMP_TASK_TYPE_TIED && thr->last_tied_task != task) ||
(task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task != task->ascending_tied_task)))
next_task = priority_queue_next_task (PQ_CHILDREN_UNTIED, &task->untied_children_queue, PQ_IGNORED, NULL, &child_q);
else
next_task = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue, PQ_CHILDREN_UNTIED, &task->untied_children_queue, &child_q);;
}
else
{
next_task = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue, PQ_CHILDREN_UNTIED, &task->untied_children_queue, &child_q);
}
if (next_task && next_task->kind == GOMP_TASK_WAITING)
{
child_task = next_task;
cancelled = gomp_task_run_pre (child_task, task, team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
if (task->icv.ult_var)
{
if (task->taskwait == NULL)
{
taskwait.in_depend_wait = false;
task->taskwait = &taskwait;
}
taskwait.in_taskwait = true;
if (task->type == GOMP_TASK_TYPE_TIED)
{
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, task, child_task);
#endif
gomp_suspend_tied_task_for_successor(thr, team, task, child_task, NULL);
}
else
{
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, task, child_task);
#endif
gomp_suspend_untied_task_for_successor(thr, team, task, child_task, NULL);
}
thr = gomp_thread ();
team = thr->ts.team;
child_task = NULL;
goto taskwait_resume_flow;
}
}
else
{
if (task->taskwait == NULL)
{
taskwait.in_depend_wait = false;
if (!task->icv.ult_var)
gomp_sem_init (&taskwait.taskwait_sem, 0);
task->taskwait = &taskwait;
}
taskwait.in_taskwait = true;
if (task->icv.ult_var)
{
if (gomp_blocked_task_switch() == 0)
{
thr = gomp_thread ();
team = thr->ts.team;
}
goto taskwait_resume_flow;
}
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
taskwait_resume_flow:
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->task = child_task;
if (child_task->type == GOMP_TASK_TYPE_TIED)
gomp_insert_task_in_tied_list (thr, task, child_task);
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
struct gomp_target_task *ttask = (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
{
#if _LIBGOMP_TASK_GRANULARITY_
child_task->sum_stages = 0ULL;
child_task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
child_task->fn (child_task->fn_data);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_TASK_TIMING_
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
{
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
}
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_GRANULARITY_
child_task->stage_end = RDTSC();
child_task->sum_stages += (child_task->stage_end - child_task->stage_start);
gomp_save_task_granularity (thr->task_granularity_table, child_task->fn, child_task->sum_stages);
#endif
}
if (child_task->type == GOMP_TASK_TYPE_TIED)
gomp_remove_task_from_tied_list (thr, task, child_task);
thr->task = task;
}
else if (!task->icv.ult_var)
gomp_sem_wait (&taskwait.taskwait_sem);
if (task->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (child_task)
{
finish_cancelled:;
size_t new_tasks = gomp_task_run_post_handle_depend (child_task, team);
if (child_task->parent == task)
{
if (child_task->type == GOMP_TASK_TYPE_TIED)
{
priority_queue_remove (PQ_CHILDREN_TIED, &task->tied_children_queue,
child_task, false, MEMMODEL_RELAXED);
child_task->pnode[PQ_CHILDREN_TIED].next = NULL;
child_task->pnode[PQ_CHILDREN_TIED].prev = NULL;
}
else
{
priority_queue_remove (PQ_CHILDREN_UNTIED, &task->untied_children_queue,
child_task, false, MEMMODEL_RELAXED);
child_task->pnode[PQ_CHILDREN_UNTIED].next = NULL;
child_task->pnode[PQ_CHILDREN_UNTIED].prev = NULL;
}
}
else
gomp_task_run_post_remove_parent (child_task);
gomp_clear_parent (PQ_CHILDREN_TIED, &child_task->tied_children_queue);
gomp_clear_parent (PQ_CHILDREN_UNTIED, &child_task->untied_children_queue);
gomp_task_run_post_remove_taskgroup (child_task);
to_free = child_task;
child_task = NULL;
team->task_count--;
if (new_tasks > 1)
{
do_wake = team->nthreads - team->task_running_count - !task->in_tied_task;
if (do_wake > new_tasks)
do_wake = new_tasks;
}
}
}
}
#if defined HAVE_TLS || defined USE_EMUTLS
void
gomp_interrupt_task_scheduling_pre (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *next_task;
struct gomp_task *next[5];
bool cancelled = false;
int do_wake;
unsigned t;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_invocations += 1;
#endif
if (gomp_signal_unblock) {
if (task->priority <= __atomic_load_n (&thr->tied_suspended.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
if (task->priority <= __atomic_load_n (&thr->untied_suspended.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
} else {
if (task->priority < __atomic_load_n (&thr->tied_suspended.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
if (task->priority < __atomic_load_n (&thr->untied_suspended.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
}
if (thr->last_tied_task != NULL) {
if (task->priority < __atomic_load_n (&thr->last_tied_task->tied_children_queue.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
if (task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task == task->ascending_tied_task &&
task->priority < __atomic_load_n (&task->tied_children_queue.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
if (task->priority < __atomic_load_n (&thr->last_tied_task->untied_children_queue.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
} else {
if (task->priority < __atomic_load_n (&task->tied_children_queue.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
if (task->priority < __atomic_load_n (&team->tied_task_queue.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
}
if (task->priority < __atomic_load_n (&task->untied_children_queue.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
if (task->priority < __atomic_load_n (&team->untied_task_queue.highest_priority, MEMMODEL_ACQUIRE))
goto interrupt_try_to_schedule;
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
interrupt_try_to_schedule:
gomp_put_task_state_in_cache (thr);
#if GOMP_ATOMIC_READ_MUTEX
if (__atomic_load_n (&team->task_lock, MEMMODEL_ACQUIRE))
{
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
#endif
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
if (gomp_mutex_trylock (&team->task_lock))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_ && _LIBGOMP_LIBGOMP_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->libgomp_time->gomp_time += (tmp_time - thr->libgomp_time->entry_time);
#elif _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_acquisition_time += (RDTSC() - thr->team_lock_time->entry_acquisition_time);
#elif _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
t = gomp_get_runnable_tasks (thr, team, task, next);
if (task->icv.untied_block_var)
next_task = get_higher_priority_task (next, t);
else
next_task = get_not_in_taskwait_higher_priority_task (next, t);
if (next_task == NULL || next_task->priority < task->priority || (next_task->priority == task->priority
&& (task->parent_depends_on || (gomp_signal_unblock && next_task->suspending_thread == NULL)
|| (!gomp_signal_unblock && !next_task->parent_depends_on))))
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
cancelled = gomp_task_run_pre (next_task, next_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (next_task->undeferred_ancestor)
next_task = next_task->undeferred_ancestor;
do_wake = gomp_terminate_task_pre (team, task, next_task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_terminate_task_post (thr, team, next_task, do_wake);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if (task->type == GOMP_TASK_TYPE_TIED)
{
next_task->state->switch_task = task;
next_task->state->switch_from_task = NULL;
task->state->switch_from_task = next_task;
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->interrupt_task_switch.number_of_tied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->interrupt_task_switch.number_of_tied_suspension_for_tied_resume += 1;
#endif
gomp_insert_task_in_tied_list (thr, task, next_task);
}
#if _LIBGOMP_TASK_SWITCH_AUDITING_
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->interrupt_task_switch.number_of_tied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->interrupt_task_switch.number_of_tied_suspension_for_untied_resume += 1;
}
#endif
if (task->taskwait && task->taskwait->in_taskwait && IS_BLOCKED_TIED(task))
{
task->is_blocked = true;
priority_queue_insert (PQ_SUSPENDED_TIED, &thr->tied_suspended, task, task->priority,
PRIORITY_INSERT_BEGIN, task->parent_depends_on, task->is_blocked);
}
else
{
task->is_blocked = false;
priority_queue_insert (PQ_SUSPENDED_TIED, &thr->tied_suspended, task, task->priority,
INS_SUSP_CHILD_POLICY(gomp_queue_policy_var), task->parent_depends_on, task->is_blocked);
}
task->suspending_thread = thr;
task->kind = GOMP_TASK_TIED_SUSPENDED;
thr->hold_team_lock = true;
thr->task = next_task;
context_restore(&next_task->state->context);
}
else
{
next_task->state->switch_task = task->state->switch_task;
next_task->state->switch_from_task = NULL;
task->state->switch_task->state->switch_from_task = next_task;
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->interrupt_task_switch.number_of_untied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->interrupt_task_switch.number_of_untied_suspension_for_tied_resume += 1;
#endif
gomp_insert_task_in_tied_list (thr, task, next_task);
}
#if _LIBGOMP_TASK_SWITCH_AUDITING_
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->interrupt_task_switch.number_of_untied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->interrupt_task_switch.number_of_untied_suspension_for_untied_resume += 1;
}
#endif
task->state->switch_task = NULL;
task->state->switch_from_task = NULL;
if (task->taskwait && task->taskwait->in_taskwait && IS_BLOCKED_UNTIED(task))
{
if (task->kind != GOMP_TASK_UNDEFERRED)
{
if (task->parent)
priority_queue_block_task (PQ_CHILDREN_UNTIED, &task->parent->untied_children_queue, task);
if (task->taskgroup)
priority_queue_block_task (PQ_TASKGROUP_UNTIED, &task->taskgroup->untied_taskgroup_queue, task);
}
task->is_blocked = true;
priority_queue_insert (PQ_SUSPENDED_UNTIED, &thr->untied_suspended, task, task->priority,
PRIORITY_INSERT_BEGIN, task->parent_depends_on, task->is_blocked);
task->suspending_thread = thr;
task->kind = GOMP_TASK_WAITING;
priority_queue_insert (PQ_TEAM_UNTIED, &team->untied_task_queue, task, task->priority,
INS_SUSP_TEAM_POLICY(gomp_queue_policy_var), task->parent_depends_on, task->is_blocked);
}
else
{
if (task->kind != GOMP_TASK_UNDEFERRED)
{
if (task->parent)
priority_queue_adjust_task (PQ_CHILDREN_UNTIED, &task->parent->untied_children_queue, task, task->parent_depends_on);
if (task->taskgroup)
priority_queue_adjust_task (PQ_TASKGROUP_UNTIED, &task->taskgroup->untied_taskgroup_queue, task, task->parent_depends_on);
}
task->is_blocked = false;
priority_queue_insert (PQ_SUSPENDED_UNTIED, &thr->untied_suspended, task, task->priority,
INS_SUSP_CHILD_POLICY(gomp_queue_policy_var), task->parent_depends_on, task->is_blocked);
task->suspending_thread = thr;
task->kind = GOMP_TASK_WAITING;
priority_queue_insert (PQ_TEAM_UNTIED, &team->untied_task_queue, task, task->priority,
INS_SUSP_TEAM_POLICY(gomp_queue_policy_var), task->parent_depends_on, task->is_blocked);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && task->priority)
ipi_mask = get_mask_of_threads_with_lower_priority_tasks(thr, team, thr->thread_pool, task->priority, false);
#endif
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
if (task->parent && task->parent->taskwait && task->parent->taskwait->in_taskwait)
gomp_task_handle_blocking (team, task->parent);
}
thr->hold_team_lock = true;
thr->task = next_task;
context_restore(&next_task->state->context);
}
}
void
gomp_interrupt_task_scheduling_post (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *next_task;
int do_wake;
if (task->type == GOMP_TASK_TYPE_TIED && (next_task = task->state->switch_from_task) != NULL)
{
task->state->switch_from_task = NULL;
if (next_task->type == GOMP_TASK_TYPE_TIED)
gomp_remove_task_from_tied_list (thr, task, next_task);
if (thr->hold_team_lock)
thr->hold_team_lock = false;
else
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
}
if (task->kind == GOMP_TASK_TIED_SUSPENDED)
{
priority_queue_remove (PQ_SUSPENDED_TIED, &task->suspending_thread->tied_suspended, task, task->is_blocked, MEMMODEL_RELAXED);
task->pnode[PQ_SUSPENDED_TIED].next = NULL;
task->pnode[PQ_SUSPENDED_TIED].prev = NULL;
if (task->undeferred_ancestor)
task->kind = GOMP_TASK_UNDEFERRED;
else
task->kind = GOMP_TASK_TIED;
}
do_wake = gomp_terminate_task_pre (team, task, next_task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_terminate_task_post (thr, team, next_task, do_wake);
}
else
{
if (thr->hold_team_lock)
{
do_wake = team->task_running_count + !task->in_tied_task < team->nthreads;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
thr->hold_team_lock = false;
if (do_wake)
gomp_team_barrier_wake (&team->barrier, 1);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
}
}
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
}
#endif
void
gomp_task_maybe_wait_for_dependencies (void **depend)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_task *task = thr->task;
struct gomp_team *team = thr->ts.team;
struct gomp_task_depend_entry elem, *ent = NULL;
struct gomp_taskwait taskwait;
size_t ndepend = (uintptr_t) depend[0];
size_t nout = (uintptr_t) depend[1];
size_t i;
size_t num_awaited = 0;
struct gomp_task *child_task = NULL;
struct gomp_task *to_free = NULL;
int do_wake = 0;
if (task->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
for (i = 0; i < ndepend; i++)
{
elem.addr = depend[i + 2];
ent = htab_find (task->depend_hash, &elem);
for (; ent; ent = ent->next)
if (i >= nout && ent->is_in)
continue;
else
{
struct gomp_task *tsk = ent->task;
if (!tsk->parent_depends_on)
{
tsk->parent_depends_on = true;
++num_awaited;
if (tsk->num_dependees == 0 && tsk->kind == GOMP_TASK_WAITING && !tsk->is_blocked)
priority_queue_upgrade_task (tsk, task);
}
}
}
if (num_awaited == 0)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
return;
}
memset (&taskwait, 0, sizeof (taskwait));
taskwait.n_depend = num_awaited;
if (!task->icv.ult_var)
gomp_sem_init (&taskwait.taskwait_sem, 0);
task->taskwait = &taskwait;
while (1)
{
bool cancelled = false;
if (taskwait.n_depend == 0)
{
task->taskwait = NULL;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
}
gomp_sem_destroy (&taskwait.taskwait_sem);
return;
}
bool ignored;
struct gomp_task *next_task;
if (task->icv.ult_var)
{
if (thr->last_tied_task != NULL &&
((task->type == GOMP_TASK_TYPE_TIED && thr->last_tied_task != task) ||
(task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task != task->ascending_tied_task)))
next_task = priority_queue_next_task (PQ_CHILDREN_UNTIED, &task->untied_children_queue, PQ_IGNORED, NULL, &ignored);
else
next_task = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue, PQ_CHILDREN_UNTIED, &task->untied_children_queue, &ignored);
}
else
{
next_task = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue, PQ_CHILDREN_UNTIED, &task->untied_children_queue, &ignored);
}
if (next_task && next_task->kind == GOMP_TASK_WAITING)
{
child_task = next_task;
cancelled = gomp_task_run_pre (child_task, task, team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
if (task->icv.ult_var)
{
taskwait.in_depend_wait = true;
if (task->type == GOMP_TASK_TYPE_TIED)
{
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, task, child_task);
#endif
gomp_suspend_tied_task_for_successor(thr, team, task, child_task, NULL);
}
else
{
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, task, child_task);
#endif
gomp_suspend_untied_task_for_successor(thr, team, task, child_task, NULL);
}
thr = gomp_thread ();
team = thr->ts.team;
child_task = NULL;
goto task_maybe_wait_resume_flow;
}
}
else
{
taskwait.in_depend_wait = true;
if (task->icv.ult_var)
{
if (gomp_blocked_task_switch() == 0)
{
thr = gomp_thread ();
team = thr->ts.team;
}
goto task_maybe_wait_resume_flow;
}
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
task_maybe_wait_resume_flow:
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->task = child_task;
if (child_task->type == GOMP_TASK_TYPE_TIED)
gomp_insert_task_in_tied_list (thr, task, child_task);
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
struct gomp_target_task *ttask = (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
{
#if _LIBGOMP_TASK_GRANULARITY_
child_task->sum_stages = 0ULL;
child_task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
child_task->fn (child_task->fn_data);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_TASK_TIMING_
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
{
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
}
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_GRANULARITY_
child_task->stage_end = RDTSC();
child_task->sum_stages += (child_task->stage_end - child_task->stage_start);
gomp_save_task_granularity (thr->task_granularity_table, child_task->fn, child_task->sum_stages);
#endif
}
if (child_task->type == GOMP_TASK_TYPE_TIED)
gomp_remove_task_from_tied_list (thr, task, child_task);
thr->task = task;
}
else if (!task->icv.ult_var)
gomp_sem_wait (&taskwait.taskwait_sem);
if (task->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (child_task)
{
finish_cancelled:;
size_t new_tasks = gomp_task_run_post_handle_depend (child_task, team);
if (child_task->parent_depends_on)
--taskwait.n_depend;
if (child_task->parent == task)
{
if (child_task->type == GOMP_TASK_TYPE_TIED)
{
priority_queue_remove (PQ_CHILDREN_TIED, &task->tied_children_queue,
child_task, false, MEMMODEL_RELAXED);
child_task->pnode[PQ_CHILDREN_TIED].next = NULL;
child_task->pnode[PQ_CHILDREN_TIED].prev = NULL;
}
else
{
priority_queue_remove (PQ_CHILDREN_UNTIED, &task->untied_children_queue,
child_task, false, MEMMODEL_RELAXED);
child_task->pnode[PQ_CHILDREN_UNTIED].next = NULL;
child_task->pnode[PQ_CHILDREN_UNTIED].prev = NULL;
}
}
else
gomp_task_run_post_remove_parent (child_task);
gomp_clear_parent (PQ_CHILDREN_TIED, &child_task->tied_children_queue);
gomp_clear_parent (PQ_CHILDREN_UNTIED, &child_task->untied_children_queue);
gomp_task_run_post_remove_taskgroup (child_task);
to_free = child_task;
child_task = NULL;
team->task_count--;
if (new_tasks > 1)
{
do_wake = team->nthreads - team->task_running_count - !task->in_tied_task;
if (do_wake > new_tasks)
do_wake = new_tasks;
}
}
}
}
void
GOMP_taskyield (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_task *next_task;
struct gomp_task *next[5];
bool cancelled = false;
int do_wake;
unsigned t;
if (task == NULL || team == NULL || !task->icv.ult_var)
return;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->voluntary_task_switch.number_of_invocations += 1;
#endif
if (__atomic_load_n(&thr->tied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
if (__atomic_load_n (&thr->untied_suspended.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
if (thr->last_tied_task != NULL) {
if (__atomic_load_n (&thr->last_tied_task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
if (task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task == task->ascending_tied_task &&
__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_RELAXED) > 0)
goto taskyield_try_to_schedule;
if (__atomic_load_n (&thr->last_tied_task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
} else {
if (__atomic_load_n (&task->tied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
if (__atomic_load_n (&team->tied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
}
if (__atomic_load_n (&task->untied_children_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
if (__atomic_load_n (&team->untied_task_queue.num_waiting_priority_node, MEMMODEL_ACQUIRE) > 0)
goto taskyield_try_to_schedule;
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
taskyield_try_to_schedule:
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
t = gomp_get_runnable_tasks (thr, team, task, next);
if ((next_task = get_higher_priority_task (next, t)) == NULL)
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
cancelled = gomp_task_run_pre (next_task, next_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (next_task->undeferred_ancestor)
next_task = next_task->undeferred_ancestor;
do_wake = gomp_terminate_task_pre (team, task, next_task);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
gomp_terminate_task_post (thr, team, next_task, do_wake);
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if (task->type == GOMP_TASK_TYPE_TIED)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->voluntary_task_switch.number_of_tied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->voluntary_task_switch.number_of_tied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->voluntary_task_switch.number_of_tied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->voluntary_task_switch.number_of_tied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_tied_task_for_successor(thr, team, task, next_task, NULL);
}
else
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
if (next_task->type == GOMP_TASK_TYPE_TIED)
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->voluntary_task_switch.number_of_untied_suspension_for_suspended_tied_resume += 1;
else
thr->task_switch_audit->voluntary_task_switch.number_of_untied_suspension_for_tied_resume += 1;
}
else
{
if (next_task->suspending_thread != NULL)
thr->task_switch_audit->voluntary_task_switch.number_of_untied_suspension_for_suspended_untied_resume += 1;
else
thr->task_switch_audit->voluntary_task_switch.number_of_untied_suspension_for_untied_resume += 1;
}
#endif
gomp_suspend_untied_task_for_successor(thr, team, task, next_task, NULL);
}
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
}
void
GOMP_taskgroup_start (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_taskgroup *taskgroup;
if (team == NULL)
return;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_GRANULARITY_
if (task && !task->icv.ult_var)
{
task->stage_end = RDTSC();
task->sum_stages += (task->stage_end - task->stage_start);
}
#endif
taskgroup = gomp_malloc (sizeof (struct gomp_taskgroup));
taskgroup->prev = task->taskgroup;
priority_queue_init (&taskgroup->tied_taskgroup_queue);
priority_queue_init (&taskgroup->untied_taskgroup_queue);
taskgroup->in_taskgroup_wait = false;
taskgroup->cancelled = false;
taskgroup->num_children = 0;
if (!task->icv.ult_var)
gomp_sem_init (&taskgroup->taskgroup_sem, 0);
task->taskgroup = taskgroup;
#if _LIBGOMP_TASK_GRANULARITY_
if (task && !task->icv.ult_var)
task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
}
void
GOMP_taskgroup_end (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
struct gomp_taskgroup *taskgroup;
struct gomp_task *child_task = NULL;
struct gomp_task *to_free = NULL;
int do_wake = 0;
if (team == NULL)
return;
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_GRANULARITY_
if (task && !task->icv.ult_var)
{
task->stage_end = RDTSC();
task->sum_stages += (task->stage_end - task->stage_start);
}
#endif
taskgroup = task->taskgroup;
if (__builtin_expect (taskgroup == NULL, 0) && thr->ts.level == 0)
{
gomp_team_barrier_wait (&team->barrier);
#if _LIBGOMP_TASK_GRANULARITY_
if (task && !task->icv.ult_var)
task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
return;
}
if (__atomic_load_n (&taskgroup->num_children, MEMMODEL_ACQUIRE) == 0)
goto finish;
bool unused;
if (task->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
while (1)
{
bool cancelled = false;
if (priority_queue_empty_p (&taskgroup->tied_taskgroup_queue, MEMMODEL_RELAXED) &&
priority_queue_empty_p (&taskgroup->untied_taskgroup_queue, MEMMODEL_RELAXED))
{
if (taskgroup->num_children)
{
if (priority_queue_empty_p (&task->tied_children_queue, MEMMODEL_RELAXED) &&
priority_queue_empty_p (&task->untied_children_queue, MEMMODEL_RELAXED))
goto do_wait;
if (task->icv.ult_var)
{
if (thr->last_tied_task != NULL &&
((task->type == GOMP_TASK_TYPE_TIED && thr->last_tied_task != task) ||
(task->type == GOMP_TASK_TYPE_UNTIED && thr->last_tied_task != task->ascending_tied_task)))
child_task = priority_queue_next_task (PQ_CHILDREN_UNTIED, &task->untied_children_queue, PQ_IGNORED, NULL, &unused);
else
child_task = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue, PQ_CHILDREN_UNTIED, &task->untied_children_queue, &unused);
}
else
{
child_task = priority_queue_next_task (PQ_CHILDREN_TIED, &task->tied_children_queue, PQ_CHILDREN_UNTIED, &task->untied_children_queue, &unused);
}
}
else
{
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
}
goto finish;
}
}
else
{
child_task = priority_queue_next_task (PQ_TASKGROUP_TIED, &taskgroup->tied_taskgroup_queue,
PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue, &unused);
if (child_task && child_task->type == GOMP_TASK_TYPE_TIED &&
thr->last_tied_task && thr->last_tied_task != child_task->ascending_tied_task)
{
child_task = priority_queue_next_task (PQ_TASKGROUP_UNTIED, &taskgroup->untied_taskgroup_queue,
PQ_IGNORED, NULL, &unused);
}
}
if (child_task && child_task->kind == GOMP_TASK_WAITING)
{
cancelled = gomp_task_run_pre (child_task, child_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
if (task->icv.ult_var)
{
taskgroup->in_taskgroup_wait = true;
if (task->type == GOMP_TASK_TYPE_TIED)
{
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, task, child_task);
#endif
gomp_suspend_tied_task_for_successor(thr, team, task, child_task, NULL);
}
else
{
#if (_LIBGOMP_TASK_SWITCH_AUDITING_ && _LIBGOMP_IN_FLOW_TASK_SWITCH_AUDITING_)
gomp_save_inflow_task_switch (thr->task_switch_audit, task, child_task);
#endif
gomp_suspend_untied_task_for_successor(thr, team, task, child_task, NULL);
}
thr = gomp_thread ();
team = thr->ts.team;
child_task = NULL;
goto taskgroup_end_resume_flow;
}
}
else
{
child_task = NULL;
do_wait:
taskgroup->in_taskgroup_wait = true;
if (task->icv.ult_var)
{
if (gomp_blocked_task_switch() == 0)
{
thr = gomp_thread ();
team = thr->ts.team;
}
goto taskgroup_end_resume_flow;
}
}
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->lock_time += (RDTSC() - thr->team_lock_time->entry_time);
#endif
gomp_mutex_unlock (&team->task_lock);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var && ipi_mask > 0)
{
#if _LIBGOMP_TASK_SWITCH_AUDITING_
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_syscall += 1;
thr->task_switch_audit->interrupt_task_switch.number_of_ipi_sent += gomp_count_1_bits(ipi_mask);
#endif
gomp_send_ipi();
}
#endif
taskgroup_end_resume_flow:
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
if (to_free)
{
gomp_finish_task (to_free);
if (to_free->icv.ult_var)
{
gomp_free_task_state (thr->global_task_state_group, thr->local_task_state_list, to_free->state);
}
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->task = child_task;
if (child_task->type == GOMP_TASK_TYPE_TIED)
gomp_insert_task_in_tied_list (thr, task, child_task);
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
struct gomp_target_task *ttask = (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
{
#if _LIBGOMP_TASK_GRANULARITY_
child_task->sum_stages = 0ULL;
child_task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
child_task->fn (child_task->fn_data);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = true;
#endif
#if _LIBGOMP_TASK_TIMING_
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
#else
if (gomp_ipi_var && gomp_ipi_decision_model > 0.0)
{
child_task->completion_time = RDTSC();
gomp_save_task_time(thr->prio_task_time, child_task->fn, child_task->kind, child_task->type, child_task->priority, (child_task->completion_time-child_task->creation_time));
}
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->entry_time = RDTSC();
#endif
#if _LIBGOMP_TASK_GRANULARITY_
child_task->stage_end = RDTSC();
child_task->sum_stages += (child_task->stage_end - child_task->stage_start);
gomp_save_task_granularity (thr->task_granularity_table, child_task->fn, child_task->sum_stages);
#endif
}
if (child_task->type == GOMP_TASK_TYPE_TIED)
gomp_remove_task_from_tied_list (thr, task, child_task);
thr->task = task;
}
else if (!task->icv.ult_var)
gomp_sem_wait (&taskgroup->taskgroup_sem);
if (task->icv.ult_var)
gomp_put_task_state_in_cache (thr);
#if _LIBGOMP_TEAM_LOCK_TIMING_
thr->team_lock_time->entry_acquisition_time = RDTSC();
#endif
gomp_mutex_lock (&team->task_lock);
#if _LIBGOMP_TEAM_LOCK_TIMING_
uint64_t tmp_time = RDTSC();
thr->team_lock_time->lock_acquisition_time += (tmp_time - thr->team_lock_time->entry_acquisition_time);
thr->team_lock_time->entry_time = tmp_time;
#endif
if (child_task)
{
finish_cancelled:;
size_t new_tasks = gomp_task_run_post_handle_depend (child_task, team);
gomp_task_run_post_remove_parent (child_task);
gomp_clear_parent (PQ_CHILDREN_TIED, &child_task->tied_children_queue);
gomp_clear_parent (PQ_CHILDREN_UNTIED, &child_task->untied_children_queue);
gomp_task_run_post_remove_taskgroup (child_task);
to_free = child_task;
child_task = NULL;
team->task_count--;
if (new_tasks > 1)
{
do_wake = team->nthreads - team->task_running_count - !task->in_tied_task;
if (do_wake > new_tasks)
do_wake = new_tasks;
}
}
}
finish:
#if _LIBGOMP_TASK_GRANULARITY_
if (task && !task->icv.ult_var)
task->stage_start = RDTSC();
#endif
#if _LIBGOMP_LIBGOMP_TIMING_
thr->libgomp_time->gomp_time += (RDTSC() - thr->libgomp_time->entry_time);
#endif
task->taskgroup = taskgroup->prev;
gomp_sem_destroy (&taskgroup->taskgroup_sem);
free (taskgroup);
#if defined HAVE_TLS || defined USE_EMUTLS
if (gomp_ipi_var)
thr->in_libgomp = false;
#endif
}
int
omp_in_final (void)
{
struct gomp_thread *thr = gomp_thread ();
return thr->task && thr->task->final_task;
}
ialias (omp_in_final)
