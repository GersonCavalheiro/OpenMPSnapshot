#include "libgomp.h"
#include <stdlib.h>
#include <string.h>
#include "gomp-constants.h"
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
task->icv = *prev_icv;
task->kind = GOMP_TASK_IMPLICIT;
task->taskwait = NULL;
task->in_tied_task = false;
task->final_task = false;
task->copy_ctors_done = false;
task->parent_depends_on = false;
priority_queue_init (&task->children_queue);
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
gomp_clear_parent_in_list (struct priority_list *list)
{
struct priority_node *p = list->tasks;
if (p)
do
{
priority_node_to_task (PQ_CHILDREN, p)->parent = NULL;
p = p->next;
}
while (p != list->tasks);
}
static void
gomp_clear_parent_in_tree (prio_splay_tree sp, prio_splay_tree_node node)
{
if (!node)
return;
prio_splay_tree_node left = node->left, right = node->right;
gomp_clear_parent_in_list (&node->key.l);
#if _LIBGOMP_CHECKING_
memset (node, 0xaf, sizeof (*node));
#endif
free (node);
gomp_clear_parent_in_tree (sp, left);
gomp_clear_parent_in_tree (sp, right);
}
static inline void
gomp_clear_parent (struct priority_queue *q)
{
if (priority_queue_multi_p (q))
{
gomp_clear_parent_in_tree (&q->t, q->t.root);
q->t.root = NULL;
}
else
gomp_clear_parent_in_list (&q->l);
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
#ifdef HAVE_BROKEN_POSIX_SEMAPHORES
if (cpyfn)
if_clause = false;
flags &= ~GOMP_TASK_FLAG_UNTIED;
#endif
if (team
&& (gomp_team_barrier_cancelled (&team->barrier)
|| (thr->task->taskgroup && thr->task->taskgroup->cancelled)))
return;
if ((flags & GOMP_TASK_FLAG_PRIORITY) == 0)
priority = 0;
else if (priority > gomp_max_task_priority_var)
priority = gomp_max_task_priority_var;
if (!if_clause || team == NULL
|| (thr->task && thr->task->final_task)
|| team->task_count > 64 * team->nthreads)
{
struct gomp_task task;
if ((flags & GOMP_TASK_FLAG_DEPEND)
&& thr->task && thr->task->depend_hash)
gomp_task_maybe_wait_for_dependencies (depend);
gomp_init_task (&task, thr->task, gomp_icv (false));
task.kind = GOMP_TASK_UNDEFERRED;
task.final_task = (thr->task && thr->task->final_task)
|| (flags & GOMP_TASK_FLAG_FINAL);
task.priority = priority;
if (thr->task)
{
task.in_tied_task = thr->task->in_tied_task;
task.taskgroup = thr->task->taskgroup;
}
thr->task = &task;
if (__builtin_expect (cpyfn != NULL, 0))
{
char buf[arg_size + arg_align - 1];
char *arg = (char *) (((uintptr_t) buf + arg_align - 1)
& ~(uintptr_t) (arg_align - 1));
cpyfn (arg, data);
fn (arg);
}
else
fn (data);
if (!priority_queue_empty_p (&task.children_queue, MEMMODEL_RELAXED))
{
gomp_mutex_lock (&team->task_lock);
gomp_clear_parent (&task.children_queue);
gomp_mutex_unlock (&team->task_lock);
}
gomp_end_task ();
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
depend_size = ((uintptr_t) depend[0]
* sizeof (struct gomp_task_depend_entry));
task = gomp_malloc (sizeof (*task) + depend_size
+ arg_size + arg_align - 1);
arg = (char *) (((uintptr_t) (task + 1) + depend_size + arg_align - 1)
& ~(uintptr_t) (arg_align - 1));
gomp_init_task (task, parent, gomp_icv (false));
task->priority = priority;
task->kind = GOMP_TASK_UNDEFERRED;
task->in_tied_task = parent->in_tied_task;
task->taskgroup = taskgroup;
thr->task = task;
if (cpyfn)
{
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
gomp_mutex_lock (&team->task_lock);
if (__builtin_expect ((gomp_team_barrier_cancelled (&team->barrier)
|| (taskgroup && taskgroup->cancelled))
&& !task->copy_ctors_done, 0))
{
gomp_mutex_unlock (&team->task_lock);
gomp_finish_task (task);
free (task);
return;
}
if (taskgroup)
taskgroup->num_children++;
if (depend_size)
{
gomp_task_handle_depend (task, parent, depend);
if (task->num_dependees)
{
gomp_mutex_unlock (&team->task_lock);
return;
}
}
priority_queue_insert (PQ_CHILDREN, &parent->children_queue,
task, priority,
PRIORITY_INSERT_BEGIN,
false,
task->parent_depends_on);
if (taskgroup)
priority_queue_insert (PQ_TASKGROUP, &taskgroup->taskgroup_queue,
task, priority,
PRIORITY_INSERT_BEGIN,
false,
task->parent_depends_on);
priority_queue_insert (PQ_TEAM, &team->task_queue,
task, priority,
PRIORITY_INSERT_END,
false,
task->parent_depends_on);
++team->task_count;
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
do_wake = team->task_running_count + !parent->in_tied_task
< team->nthreads;
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
gomp_team_barrier_wake (&team->barrier, 1);
}
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
priority_list_remove (list, task_to_priority_node (type, task), 0);
priority_list_insert (type, list, task, task->priority,
PRIORITY_INSERT_BEGIN, type == PQ_CHILDREN,
task->parent_depends_on);
}
static void
gomp_target_task_completion (struct gomp_team *team, struct gomp_task *task)
{
struct gomp_task *parent = task->parent;
if (parent)
priority_queue_move_task_first (PQ_CHILDREN, &parent->children_queue,
task);
struct gomp_taskgroup *taskgroup = task->taskgroup;
if (taskgroup)
priority_queue_move_task_first (PQ_TASKGROUP, &taskgroup->taskgroup_queue,
task);
priority_queue_insert (PQ_TEAM, &team->task_queue, task, task->priority,
PRIORITY_INSERT_BEGIN, false,
task->parent_depends_on);
task->kind = GOMP_TASK_WAITING;
if (parent && parent->taskwait)
{
if (parent->taskwait->in_taskwait)
{
parent->taskwait->in_taskwait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
else if (parent->taskwait->in_depend_wait)
{
parent->taskwait->in_depend_wait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
if (taskgroup && taskgroup->in_taskgroup_wait)
{
taskgroup->in_taskgroup_wait = false;
gomp_sem_post (&taskgroup->taskgroup_sem);
}
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
if (team->nthreads > team->task_running_count)
gomp_team_barrier_wake (&team->barrier, 1);
}
void
GOMP_PLUGIN_target_task_completion (void *data)
{
struct gomp_target_task *ttask = (struct gomp_target_task *) data;
struct gomp_task *task = ttask->task;
struct gomp_team *team = ttask->team;
gomp_mutex_lock (&team->task_lock);
if (ttask->state == GOMP_TARGET_TASK_READY_TO_RUN)
{
ttask->state = GOMP_TARGET_TASK_FINISHED;
gomp_mutex_unlock (&team->task_lock);
return;
}
ttask->state = GOMP_TARGET_TASK_FINISHED;
gomp_target_task_completion (team, task);
gomp_mutex_unlock (&team->task_lock);
}
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
if (team
&& (gomp_team_barrier_cancelled (&team->barrier)
|| (thr->task->taskgroup && thr->task->taskgroup->cancelled)))
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
task->priority = 0;
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
gomp_mutex_lock (&team->task_lock);
if (__builtin_expect (gomp_team_barrier_cancelled (&team->barrier)
|| (taskgroup && taskgroup->cancelled), 0))
{
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
gomp_mutex_unlock (&team->task_lock);
return true;
}
}
if (state == GOMP_TARGET_TASK_DATA)
{
gomp_task_run_post_handle_depend_hash (task);
gomp_mutex_unlock (&team->task_lock);
gomp_finish_task (task);
free (task);
return false;
}
if (taskgroup)
taskgroup->num_children++;
if (devicep != NULL
&& (devicep->capabilities & GOMP_OFFLOAD_CAP_OPENMP_400))
{
priority_queue_insert (PQ_CHILDREN, &parent->children_queue, task, 0,
PRIORITY_INSERT_END,
false,
task->parent_depends_on);
if (taskgroup)
priority_queue_insert (PQ_TASKGROUP, &taskgroup->taskgroup_queue,
task, 0, PRIORITY_INSERT_END,
false,
task->parent_depends_on);
task->pnode[PQ_TEAM].next = NULL;
task->pnode[PQ_TEAM].prev = NULL;
task->kind = GOMP_TASK_TIED;
++team->task_count;
gomp_mutex_unlock (&team->task_lock);
thr->task = task;
gomp_target_task_fn (task->fn_data);
thr->task = parent;
gomp_mutex_lock (&team->task_lock);
task->kind = GOMP_TASK_ASYNC_RUNNING;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
gomp_mutex_unlock (&team->task_lock);
return true;
}
priority_queue_insert (PQ_CHILDREN, &parent->children_queue, task, 0,
PRIORITY_INSERT_BEGIN,
false,
task->parent_depends_on);
if (taskgroup)
priority_queue_insert (PQ_TASKGROUP, &taskgroup->taskgroup_queue, task, 0,
PRIORITY_INSERT_BEGIN,
false,
task->parent_depends_on);
priority_queue_insert (PQ_TEAM, &team->task_queue, task, 0,
PRIORITY_INSERT_END,
false,
task->parent_depends_on);
++team->task_count;
++team->task_queued_count;
gomp_team_barrier_set_task_pending (&team->barrier);
do_wake = team->task_running_count + !parent->in_tied_task
< team->nthreads;
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
gomp_team_barrier_wake (&team->barrier, 1);
return true;
}
static void inline
priority_list_upgrade_task (struct priority_list *list,
struct priority_node *node)
{
struct priority_node *last_parent_depends_on
= list->last_parent_depends_on;
if (last_parent_depends_on)
{
node->prev->next = node->next;
node->next->prev = node->prev;
node->prev = last_parent_depends_on;
node->next = last_parent_depends_on->next;
node->prev->next = node;
node->next->prev = node;
}
else if (node != list->tasks)
{
node->prev->next = node->next;
node->next->prev = node->prev;
node->prev = list->tasks->prev;
node->next = list->tasks;
list->tasks = node;
node->prev->next = node;
node->next->prev = node;
}
list->last_parent_depends_on = node;
}
static void inline
priority_queue_upgrade_task (struct gomp_task *task,
struct gomp_task *parent)
{
struct priority_queue *head = &parent->children_queue;
struct priority_node *node = &task->pnode[PQ_CHILDREN];
#if _LIBGOMP_CHECKING_
if (!task->parent_depends_on)
gomp_fatal ("priority_queue_upgrade_task: task must be a "
"parent_depends_on task");
if (!priority_queue_task_in_queue_p (PQ_CHILDREN, head, task))
gomp_fatal ("priority_queue_upgrade_task: cannot find task=%p", task);
#endif
if (priority_queue_multi_p (head))
{
struct priority_list *list
= priority_queue_lookup_priority (head, task->priority);
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
list->tasks = node->next;
else if (node->next != list->tasks)
{
struct gomp_task *next_task = priority_node_to_task (type, node->next);
if (next_task->kind == GOMP_TASK_WAITING)
{
node->prev->next = node->next;
node->next->prev = node->prev;
node->next = list->tasks;
node->prev = list->tasks->prev;
list->tasks->prev->next = node;
list->tasks->prev = node;
}
}
if (__builtin_expect (child_task->parent_depends_on, 0)
&& list->last_parent_depends_on == node)
{
struct gomp_task *prev_child = priority_node_to_task (type, node->prev);
if (node->prev != node
&& prev_child->kind == GOMP_TASK_WAITING
&& prev_child->parent_depends_on)
list->last_parent_depends_on = node->prev;
else
{
list->last_parent_depends_on = NULL;
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
struct priority_list *list
= priority_queue_lookup_priority (head, task->priority);
priority_list_downgrade_task (type, list, task);
}
else
priority_list_downgrade_task (type, &head->l, task);
}
static inline bool
gomp_task_run_pre (struct gomp_task *child_task, struct gomp_task *parent,
struct gomp_team *team)
{
#if _LIBGOMP_CHECKING_
if (child_task->parent)
priority_queue_verify (PQ_CHILDREN,
&child_task->parent->children_queue, true);
if (child_task->taskgroup)
priority_queue_verify (PQ_TASKGROUP,
&child_task->taskgroup->taskgroup_queue, false);
priority_queue_verify (PQ_TEAM, &team->task_queue, false);
#endif
if (parent)
priority_queue_downgrade_task (PQ_CHILDREN, &parent->children_queue,
child_task);
struct gomp_taskgroup *taskgroup = child_task->taskgroup;
if (taskgroup)
priority_queue_downgrade_task (PQ_TASKGROUP, &taskgroup->taskgroup_queue,
child_task);
priority_queue_remove (PQ_TEAM, &team->task_queue, child_task,
MEMMODEL_RELAXED);
child_task->pnode[PQ_TEAM].next = NULL;
child_task->pnode[PQ_TEAM].prev = NULL;
child_task->kind = GOMP_TASK_TIED;
if (--team->task_queued_count == 0)
gomp_team_barrier_clear_task_pending (&team->barrier);
if ((gomp_team_barrier_cancelled (&team->barrier)
|| (taskgroup && taskgroup->cancelled))
&& !child_task->copy_ctors_done)
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
for (i = 0; i < count; i++)
{
struct gomp_task *task = child_task->dependers->elem[i];
if (--task->num_dependees != 0)
continue;
struct gomp_taskgroup *taskgroup = task->taskgroup;
if (parent)
{
priority_queue_insert (PQ_CHILDREN, &parent->children_queue,
task, task->priority,
PRIORITY_INSERT_BEGIN,
true,
task->parent_depends_on);
if (parent->taskwait)
{
if (parent->taskwait->in_taskwait)
{
parent->taskwait->in_taskwait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
else if (parent->taskwait->in_depend_wait)
{
parent->taskwait->in_depend_wait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
}
}
if (taskgroup)
{
priority_queue_insert (PQ_TASKGROUP, &taskgroup->taskgroup_queue,
task, task->priority,
PRIORITY_INSERT_BEGIN,
false,
task->parent_depends_on);
if (taskgroup->in_taskgroup_wait)
{
taskgroup->in_taskgroup_wait = false;
gomp_sem_post (&taskgroup->taskgroup_sem);
}
}
priority_queue_insert (PQ_TEAM, &team->task_queue,
task, task->priority,
PRIORITY_INSERT_END,
false,
task->parent_depends_on);
++team->task_count;
++team->task_queued_count;
++ret;
}
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
struct gomp_task *parent = child_task->parent;
if (parent == NULL)
return;
if (__builtin_expect (child_task->parent_depends_on, 0)
&& --parent->taskwait->n_depend == 0
&& parent->taskwait->in_depend_wait)
{
parent->taskwait->in_depend_wait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
if (priority_queue_remove (PQ_CHILDREN, &parent->children_queue,
child_task, MEMMODEL_RELEASE)
&& parent->taskwait && parent->taskwait->in_taskwait)
{
parent->taskwait->in_taskwait = false;
gomp_sem_post (&parent->taskwait->taskwait_sem);
}
child_task->pnode[PQ_CHILDREN].next = NULL;
child_task->pnode[PQ_CHILDREN].prev = NULL;
}
static inline void
gomp_task_run_post_remove_taskgroup (struct gomp_task *child_task)
{
struct gomp_taskgroup *taskgroup = child_task->taskgroup;
if (taskgroup == NULL)
return;
bool empty = priority_queue_remove (PQ_TASKGROUP,
&taskgroup->taskgroup_queue,
child_task, MEMMODEL_RELAXED);
child_task->pnode[PQ_TASKGROUP].next = NULL;
child_task->pnode[PQ_TASKGROUP].prev = NULL;
if (taskgroup->num_children > 1)
--taskgroup->num_children;
else
{
__atomic_store_n (&taskgroup->num_children, 0, MEMMODEL_RELEASE);
}
if (empty && taskgroup->in_taskgroup_wait)
{
taskgroup->in_taskgroup_wait = false;
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
gomp_mutex_lock (&team->task_lock);
if (gomp_barrier_last_thread (state))
{
if (team->task_count == 0)
{
gomp_team_barrier_done (&team->barrier, state);
gomp_mutex_unlock (&team->task_lock);
gomp_team_barrier_wake (&team->barrier, 0);
return;
}
gomp_team_barrier_set_waiting_for_tasks (&team->barrier);
}
while (1)
{
bool cancelled = false;
if (!priority_queue_empty_p (&team->task_queue, MEMMODEL_RELAXED))
{
bool ignored;
child_task
= priority_queue_next_task (PQ_TEAM, &team->task_queue,
PQ_IGNORED, NULL,
&ignored);
cancelled = gomp_task_run_pre (child_task, child_task->parent,
team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
team->task_running_count++;
child_task->in_tied_task = true;
}
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->task = child_task;
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
gomp_mutex_lock (&team->task_lock);
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
team->task_running_count--;
struct gomp_target_task *ttask
= (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
child_task->fn (child_task->fn_data);
thr->task = task;
}
else
return;
gomp_mutex_lock (&team->task_lock);
if (child_task)
{
finish_cancelled:;
size_t new_tasks
= gomp_task_run_post_handle_depend (child_task, team);
gomp_task_run_post_remove_parent (child_task);
gomp_clear_parent (&child_task->children_queue);
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
if (--team->task_count == 0
&& gomp_team_barrier_waiting_for_tasks (&team->barrier))
{
gomp_team_barrier_done (&team->barrier, state);
gomp_mutex_unlock (&team->task_lock);
gomp_team_barrier_wake (&team->barrier, 0);
gomp_mutex_lock (&team->task_lock);
}
}
}
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
if (task == NULL
|| priority_queue_empty_p (&task->children_queue, MEMMODEL_ACQUIRE))
return;
memset (&taskwait, 0, sizeof (taskwait));
bool child_q = false;
gomp_mutex_lock (&team->task_lock);
while (1)
{
bool cancelled = false;
if (priority_queue_empty_p (&task->children_queue, MEMMODEL_RELAXED))
{
bool destroy_taskwait = task->taskwait != NULL;
task->taskwait = NULL;
gomp_mutex_unlock (&team->task_lock);
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
}
if (destroy_taskwait)
gomp_sem_destroy (&taskwait.taskwait_sem);
return;
}
struct gomp_task *next_task
= priority_queue_next_task (PQ_CHILDREN, &task->children_queue,
PQ_TEAM, &team->task_queue, &child_q);
if (next_task->kind == GOMP_TASK_WAITING)
{
child_task = next_task;
cancelled
= gomp_task_run_pre (child_task, task, team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
}
else
{
if (task->taskwait == NULL)
{
taskwait.in_depend_wait = false;
gomp_sem_init (&taskwait.taskwait_sem, 0);
task->taskwait = &taskwait;
}
taskwait.in_taskwait = true;
}
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->task = child_task;
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
gomp_mutex_lock (&team->task_lock);
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
struct gomp_target_task *ttask
= (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
child_task->fn (child_task->fn_data);
thr->task = task;
}
else
gomp_sem_wait (&taskwait.taskwait_sem);
gomp_mutex_lock (&team->task_lock);
if (child_task)
{
finish_cancelled:;
size_t new_tasks
= gomp_task_run_post_handle_depend (child_task, team);
if (child_q)
{
priority_queue_remove (PQ_CHILDREN, &task->children_queue,
child_task, MEMMODEL_RELAXED);
child_task->pnode[PQ_CHILDREN].next = NULL;
child_task->pnode[PQ_CHILDREN].prev = NULL;
}
gomp_clear_parent (&child_task->children_queue);
gomp_task_run_post_remove_taskgroup (child_task);
to_free = child_task;
child_task = NULL;
team->task_count--;
if (new_tasks > 1)
{
do_wake = team->nthreads - team->task_running_count
- !task->in_tied_task;
if (do_wake > new_tasks)
do_wake = new_tasks;
}
}
}
}
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
gomp_mutex_lock (&team->task_lock);
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
if (tsk->num_dependees == 0 && tsk->kind == GOMP_TASK_WAITING)
priority_queue_upgrade_task (tsk, task);
}
}
}
if (num_awaited == 0)
{
gomp_mutex_unlock (&team->task_lock);
return;
}
memset (&taskwait, 0, sizeof (taskwait));
taskwait.n_depend = num_awaited;
gomp_sem_init (&taskwait.taskwait_sem, 0);
task->taskwait = &taskwait;
while (1)
{
bool cancelled = false;
if (taskwait.n_depend == 0)
{
task->taskwait = NULL;
gomp_mutex_unlock (&team->task_lock);
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
}
gomp_sem_destroy (&taskwait.taskwait_sem);
return;
}
bool ignored;
struct gomp_task *next_task
= priority_queue_next_task (PQ_CHILDREN, &task->children_queue,
PQ_IGNORED, NULL, &ignored);
if (next_task->kind == GOMP_TASK_WAITING)
{
child_task = next_task;
cancelled
= gomp_task_run_pre (child_task, task, team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
}
else
taskwait.in_depend_wait = true;
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->task = child_task;
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
gomp_mutex_lock (&team->task_lock);
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
struct gomp_target_task *ttask
= (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
child_task->fn (child_task->fn_data);
thr->task = task;
}
else
gomp_sem_wait (&taskwait.taskwait_sem);
gomp_mutex_lock (&team->task_lock);
if (child_task)
{
finish_cancelled:;
size_t new_tasks
= gomp_task_run_post_handle_depend (child_task, team);
if (child_task->parent_depends_on)
--taskwait.n_depend;
priority_queue_remove (PQ_CHILDREN, &task->children_queue,
child_task, MEMMODEL_RELAXED);
child_task->pnode[PQ_CHILDREN].next = NULL;
child_task->pnode[PQ_CHILDREN].prev = NULL;
gomp_clear_parent (&child_task->children_queue);
gomp_task_run_post_remove_taskgroup (child_task);
to_free = child_task;
child_task = NULL;
team->task_count--;
if (new_tasks > 1)
{
do_wake = team->nthreads - team->task_running_count
- !task->in_tied_task;
if (do_wake > new_tasks)
do_wake = new_tasks;
}
}
}
}
void
GOMP_taskyield (void)
{
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
taskgroup = gomp_malloc (sizeof (struct gomp_taskgroup));
taskgroup->prev = task->taskgroup;
priority_queue_init (&taskgroup->taskgroup_queue);
taskgroup->in_taskgroup_wait = false;
taskgroup->cancelled = false;
taskgroup->num_children = 0;
gomp_sem_init (&taskgroup->taskgroup_sem, 0);
task->taskgroup = taskgroup;
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
taskgroup = task->taskgroup;
if (__builtin_expect (taskgroup == NULL, 0)
&& thr->ts.level == 0)
{
gomp_team_barrier_wait (&team->barrier);
return;
}
if (__atomic_load_n (&taskgroup->num_children, MEMMODEL_ACQUIRE) == 0)
goto finish;
bool unused;
gomp_mutex_lock (&team->task_lock);
while (1)
{
bool cancelled = false;
if (priority_queue_empty_p (&taskgroup->taskgroup_queue,
MEMMODEL_RELAXED))
{
if (taskgroup->num_children)
{
if (priority_queue_empty_p (&task->children_queue,
MEMMODEL_RELAXED))
goto do_wait;
child_task
= priority_queue_next_task (PQ_CHILDREN, &task->children_queue,
PQ_TEAM, &team->task_queue,
&unused);
}
else
{
gomp_mutex_unlock (&team->task_lock);
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
}
goto finish;
}
}
else
child_task
= priority_queue_next_task (PQ_TASKGROUP, &taskgroup->taskgroup_queue,
PQ_TEAM, &team->task_queue, &unused);
if (child_task->kind == GOMP_TASK_WAITING)
{
cancelled
= gomp_task_run_pre (child_task, child_task->parent, team);
if (__builtin_expect (cancelled, 0))
{
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
goto finish_cancelled;
}
}
else
{
child_task = NULL;
do_wait:
taskgroup->in_taskgroup_wait = true;
}
gomp_mutex_unlock (&team->task_lock);
if (do_wake)
{
gomp_team_barrier_wake (&team->barrier, do_wake);
do_wake = 0;
}
if (to_free)
{
gomp_finish_task (to_free);
free (to_free);
to_free = NULL;
}
if (child_task)
{
thr->task = child_task;
if (__builtin_expect (child_task->fn == NULL, 0))
{
if (gomp_target_task_fn (child_task->fn_data))
{
thr->task = task;
gomp_mutex_lock (&team->task_lock);
child_task->kind = GOMP_TASK_ASYNC_RUNNING;
struct gomp_target_task *ttask
= (struct gomp_target_task *) child_task->fn_data;
if (ttask->state == GOMP_TARGET_TASK_FINISHED)
gomp_target_task_completion (team, child_task);
else
ttask->state = GOMP_TARGET_TASK_RUNNING;
child_task = NULL;
continue;
}
}
else
child_task->fn (child_task->fn_data);
thr->task = task;
}
else
gomp_sem_wait (&taskgroup->taskgroup_sem);
gomp_mutex_lock (&team->task_lock);
if (child_task)
{
finish_cancelled:;
size_t new_tasks
= gomp_task_run_post_handle_depend (child_task, team);
gomp_task_run_post_remove_parent (child_task);
gomp_clear_parent (&child_task->children_queue);
gomp_task_run_post_remove_taskgroup (child_task);
to_free = child_task;
child_task = NULL;
team->task_count--;
if (new_tasks > 1)
{
do_wake = team->nthreads - team->task_running_count
- !task->in_tied_task;
if (do_wake > new_tasks)
do_wake = new_tasks;
}
}
}
finish:
task->taskgroup = taskgroup->prev;
gomp_sem_destroy (&taskgroup->taskgroup_sem);
free (taskgroup);
}
int
omp_in_final (void)
{
struct gomp_thread *thr = gomp_thread ();
return thr->task && thr->task->final_task;
}
ialias (omp_in_final)
