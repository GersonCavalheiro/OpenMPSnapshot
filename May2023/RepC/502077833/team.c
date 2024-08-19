#include "libgomp.h"
#include "pool.h"
#include <stdlib.h>
#include <string.h>
#ifdef LIBGOMP_USE_PTHREADS
pthread_attr_t gomp_thread_attr;
pthread_key_t gomp_thread_destructor;
#if defined HAVE_TLS || defined USE_EMUTLS
__thread struct gomp_thread gomp_tls_data;
#else
pthread_key_t gomp_tls_key;
#endif
struct gomp_thread_start_data
{
void (*fn) (void *);
void *fn_data;
struct gomp_team_state ts;
struct gomp_task *task;
struct gomp_thread_pool *thread_pool;
unsigned int place;
bool nested;
pthread_t handle;
};
static void *
gomp_thread_start (void *xdata)
{
struct gomp_thread_start_data *data = xdata;
struct gomp_thread *thr;
struct gomp_thread_pool *pool;
void (*local_fn) (void *);
void *local_data;
#if defined HAVE_TLS || defined USE_EMUTLS
thr = &gomp_tls_data;
#else
struct gomp_thread local_thr;
thr = &local_thr;
pthread_setspecific (gomp_tls_key, thr);
#endif
gomp_sem_init (&thr->release, 0);
local_fn = data->fn;
local_data = data->fn_data;
thr->thread_pool = data->thread_pool;
thr->ts = data->ts;
thr->task = data->task;
thr->place = data->place;
#ifdef GOMP_NEEDS_THREAD_HANDLE
thr->handle = data->handle;
#endif
thr->ts.team->ordered_release[thr->ts.team_id] = &thr->release;
pool = thr->thread_pool;
if (data->nested)
{
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
gomp_barrier_wait (&team->barrier);
local_fn (local_data);
gomp_team_barrier_wait_final (&team->barrier);
gomp_finish_task (task);
gomp_barrier_wait_last (&team->barrier);
}
else
{
pool->threads[thr->ts.team_id] = thr;
gomp_simple_barrier_wait (&pool->threads_dock);
do
{
struct gomp_team *team = thr->ts.team;
struct gomp_task *task = thr->task;
local_fn (local_data);
gomp_team_barrier_wait_final (&team->barrier);
gomp_finish_task (task);
gomp_simple_barrier_wait (&pool->threads_dock);
local_fn = thr->fn;
local_data = thr->data;
thr->fn = NULL;
}
while (local_fn);
}
gomp_sem_destroy (&thr->release);
pthread_detach (pthread_self ());
thr->thread_pool = NULL;
thr->task = NULL;
return NULL;
}
#endif
static inline struct gomp_team *
get_last_team (unsigned nthreads)
{
struct gomp_thread *thr = gomp_thread ();
if (thr->ts.team == NULL)
{
struct gomp_thread_pool *pool = gomp_get_thread_pool (thr, nthreads);
struct gomp_team *last_team = pool->last_team;
if (last_team != NULL && last_team->nthreads == nthreads)
{
pool->last_team = NULL;
return last_team;
}
}
return NULL;
}
struct gomp_team *
gomp_new_team (unsigned nthreads)
{
struct gomp_team *team;
int i;
team = get_last_team (nthreads);
if (team == NULL)
{
size_t extra = sizeof (team->ordered_release[0])
+ sizeof (team->implicit_task[0]);
team = gomp_malloc (sizeof (*team) + nthreads * extra);
#ifndef HAVE_SYNC_BUILTINS
gomp_mutex_init (&team->work_share_list_free_lock);
#endif
gomp_barrier_init (&team->barrier, nthreads);
gomp_mutex_init (&team->task_lock);
team->nthreads = nthreads;
}
team->work_share_chunk = 8;
#ifdef HAVE_SYNC_BUILTINS
team->single_count = 0;
#endif
team->work_shares_to_free = &team->work_shares[0];
gomp_init_work_share (&team->work_shares[0], 0, nthreads);
team->work_shares[0].next_alloc = NULL;
team->work_share_list_free = NULL;
team->work_share_list_alloc = &team->work_shares[1];
for (i = 1; i < 7; i++)
team->work_shares[i].next_free = &team->work_shares[i + 1];
team->work_shares[i].next_free = NULL;
gomp_sem_init (&team->master_release, 0);
team->ordered_release = (void *) &team->implicit_task[nthreads];
team->ordered_release[0] = &team->master_release;
priority_queue_init (&team->task_queue);
team->task_count = 0;
team->task_queued_count = 0;
team->task_running_count = 0;
team->work_share_cancelled = 0;
team->team_cancelled = 0;
return team;
}
static void
free_team (struct gomp_team *team)
{
#ifndef HAVE_SYNC_BUILTINS
gomp_mutex_destroy (&team->work_share_list_free_lock);
#endif
gomp_barrier_destroy (&team->barrier);
gomp_mutex_destroy (&team->task_lock);
priority_queue_free (&team->task_queue);
free (team);
}
static void
gomp_free_pool_helper (void *thread_pool)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_thread_pool *pool
= (struct gomp_thread_pool *) thread_pool;
gomp_simple_barrier_wait_last (&pool->threads_dock);
gomp_sem_destroy (&thr->release);
thr->thread_pool = NULL;
thr->task = NULL;
#ifdef LIBGOMP_USE_PTHREADS
pthread_detach (pthread_self ());
pthread_exit (NULL);
#elif defined(__nvptx__)
asm ("exit;");
#else
#error gomp_free_pool_helper must terminate the thread
#endif
}
void
gomp_free_thread (void *arg __attribute__((unused)))
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_thread_pool *pool = thr->thread_pool;
if (pool)
{
if (pool->threads_used > 0)
{
int i;
for (i = 1; i < pool->threads_used; i++)
{
struct gomp_thread *nthr = pool->threads[i];
nthr->fn = gomp_free_pool_helper;
nthr->data = pool;
}
gomp_simple_barrier_wait (&pool->threads_dock);
gomp_simple_barrier_wait (&pool->threads_dock);
gomp_simple_barrier_destroy (&pool->threads_dock);
#ifdef HAVE_SYNC_BUILTINS
__sync_fetch_and_add (&gomp_managed_threads,
1L - pool->threads_used);
#else
gomp_mutex_lock (&gomp_managed_threads_lock);
gomp_managed_threads -= pool->threads_used - 1L;
gomp_mutex_unlock (&gomp_managed_threads_lock);
#endif
}
if (pool->last_team)
free_team (pool->last_team);
#ifndef __nvptx__
free (pool->threads);
free (pool);
#endif
thr->thread_pool = NULL;
}
if (thr->ts.level == 0 && __builtin_expect (thr->ts.team != NULL, 0))
gomp_team_end ();
if (thr->task != NULL)
{
struct gomp_task *task = thr->task;
gomp_end_task ();
free (task);
}
}
#ifdef LIBGOMP_USE_PTHREADS
void
gomp_team_start (void (*fn) (void *), void *data, unsigned nthreads,
unsigned flags, struct gomp_team *team,
struct gomp_taskgroup *taskgroup)
{
struct gomp_thread_start_data *start_data;
struct gomp_thread *thr, *nthr;
struct gomp_task *task;
struct gomp_task_icv *icv;
bool nested;
struct gomp_thread_pool *pool;
unsigned i, n, old_threads_used = 0;
pthread_attr_t thread_attr, *attr;
unsigned long nthreads_var;
char bind, bind_var;
unsigned int s = 0, rest = 0, p = 0, k = 0;
unsigned int affinity_count = 0;
struct gomp_thread **affinity_thr = NULL;
bool force_display = false;
thr = gomp_thread ();
nested = thr->ts.level;
pool = thr->thread_pool;
task = thr->task;
icv = task ? &task->icv : &gomp_global_icv;
if (__builtin_expect (gomp_places_list != NULL, 0) && thr->place == 0)
{
gomp_init_affinity ();
if (__builtin_expect (gomp_display_affinity_var, 0) && nthreads == 1)
gomp_display_affinity_thread (gomp_thread_self (), &thr->ts,
thr->place);
}
team->prev_ts = thr->ts;
thr->ts.team = team;
thr->ts.team_id = 0;
++thr->ts.level;
if (nthreads > 1)
++thr->ts.active_level;
thr->ts.work_share = &team->work_shares[0];
thr->ts.last_work_share = NULL;
#ifdef HAVE_SYNC_BUILTINS
thr->ts.single_count = 0;
#endif
thr->ts.static_trip = 0;
thr->task = &team->implicit_task[0];
#ifdef GOMP_NEEDS_THREAD_HANDLE
thr->handle = pthread_self ();
#endif
nthreads_var = icv->nthreads_var;
if (__builtin_expect (gomp_nthreads_var_list != NULL, 0)
&& thr->ts.level < gomp_nthreads_var_list_len)
nthreads_var = gomp_nthreads_var_list[thr->ts.level];
bind_var = icv->bind_var;
if (bind_var != omp_proc_bind_false && (flags & 7) != omp_proc_bind_false)
bind_var = flags & 7;
bind = bind_var;
if (__builtin_expect (gomp_bind_var_list != NULL, 0)
&& thr->ts.level < gomp_bind_var_list_len)
bind_var = gomp_bind_var_list[thr->ts.level];
gomp_init_task (thr->task, task, icv);
thr->task->taskgroup = taskgroup;
team->implicit_task[0].icv.nthreads_var = nthreads_var;
team->implicit_task[0].icv.bind_var = bind_var;
if (nthreads == 1)
return;
i = 1;
if (__builtin_expect (gomp_places_list != NULL, 0))
{
p = thr->place - 1;
switch (bind)
{
case omp_proc_bind_true:
case omp_proc_bind_close:
if (nthreads > thr->ts.place_partition_len)
{
s = nthreads / thr->ts.place_partition_len;
rest = nthreads % thr->ts.place_partition_len;
}
else
s = 1;
k = 1;
break;
case omp_proc_bind_master:
break;
case omp_proc_bind_spread:
if (nthreads <= thr->ts.place_partition_len)
{
s = thr->ts.place_partition_len / nthreads;
rest = thr->ts.place_partition_len % nthreads;
rest = (s + 1) * rest + thr->ts.place_partition_off;
if (p < rest)
{
p -= (p - thr->ts.place_partition_off) % (s + 1);
thr->ts.place_partition_len = s + 1;
}
else
{
p -= (p - rest) % s;
thr->ts.place_partition_len = s;
}
thr->ts.place_partition_off = p;
}
else
{
s = nthreads / thr->ts.place_partition_len;
rest = nthreads % thr->ts.place_partition_len;
thr->ts.place_partition_off = p;
thr->ts.place_partition_len = 1;
k = 1;
}
break;
}
}
else
bind = omp_proc_bind_false;
if (!nested)
{
old_threads_used = pool->threads_used;
if (nthreads <= old_threads_used)
n = nthreads;
else if (old_threads_used == 0)
{
n = 0;
gomp_simple_barrier_init (&pool->threads_dock, nthreads);
}
else
{
n = old_threads_used;
gomp_simple_barrier_reinit (&pool->threads_dock, nthreads);
}
pool->threads_used = nthreads;
if (nthreads >= pool->threads_size)
{
pool->threads_size = nthreads + 1;
pool->threads
= gomp_realloc (pool->threads,
pool->threads_size
* sizeof (struct gomp_thread *));
pool->threads[0] = thr;
}
for (; i < n; ++i)
{
unsigned int place_partition_off = thr->ts.place_partition_off;
unsigned int place_partition_len = thr->ts.place_partition_len;
unsigned int place = 0;
if (__builtin_expect (gomp_places_list != NULL, 0))
{
switch (bind)
{
case omp_proc_bind_true:
case omp_proc_bind_close:
if (k == s)
{
++p;
if (p == (team->prev_ts.place_partition_off
+ team->prev_ts.place_partition_len))
p = team->prev_ts.place_partition_off;
k = 1;
if (i == nthreads - rest)
s = 1;
}
else
++k;
break;
case omp_proc_bind_master:
break;
case omp_proc_bind_spread:
if (k == 0)
{
if (p < rest)
p += s + 1;
else
p += s;
if (p == (team->prev_ts.place_partition_off
+ team->prev_ts.place_partition_len))
p = team->prev_ts.place_partition_off;
place_partition_off = p;
if (p < rest)
place_partition_len = s + 1;
else
place_partition_len = s;
}
else
{
if (k == s)
{
++p;
if (p == (team->prev_ts.place_partition_off
+ team->prev_ts.place_partition_len))
p = team->prev_ts.place_partition_off;
k = 1;
if (i == nthreads - rest)
s = 1;
}
else
++k;
place_partition_off = p;
place_partition_len = 1;
}
break;
}
if (affinity_thr != NULL
|| (bind != omp_proc_bind_true
&& pool->threads[i]->place != p + 1)
|| pool->threads[i]->place <= place_partition_off
|| pool->threads[i]->place > (place_partition_off
+ place_partition_len))
{
unsigned int l;
force_display = true;
if (affinity_thr == NULL)
{
unsigned int j;
if (team->prev_ts.place_partition_len > 64)
affinity_thr
= gomp_malloc (team->prev_ts.place_partition_len
* sizeof (struct gomp_thread *));
else
affinity_thr
= gomp_alloca (team->prev_ts.place_partition_len
* sizeof (struct gomp_thread *));
memset (affinity_thr, '\0',
team->prev_ts.place_partition_len
* sizeof (struct gomp_thread *));
for (j = i; j < old_threads_used; j++)
{
if (pool->threads[j]->place
> team->prev_ts.place_partition_off
&& (pool->threads[j]->place
<= (team->prev_ts.place_partition_off
+ team->prev_ts.place_partition_len)))
{
l = pool->threads[j]->place - 1
- team->prev_ts.place_partition_off;
pool->threads[j]->data = affinity_thr[l];
affinity_thr[l] = pool->threads[j];
}
pool->threads[j] = NULL;
}
if (nthreads > old_threads_used)
memset (&pool->threads[old_threads_used],
'\0', ((nthreads - old_threads_used)
* sizeof (struct gomp_thread *)));
n = nthreads;
affinity_count = old_threads_used - i;
}
if (affinity_count == 0)
break;
l = p;
if (affinity_thr[l - team->prev_ts.place_partition_off]
== NULL)
{
if (bind != omp_proc_bind_true)
continue;
for (l = place_partition_off;
l < place_partition_off + place_partition_len;
l++)
if (affinity_thr[l - team->prev_ts.place_partition_off]
!= NULL)
break;
if (l == place_partition_off + place_partition_len)
continue;
}
nthr = affinity_thr[l - team->prev_ts.place_partition_off];
affinity_thr[l - team->prev_ts.place_partition_off]
= (struct gomp_thread *) nthr->data;
affinity_count--;
pool->threads[i] = nthr;
}
else
nthr = pool->threads[i];
place = p + 1;
}
else
nthr = pool->threads[i];
nthr->ts.team = team;
nthr->ts.work_share = &team->work_shares[0];
nthr->ts.last_work_share = NULL;
nthr->ts.team_id = i;
nthr->ts.level = team->prev_ts.level + 1;
nthr->ts.active_level = thr->ts.active_level;
nthr->ts.place_partition_off = place_partition_off;
nthr->ts.place_partition_len = place_partition_len;
#ifdef HAVE_SYNC_BUILTINS
nthr->ts.single_count = 0;
#endif
nthr->ts.static_trip = 0;
nthr->task = &team->implicit_task[i];
nthr->place = place;
gomp_init_task (nthr->task, task, icv);
team->implicit_task[i].icv.nthreads_var = nthreads_var;
team->implicit_task[i].icv.bind_var = bind_var;
nthr->task->taskgroup = taskgroup;
nthr->fn = fn;
nthr->data = data;
team->ordered_release[i] = &nthr->release;
}
if (__builtin_expect (affinity_thr != NULL, 0))
{
if (nthreads <= old_threads_used
? (affinity_count == old_threads_used - nthreads)
: (i == old_threads_used))
{
if (team->prev_ts.place_partition_len > 64)
free (affinity_thr);
affinity_thr = NULL;
affinity_count = 0;
}
else
{
i = 1;
switch (bind)
{
case omp_proc_bind_true:
case omp_proc_bind_close:
if (nthreads > thr->ts.place_partition_len)
s = nthreads / thr->ts.place_partition_len;
k = 1;
p = thr->place - 1;
break;
case omp_proc_bind_master:
break;
case omp_proc_bind_spread:
p = thr->ts.place_partition_off;
if (k != 0)
{
s = nthreads / team->prev_ts.place_partition_len;
k = 1;
}
break;
}
if (affinity_count)
gomp_simple_barrier_reinit (&pool->threads_dock,
nthreads + affinity_count);
}
}
if (i == nthreads)
goto do_release;
}
if (__builtin_expect (nthreads + affinity_count > old_threads_used, 0))
{
long diff = (long) (nthreads + affinity_count) - (long) old_threads_used;
if (old_threads_used == 0)
--diff;
#ifdef HAVE_SYNC_BUILTINS
__sync_fetch_and_add (&gomp_managed_threads, diff);
#else
gomp_mutex_lock (&gomp_managed_threads_lock);
gomp_managed_threads += diff;
gomp_mutex_unlock (&gomp_managed_threads_lock);
#endif
}
attr = &gomp_thread_attr;
if (__builtin_expect (gomp_places_list != NULL, 0))
{
size_t stacksize;
pthread_attr_init (&thread_attr);
if (! pthread_attr_getstacksize (&gomp_thread_attr, &stacksize))
pthread_attr_setstacksize (&thread_attr, stacksize);
attr = &thread_attr;
}
start_data = gomp_alloca (sizeof (struct gomp_thread_start_data)
* (nthreads - i));
for (; i < nthreads; ++i)
{
int err;
start_data->ts.place_partition_off = thr->ts.place_partition_off;
start_data->ts.place_partition_len = thr->ts.place_partition_len;
start_data->place = 0;
if (__builtin_expect (gomp_places_list != NULL, 0))
{
switch (bind)
{
case omp_proc_bind_true:
case omp_proc_bind_close:
if (k == s)
{
++p;
if (p == (team->prev_ts.place_partition_off
+ team->prev_ts.place_partition_len))
p = team->prev_ts.place_partition_off;
k = 1;
if (i == nthreads - rest)
s = 1;
}
else
++k;
break;
case omp_proc_bind_master:
break;
case omp_proc_bind_spread:
if (k == 0)
{
if (p < rest)
p += s + 1;
else
p += s;
if (p == (team->prev_ts.place_partition_off
+ team->prev_ts.place_partition_len))
p = team->prev_ts.place_partition_off;
start_data->ts.place_partition_off = p;
if (p < rest)
start_data->ts.place_partition_len = s + 1;
else
start_data->ts.place_partition_len = s;
}
else
{
if (k == s)
{
++p;
if (p == (team->prev_ts.place_partition_off
+ team->prev_ts.place_partition_len))
p = team->prev_ts.place_partition_off;
k = 1;
if (i == nthreads - rest)
s = 1;
}
else
++k;
start_data->ts.place_partition_off = p;
start_data->ts.place_partition_len = 1;
}
break;
}
start_data->place = p + 1;
if (affinity_thr != NULL && pool->threads[i] != NULL)
continue;
gomp_init_thread_affinity (attr, p);
}
start_data->fn = fn;
start_data->fn_data = data;
start_data->ts.team = team;
start_data->ts.work_share = &team->work_shares[0];
start_data->ts.last_work_share = NULL;
start_data->ts.team_id = i;
start_data->ts.level = team->prev_ts.level + 1;
start_data->ts.active_level = thr->ts.active_level;
#ifdef HAVE_SYNC_BUILTINS
start_data->ts.single_count = 0;
#endif
start_data->ts.static_trip = 0;
start_data->task = &team->implicit_task[i];
gomp_init_task (start_data->task, task, icv);
team->implicit_task[i].icv.nthreads_var = nthreads_var;
team->implicit_task[i].icv.bind_var = bind_var;
start_data->task->taskgroup = taskgroup;
start_data->thread_pool = pool;
start_data->nested = nested;
attr = gomp_adjust_thread_attr (attr, &thread_attr);
err = pthread_create (&start_data->handle, attr, gomp_thread_start,
start_data);
start_data++;
if (err != 0)
gomp_fatal ("Thread creation failed: %s", strerror (err));
}
if (__builtin_expect (attr == &thread_attr, 0))
pthread_attr_destroy (&thread_attr);
do_release:
if (nested)
gomp_barrier_wait (&team->barrier);
else
gomp_simple_barrier_wait (&pool->threads_dock);
if (__builtin_expect (nthreads < old_threads_used, 0)
|| __builtin_expect (affinity_count, 0))
{
long diff = (long) nthreads - (long) old_threads_used;
if (affinity_count)
diff = -affinity_count;
gomp_simple_barrier_reinit (&pool->threads_dock, nthreads);
#ifdef HAVE_SYNC_BUILTINS
__sync_fetch_and_add (&gomp_managed_threads, diff);
#else
gomp_mutex_lock (&gomp_managed_threads_lock);
gomp_managed_threads += diff;
gomp_mutex_unlock (&gomp_managed_threads_lock);
#endif
}
if (__builtin_expect (gomp_display_affinity_var, 0))
{
if (nested
|| nthreads != old_threads_used
|| force_display)
{
gomp_display_affinity_thread (gomp_thread_self (), &thr->ts,
thr->place);
if (nested)
{
start_data -= nthreads - 1;
for (i = 1; i < nthreads; ++i)
{
gomp_display_affinity_thread (
#ifdef LIBGOMP_USE_PTHREADS
start_data->handle,
#else
gomp_thread_self (),
#endif
&start_data->ts,
start_data->place);
start_data++;
}
}
else
{
for (i = 1; i < nthreads; ++i)
{
gomp_thread_handle handle
= gomp_thread_to_pthread_t (pool->threads[i]);
gomp_display_affinity_thread (handle, &pool->threads[i]->ts,
pool->threads[i]->place);
}
}
}
}
if (__builtin_expect (affinity_thr != NULL, 0)
&& team->prev_ts.place_partition_len > 64)
free (affinity_thr);
}
#endif
void
gomp_team_end (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_team *team = thr->ts.team;
gomp_team_barrier_wait_final (&team->barrier);
if (__builtin_expect (team->team_cancelled, 0))
{
struct gomp_work_share *ws = team->work_shares_to_free;
do
{
struct gomp_work_share *next_ws = gomp_ptrlock_get (&ws->next_ws);
if (next_ws == NULL)
gomp_ptrlock_set (&ws->next_ws, ws);
gomp_fini_work_share (ws);
ws = next_ws;
}
while (ws != NULL);
}
else
gomp_fini_work_share (thr->ts.work_share);
gomp_end_task ();
thr->ts = team->prev_ts;
if (__builtin_expect (thr->ts.level != 0, 0))
{
#ifdef HAVE_SYNC_BUILTINS
__sync_fetch_and_add (&gomp_managed_threads, 1L - team->nthreads);
#else
gomp_mutex_lock (&gomp_managed_threads_lock);
gomp_managed_threads -= team->nthreads - 1L;
gomp_mutex_unlock (&gomp_managed_threads_lock);
#endif
gomp_barrier_wait (&team->barrier);
}
if (__builtin_expect (team->work_shares[0].next_alloc != NULL, 0))
{
struct gomp_work_share *ws = team->work_shares[0].next_alloc;
do
{
struct gomp_work_share *next_ws = ws->next_alloc;
free (ws);
ws = next_ws;
}
while (ws != NULL);
}
gomp_sem_destroy (&team->master_release);
if (__builtin_expect (thr->ts.team != NULL, 0)
|| __builtin_expect (team->nthreads == 1, 0))
free_team (team);
else
{
struct gomp_thread_pool *pool = thr->thread_pool;
if (pool->last_team)
free_team (pool->last_team);
pool->last_team = team;
gomp_release_thread_pool (pool);
}
}
#ifdef LIBGOMP_USE_PTHREADS
static void __attribute__((constructor))
initialize_team (void)
{
#if !defined HAVE_TLS && !defined USE_EMUTLS
static struct gomp_thread initial_thread_tls_data;
pthread_key_create (&gomp_tls_key, NULL);
pthread_setspecific (gomp_tls_key, &initial_thread_tls_data);
#endif
if (pthread_key_create (&gomp_thread_destructor, gomp_free_thread) != 0)
gomp_fatal ("could not create thread pool destructor.");
}
static void __attribute__((destructor))
team_destructor (void)
{
lib_destructor();
pthread_key_delete (gomp_thread_destructor);
}
static void
gomp_pause_pool_helper (void *thread_pool)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_thread_pool *pool
= (struct gomp_thread_pool *) thread_pool;
gomp_simple_barrier_wait_last (&pool->threads_dock);
gomp_sem_destroy (&thr->release);
thr->thread_pool = NULL;
thr->task = NULL;
pthread_exit (NULL);
}
int
gomp_pause_host (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_thread_pool *pool = thr->thread_pool;
if (thr->ts.level)
return -1;
if (pool)
{
if (pool->threads_used > 0)
{
int i;
pthread_t *thrs
= gomp_alloca (sizeof (pthread_t) * pool->threads_used);
for (i = 1; i < pool->threads_used; i++)
{
struct gomp_thread *nthr = pool->threads[i];
nthr->fn = gomp_pause_pool_helper;
nthr->data = pool;
thrs[i] = gomp_thread_to_pthread_t (nthr);
}
gomp_simple_barrier_wait (&pool->threads_dock);
gomp_simple_barrier_wait (&pool->threads_dock);
gomp_simple_barrier_destroy (&pool->threads_dock);
#ifdef HAVE_SYNC_BUILTINS
__sync_fetch_and_add (&gomp_managed_threads,
1L - pool->threads_used);
#else
gomp_mutex_lock (&gomp_managed_threads_lock);
gomp_managed_threads -= pool->threads_used - 1L;
gomp_mutex_unlock (&gomp_managed_threads_lock);
#endif
for (i = 1; i < pool->threads_used; i++)
pthread_join (thrs[i], NULL);
}
if (pool->last_team)
free_team (pool->last_team);
#ifndef __nvptx__
free (pool->threads);
free (pool);
#endif
thr->thread_pool = NULL;
}
return 0;
}
#endif
struct gomp_task_icv *
gomp_new_icv (void)
{
struct gomp_thread *thr = gomp_thread ();
struct gomp_task *task = gomp_malloc (sizeof (struct gomp_task));
gomp_init_task (task, NULL, &gomp_global_icv);
thr->task = task;
#ifdef LIBGOMP_USE_PTHREADS
pthread_setspecific (gomp_thread_destructor, thr);
#endif
return &task->icv;
}
