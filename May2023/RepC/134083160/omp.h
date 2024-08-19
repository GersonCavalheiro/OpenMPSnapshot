#ifndef __OMP_H__
#define __OMP_H__
#ifndef _OPENMP
#define _OPENMP 201511
#endif
#ifdef __cplusplus
extern "C" {
#endif
#pragma omp declare target
int  omp_in_parallel(void);
int  omp_get_thread_num(void);
void omp_set_num_threads(int num_threads);
int  omp_get_num_threads(void);
int  omp_get_max_threads(void);
int  omp_get_num_procs(void);
void omp_set_dynamic(int dynamic_threads);
int  omp_get_dynamic(void);
void omp_set_nested(int nested);
int  omp_get_nested(void);
typedef enum omp_sched_t
{
omp_sched_static = 1,
omp_sched_dynamic = 2,
omp_sched_guided = 3,
omp_sched_auto = 4
} omp_sched_t;
typedef enum omp_proc_bind_t
{
omp_proc_bind_false = 0,
omp_proc_bind_true = 1,
omp_proc_bind_primary = 2,
omp_proc_bind_master = omp_proc_bind_primary, 
omp_proc_bind_close = 3,
omp_proc_bind_spread = 4
} omp_proc_bind_t;
typedef enum omp_lock_hint_t 
{
omp_lock_hint_none = 0,
omp_lock_hint_uncontended = 1,
omp_lock_hint_contended = 2,
omp_lock_hint_nonspeculative = 4,
omp_lock_hint_speculative = 8
} omp_lock_hint_t;
typedef void *omp_lock_t;
void omp_init_lock(omp_lock_t *lock);
void omp_init_lock_with_hint(omp_lock_t *lock, omp_lock_hint_t hint);
void omp_destroy_lock(omp_lock_t *lock);
void omp_set_lock(omp_lock_t *lock);
void omp_unset_lock(omp_lock_t *lock);
int  omp_test_lock(omp_lock_t *lock);
typedef void *omp_nest_lock_t;
void omp_init_nest_lock(omp_nest_lock_t *lock);
void omp_init_next_lock_with_hint(omp_nest_lock_t *lock, omp_lock_hint_t hint);
void omp_destroy_nest_lock(omp_nest_lock_t *lock);
void omp_set_nest_lock(omp_nest_lock_t *lock);
void omp_unset_nest_lock(omp_nest_lock_t *lock);
int  omp_test_nest_lock(omp_nest_lock_t *lock);
double omp_get_wtime(void);
double omp_get_wtick(void);
void omp_set_schedule(omp_sched_t kind, int chunk);
void omp_get_schedule(omp_sched_t *kind, int *chunk);
int  omp_get_thread_limit(void);
void omp_set_max_active_levels(int levels);
int  omp_get_max_active_levels(void);
int  omp_get_level(void);
int  omp_get_ancestor_thread_num(int level);
int  omp_get_team_size(int level);
int  omp_get_active_level(void);
int  omp_in_final(void);
int             omp_get_cancellation(void);
omp_proc_bind_t omp_get_proc_bind(void);
int             omp_get_num_teams(void);
int             omp_get_team_num(void);
int             omp_is_initial_device(void);
int  omp_get_max_task_priority(void);
int  omp_get_num_places(void);
int  omp_get_place_num_procs(int place_num);
void omp_get_place_proc_ids(int place_num, int *ids);
int  omp_get_place_num(void);
int  omp_get_partition_num_places(void);
void omp_get_partition_place_nums(int *place_nums);
void omp_set_default_device(int device_num);
int  omp_get_default_device(void);
int  omp_get_initial_device(void);
int  omp_get_num_devices(void);
void          omp_set_affinity_format(const char *format);
unsigned long omp_get_affinity_format(char *buffer, unsigned long size);
void          omp_display_affinity(const char *format);
unsigned long omp_capture_affinity(char *buffer, unsigned long size,
const char *format);
#pragma omp end declare target
int  omp_target_is_present(void *ptr, int devid);
void *omp_target_alloc(unsigned long size, int devid);
void omp_target_free(void *devaddr, int devid);
int  omp_target_memcpy(void *dst, void *src, unsigned long length, 
unsigned long dst_off, unsigned long src_off, 
int dst_devid, int src_devid);
int  omp_target_memcpy_rect(void *dst, void *src, unsigned long elemsize,
int numdims, unsigned long *volume,
unsigned long *dst_offs, unsigned long *src_offs,
unsigned long *dst_dims, unsigned long *src_dims,
int dst_devid, int src_devid);
int omp_target_associate_ptr(void *hostptr, void *devptr, 
unsigned long size, unsigned long devoff, int devid);
int omp_target_disassociate_ptr(void *hostptr, int devid);
#ifdef __cplusplus
}
#endif
#endif
