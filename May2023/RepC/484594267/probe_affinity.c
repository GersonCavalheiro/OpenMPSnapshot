#include "omp.h"
#include "tackle.h"
void probe_affinity(int wid, machine_info_t *p_mach_info,
worker_opts_t *p_workers, int *p_global_affinity)
{
int ncores = p_mach_info->total_log_cores;
int worker_offset = 0;
for (int w = wid; w > 0; w--) {
worker_offset +=
p_workers[w - 1].num_threads * p_mach_info->total_log_cores;
}
setenv("KMP_AFFINITY", p_workers[wid].affinity_str, 1);
#pragma omp parallel num_threads(p_workers[wid].num_threads)
{
int ithr = omp_get_thread_num();
int *p_ithr_affinity = p_global_affinity + worker_offset + ithr * ncores;
cpu_set_t mask;
int status = sched_getaffinity(0, sizeof(cpu_set_t), &mask);
if (status) {
printf("ERROR: TID-%d sched_getaffinity() returned %d\n", ithr, status);
fflush(0);
}
#ifdef DEBUG
printf("\nWID-%d TID-%d is pinned to:", wid, ithr);
fflush(0);
#endif
for (int i = 0; i < ncores; i++) {
p_ithr_affinity[i] = ((CPU_ISSET(i, &mask)) ? i : -1);
#ifdef DEBUG
if (p_ithr_affinity[i] != -1) {
printf(" %d", i);
fflush(0);
}
#endif
}
}
}
