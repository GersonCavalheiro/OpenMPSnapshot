#include "apis/dlb_talp.h"
#include "apis/dlb_errors.h"
#include "LB_core/spd.h"
#include "LB_core/DLB_talp.h"
#include "LB_comm/shmem_cpuinfo.h"
#include "LB_comm/shmem_procinfo.h"
#include "support/debug.h"
#include "support/mask_utils.h"
#include "support/mytime.h"
#pragma GCC visibility push(default)
int DLB_TALP_Attach(void) {
char shm_key[MAX_OPTION_LENGTH];
options_parse_entry("--shm-key", &shm_key);
shmem_cpuinfo_ext__init(shm_key);
shmem_procinfo_ext__init(shm_key);
return DLB_SUCCESS;
}
int DLB_TALP_Detach(void) {
int error = shmem_cpuinfo_ext__finalize();
error = error ? error : shmem_procinfo_ext__finalize();
return error;
}
int DLB_TALP_GetNumCPUs(int *ncpus) {
*ncpus = shmem_cpuinfo_ext__getnumcpus();
return DLB_SUCCESS;
}
int DLB_TALP_GetPidList(int *pidlist, int *nelems, int max_len) {
return shmem_procinfo__getpidlist(pidlist, nelems, max_len);
}
int DLB_TALP_GetTimes(int pid, double *mpi_time, double *useful_time) {
int64_t mpi_time_ns;
int64_t useful_time_ns;
int error = shmem_procinfo__get_app_times(pid, &mpi_time_ns, &useful_time_ns);
if (error == DLB_SUCCESS) {
*mpi_time = nsecs_to_secs(mpi_time_ns);
*useful_time = nsecs_to_secs(useful_time_ns);
}
return error;
}
const dlb_monitor_t* DLB_MonitoringRegionGetMPIRegion(void) {
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return NULL;
}
return monitoring_region_get_MPI_region(thread_spd);
}
dlb_monitor_t* DLB_MonitoringRegionRegister(const char *name){
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return NULL;
}
return monitoring_region_register(name);
}
int DLB_MonitoringRegionReset(dlb_monitor_t *handle){
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return DLB_ERR_NOTALP;
}
return monitoring_region_reset(handle);
}
int DLB_MonitoringRegionStart(dlb_monitor_t *handle){
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return DLB_ERR_NOTALP;
}
return monitoring_region_start(thread_spd, handle);
}
int DLB_MonitoringRegionStop(dlb_monitor_t *handle){
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return DLB_ERR_NOTALP;
}
return monitoring_region_stop(thread_spd, handle);
}
int DLB_MonitoringRegionReport(const dlb_monitor_t *handle){
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return DLB_ERR_NOTALP;
}
return monitoring_region_report(thread_spd, handle);
}
int DLB_MonitoringRegionsUpdate(void) {
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return DLB_ERR_NOTALP;
}
return monitoring_regions_force_update(thread_spd);
}
int DLB_TALP_CollectPOPMetrics(dlb_monitor_t *monitor, dlb_pop_metrics_t *pop_metrics) {
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return DLB_ERR_NOTALP;
}
return talp_collect_pop_metrics(thread_spd, monitor, pop_metrics);
}
int DLB_TALP_CollectNodeMetrics(dlb_monitor_t *monitor, dlb_node_metrics_t *node_metrics) {
spd_enter_dlb(NULL);
if (unlikely(!thread_spd->talp_info)) {
return DLB_ERR_NOTALP;
}
if (unlikely(!thread_spd->options.barrier)) {
return DLB_ERR_NOCOMP;
}
return talp_collect_node_metrics(thread_spd, monitor, node_metrics);
}
