#include "apis/dlb_sp.h"
#include "LB_core/spd.h"
#include "LB_core/DLB_kernel.h"
#include "support/atomic.h"
#include <stdlib.h>
#include <unistd.h>
static int get_atomic_spid(void) {
enum { SPID_INCREMENT = 100000 };
static atomic_int spid_seed = 0;
return DLB_ATOMIC_ADD_FETCH_RLX(&spid_seed, SPID_INCREMENT) + getpid();
}
#pragma GCC visibility push(default)
dlb_handler_t DLB_Init_sp(int ncpus, const_dlb_cpu_set_t mask, const char *dlb_args) {
subprocess_descriptor_t *spd = malloc(sizeof(subprocess_descriptor_t));
spd->dlb_initialized = true;
spd_register(spd);
spd_enter_dlb(spd);
int error = Initialize(spd, get_atomic_spid(), ncpus, mask, dlb_args);
if (error != DLB_SUCCESS) {
spd_unregister(spd);
free(spd);
return NULL;
}
return spd;
}
int DLB_Finalize_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
int error = Finish(handler);
spd_unregister(handler);
free(handler);
return error;
}
int DLB_Enable_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return set_lewi_enabled(handler, true);
}
int DLB_Disable_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return set_lewi_enabled(handler, false);
}
int DLB_SetMaxParallelism_sp(dlb_handler_t handler, int max) {
spd_enter_dlb(handler);
return set_max_parallelism(handler, max);
}
int DLB_UnsetMaxParallelism_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return unset_max_parallelism(handler);
}
int DLB_CallbackSet_sp(dlb_handler_t handler, dlb_callbacks_t which,
dlb_callback_t callback, void *arg) {
spd_enter_dlb(handler);
pm_interface_t *pm = &((subprocess_descriptor_t*)handler)->pm;
return pm_callback_set(pm, which, callback, arg);
}
int DLB_CallbackGet_sp(dlb_handler_t handler, dlb_callbacks_t which,
dlb_callback_t *callback, void **arg) {
spd_enter_dlb(handler);
const pm_interface_t *pm = &((subprocess_descriptor_t*)handler)->pm;
return pm_callback_get(pm, which, callback, arg);
}
int DLB_Lend_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return lend(handler);
}
int DLB_LendCpu_sp(dlb_handler_t handler, int cpuid) {
spd_enter_dlb(handler);
return lend_cpu(handler, cpuid);
}
int DLB_LendCpus_sp(dlb_handler_t handler, int ncpus) {
spd_enter_dlb(handler);
return lend_cpus(handler, ncpus);
}
int DLB_LendCpuMask_sp(dlb_handler_t handler, const_dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return lend_cpu_mask(handler, mask);
}
int DLB_Reclaim_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return reclaim(handler);
}
int DLB_ReclaimCpu_sp(dlb_handler_t handler, int cpuid) {
spd_enter_dlb(handler);
return reclaim_cpu(handler, cpuid);
}
int DLB_ReclaimCpus_sp(dlb_handler_t handler, int ncpus) {
spd_enter_dlb(handler);
return reclaim_cpus(handler, ncpus);
}
int DLB_ReclaimCpuMask_sp(dlb_handler_t handler, const_dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return reclaim_cpu_mask(handler, mask);
}
int DLB_AcquireCpu_sp(dlb_handler_t handler, int cpuid) {
spd_enter_dlb(handler);
return acquire_cpu(handler, cpuid);
}
int DLB_AcquireCpus_sp(dlb_handler_t handler, int ncpus) {
spd_enter_dlb(handler);
return acquire_cpus(handler, ncpus);
}
int DLB_AcquireCpuMask_sp(dlb_handler_t handler, const_dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return acquire_cpu_mask(handler, mask);
}
int DLB_AcquireCpusInMask_sp(dlb_handler_t handler, int ncpus, const_dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return acquire_cpus_in_mask(handler, ncpus, mask);
}
int DLB_Borrow_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return borrow(handler);
}
int DLB_BorrowCpu_sp(dlb_handler_t handler, int cpuid) {
spd_enter_dlb(handler);
return borrow_cpu(handler, cpuid);
}
int DLB_BorrowCpus_sp(dlb_handler_t handler, int ncpus) {
spd_enter_dlb(handler);
return borrow_cpus(handler, ncpus);
}
int DLB_BorrowCpuMask_sp(dlb_handler_t handler, const_dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return borrow_cpu_mask(handler, mask);
}
int DLB_BorrowCpusInMask_sp(dlb_handler_t handler, int ncpus, const_dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return borrow_cpus_in_mask(handler, ncpus, mask);
}
int DLB_Return_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return return_all(handler);
}
int DLB_ReturnCpu_sp(dlb_handler_t handler, int cpuid) {
spd_enter_dlb(handler);
return return_cpu(handler, cpuid);
}
int DLB_ReturnCpuMask_sp(dlb_handler_t handler, const_dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return return_cpu_mask(handler, mask);
}
int DLB_PollDROM_sp(dlb_handler_t handler, int *ncpus, dlb_cpu_set_t mask) {
spd_enter_dlb(handler);
return poll_drom(handler, ncpus, mask);
}
int DLB_PollDROM_Update_sp(dlb_handler_t handler) {
spd_enter_dlb(handler);
return poll_drom_update(handler);
}
int DLB_CheckCpuAvailability_sp(dlb_handler_t handler, int cpuid) {
spd_enter_dlb(handler);
return check_cpu_availability(handler, cpuid);
}
int DLB_SetVariable_sp(dlb_handler_t handler, const char *variable, const char *value) {
spd_enter_dlb(handler);
options_t *options = &((subprocess_descriptor_t*)handler)->options;
return options_set_variable(options, variable, value);
}
int DLB_GetVariable_sp(dlb_handler_t handler, const char *variable, char *value) {
spd_enter_dlb(handler);
const options_t *options = &((subprocess_descriptor_t*)handler)->options;
return options_get_variable(options, variable, value);
}
int DLB_PrintVariables_sp(dlb_handler_t handler, int print_extended) {
spd_enter_dlb(handler);
const options_t *options = &((subprocess_descriptor_t*)handler)->options;
options_print_variables(options, print_extended);
return DLB_SUCCESS;
}
#pragma GCC visibility pop
