

#if DO_ITT_NOTIFY

#if _WIN32||_WIN64
#ifndef UNICODE
#define UNICODE
#endif
#else
#pragma weak dlopen
#pragma weak dlsym
#pragma weak dlerror
#endif 

#if __TBB_BUILD

extern "C" void ITT_DoOneTimeInitialization();
#define __itt_init_ittlib_name(x,y) (ITT_DoOneTimeInitialization(), true)

#elif __TBBMALLOC_BUILD

extern "C" void MallocInitializeITT();
#define __itt_init_ittlib_name(x,y) (MallocInitializeITT(), true)

#else
#error This file is expected to be used for either TBB or TBB allocator build.
#endif 

#include "tools_api/ittnotify_static.c"

namespace tbb {
namespace internal {
int __TBB_load_ittnotify() {
#if !(_WIN32||_WIN64)
if (dlopen == NULL)
return 0;
#endif
return __itt_init_ittlib(NULL,          
(__itt_group_id)(__itt_group_sync     
| __itt_group_thread 
| __itt_group_stitch 
#if __TBB_CPF_BUILD
| __itt_group_structure
#endif
));
}

}} 

#endif 

#define __TBB_NO_IMPLICIT_LINKAGE 1
#include "itt_notify.h"

namespace tbb {

#if DO_ITT_NOTIFY
const tchar
*SyncType_GlobalLock = _T("TbbGlobalLock"),
*SyncType_Scheduler = _T("%Constant")
;
const tchar
*SyncObj_SchedulerInitialization = _T("TbbSchedulerInitialization"),
*SyncObj_SchedulersList = _T("TbbSchedulersList"),
*SyncObj_WorkerLifeCycleMgmt = _T("TBB Scheduler"),
*SyncObj_TaskStealingLoop = _T("TBB Scheduler"),
*SyncObj_WorkerTaskPool = _T("TBB Scheduler"),
*SyncObj_MasterTaskPool = _T("TBB Scheduler"),
*SyncObj_TaskPoolSpinning = _T("TBB Scheduler"),
*SyncObj_Mailbox = _T("TBB Scheduler"),
*SyncObj_TaskReturnList = _T("TBB Scheduler"),
*SyncObj_TaskStream = _T("TBB Scheduler"),
*SyncObj_ContextsList = _T("TBB Scheduler")
;
#endif 

} 

