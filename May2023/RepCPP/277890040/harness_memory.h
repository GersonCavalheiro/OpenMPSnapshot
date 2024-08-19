


#if __linux__ || __sun
#include <sys/resource.h>
#include <unistd.h>

#elif __APPLE__ && !__ARM_ARCH
#include <unistd.h>
#include <mach/mach.h>
#include <AvailabilityMacros.h>
#if MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_6 || __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_8_0
#include <mach/shared_region.h>
#else
#include <mach/shared_memory_server.h>
#endif
#if SHARED_TEXT_REGION_SIZE || SHARED_DATA_REGION_SIZE
const size_t shared_size = SHARED_TEXT_REGION_SIZE+SHARED_DATA_REGION_SIZE;
#else
const size_t shared_size = 0;
#endif

#elif _WIN32 && !__TBB_WIN8UI_SUPPORT
#include <windows.h>
#include <psapi.h>
#if _MSC_VER
#pragma comment(lib, "psapi")
#endif

#endif 

enum MemoryStatType {
currentUsage,
peakUsage
};


size_t GetMemoryUsage(MemoryStatType stat = currentUsage) {
ASSERT(stat==currentUsage || stat==peakUsage, NULL);
#if __TBB_WIN8UI_SUPPORT
return 0;
#elif _WIN32
PROCESS_MEMORY_COUNTERS mem;
bool status = GetProcessMemoryInfo(GetCurrentProcess(), &mem, sizeof(mem))!=0;
ASSERT(status, NULL);
return stat==currentUsage? mem.PagefileUsage : mem.PeakPagefileUsage;
#elif __linux__
long unsigned size = 0;
FILE *fst = fopen("/proc/self/status", "r");
ASSERT(fst, NULL);
const int BUF_SZ = 200;
char buf_stat[BUF_SZ];
const char *pattern = stat==peakUsage ? "VmPeak: %lu" : "VmSize: %lu";
while (NULL != fgets(buf_stat, BUF_SZ, fst)) {
if (1==sscanf(buf_stat, pattern, &size)) {
ASSERT(size, "Invalid value of memory consumption.");
break;
}
}
if (stat!=peakUsage || LinuxKernelVersion() >= 2006015)
ASSERT(size, "Invalid /proc/self/status format, pattern not found.");
fclose(fst);
return size*1024;
#elif __APPLE__ && !__ARM_ARCH
if (stat == peakUsage)
return 0;
kern_return_t status;
task_basic_info info;
mach_msg_type_number_t msg_type = TASK_BASIC_INFO_COUNT;
status = task_info(mach_task_self(), TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &msg_type);
ASSERT(status==KERN_SUCCESS, NULL);
return info.virtual_size - shared_size;
#else
return 0;
#endif
}


void UseStackSpace( size_t amount, char* top=0 ) {
char x[1000];
memset( x, -1, sizeof(x) );
if( !top )
top = x;
ASSERT( x<=top, "test assumes that stacks grow downwards" );
if( size_t(top-x)<amount )
UseStackSpace( amount, top );
}

#if __linux__
#include "../tbbmalloc/shared_utils.h"

inline bool isTHPEnabledOnMachine() {
unsigned long long thpPresent = 'n';
parseFileItem thpItem[] = { { "[alwa%cs] madvise never\n", thpPresent } };
parseFile<100>("/sys/kernel/mm/transparent_hugepage/enabled", thpItem);

if (thpPresent == 'y') {
return true;
} else {
return false;
}
}
inline unsigned long long getSystemTHPAllocatedSize() {
unsigned long long anonHugePagesSize = 0;
parseFileItem meminfoItems[] = {
{ "AnonHugePages: %llu kB", anonHugePagesSize } };
parseFile<100>("/proc/meminfo", meminfoItems);
return anonHugePagesSize;
}
inline unsigned long long getSystemTHPCount() {
unsigned long long anonHugePages = 0;
parseFileItem vmstatItems[] = {
{ "nr_anon_transparent_hugepages %llu", anonHugePages } };
parseFile<100>("/proc/vmstat", vmstatItems);
return anonHugePages;
}
#endif 

