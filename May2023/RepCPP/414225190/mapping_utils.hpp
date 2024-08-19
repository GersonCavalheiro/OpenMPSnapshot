







#ifndef FF_MAPPING_UTILS_HPP
#define FF_MAPPING_UTILS_HPP

#include <climits>
#include <set>
#include <algorithm>
#include <iosfwd>
#include <errno.h>
#include <ff/config.hpp>
#include <ff/utils.hpp>
#if defined(__linux__)
#include <sched.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <asm/unistd.h>
#include <stdio.h>
#include <unistd.h>

static inline int ff_gettid() { return syscall(__NR_gettid);}

#if defined(MAMMUT)
#include <mammut/mammut.hpp>
#endif

#elif defined(__APPLE__)

#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/syscall.h>
#include <mach/mach.h> 
#include <mach/mach_init.h>
#include <mach/thread_policy.h> 
#elif defined(_WIN32)
#include <ff/platforms/platform.h>
#endif
#if defined(__APPLE__) && MAC_OS_X_HAS_AFFINITY
#include<vector> 
#endif


static inline size_t ff_getThreadID() {
#if (defined(__GNUC__) && defined(__linux))
return  ff_gettid();
#elif defined(__APPLE__) && MAC_OS_X_HAS_AFFINITY
#elif defined(_WIN32)
return GetCurrentThreadId();
#endif
return -1;
}


static inline unsigned long ff_getCpuFreq() {
unsigned long  t = 0;
#if defined(__linux__)
FILE       *f;
float       mhz;

f = popen("cat /proc/cpuinfo |grep MHz |head -1|sed 's/^.*: 
if (fscanf(f, "%f", &mhz) == EOF) {pclose(f); return t;}
t = (unsigned long)(mhz * 1000000);
pclose(f);
#elif defined(__APPLE__) && MAC_OS_X_HAS_AFFINITY
size_t len = 8;
if (sysctlbyname("hw.cpufrequency", &t, &len, NULL, 0) != 0) {
perror("sysctl");
}
#elif defined(_WIN32)
#else
#endif
return (t);
}


static inline ssize_t ff_numCores() {
if (FF_NUM_CORES != -1) return FF_NUM_CORES;
ssize_t  n=-1;
#if defined(__linux__)    
#if defined(MAMMUT)
mammut::Mammut m;
std::vector<mammut::topology::Cpu*> cpus = m.getInstanceTopology()->getCpus();
if (cpus.size()>0 && cpus[0]->getPhysicalCores().size()>0) {
n = 0;
for(size_t i = 0; i < cpus.size(); i++){
std::vector<mammut::topology::PhysicalCore*> phyCores  = cpus.at(i)->getPhysicalCores();
std::vector<mammut::topology::VirtualCore*>  virtCores = phyCores.at(0)->getVirtualCores();
n+= phyCores.size()*virtCores.size();
}
}
#else
#if defined(HAVE_PTHREAD_SETAFFINITY_NP)
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
do {
if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) 
break;        
#if defined(CPU_COUNT_S)
n = CPU_COUNT_S(sizeof(cpu_set_t), &cpuset);
#elif defined(CPU_COUNT)
n = CPU_COUNT(&cpuset);
#else
n=0;
for(size_t i=0;i<sizeof(cpu_set_t);++i) {
if (CPU_ISSET(i, &cpuset)) ++n;
}
#endif
return n;
} while(0);
#endif    
FILE       *f;    
f = popen("cat /proc/cpuinfo |grep processor | wc -l", "r");
if (fscanf(f, "%ld", &n) == EOF) { pclose(f); return n;}
pclose(f);
#endif 

#elif defined(__APPLE__) 
int nn;
size_t len = sizeof(nn);
if (sysctlbyname("hw.logicalcpu", &nn, &len, NULL, 0) == -1)
perror("sysctl");
n = nn;
#elif defined(_WIN32)
SYSTEM_INFO sysinfo;
GetSystemInfo( &sysinfo );
n = sysinfo.dwNumberOfProcessors;
#else
#endif
return n;
}



static inline ssize_t ff_realNumCores() {
if (FF_NUM_REAL_CORES != -1) return FF_NUM_REAL_CORES;
ssize_t  n=-1;
#if defined(_WIN32)
n = 2; 
#else
#if defined(__linux__)
#if defined(MAMMUT)

mammut::Mammut m;
std::vector<mammut::topology::Cpu*> cpus = m.getInstanceTopology()->getCpus();
if (cpus.size()>0 && cpus[0]->getPhysicalCores().size()>0) {
n = 0;
for(size_t i = 0; i < cpus.size(); i++){
std::vector<mammut::topology::PhysicalCore*> phyCores  = cpus.at(i)->getPhysicalCores();
n+= phyCores.size();
}
}
return n;    
#else 

#if defined(HAVE_PTHREAD_SETAFFINITY_NP)
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
int cnt=-1;
do {
if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) 
break;        
#if defined(CPU_COUNT_S)
cnt = CPU_COUNT_S(sizeof(cpu_set_t), &cpuset);
#elif defined(CPU_COUNT)
cnt = CPU_COUNT(&cpuset);
#endif
std::set<int> S;    
for(size_t i=0;i<CHAR_BIT*sizeof(cpu_set_t);++i) {
if (CPU_ISSET(i, &cpuset)) {
std::string str="cat /sys/devices/system/cpu/cpu"+std::to_string(i)+"/topology/thread_siblings_list";
char buf[64];
FILE       *f;                
if ((f = popen(str.c_str(), "r"))!= NULL) {
if (fscanf(f, "%s", buf) == EOF) { 
perror("fscanf");
return -1;
}
pclose(f);
} else { perror("popen"); return -1; }
const std::string bufstr(buf);
size_t pos = bufstr.find_first_of('-');
int first = std::stoi(bufstr.substr(0,pos));
int second = std::stoi(bufstr.substr(pos+1));
bool found = false;
for(int j=first;j<=second; ++j)
if (S.find(j) != S.end()) {
found = true;
break;
}
if (!found) S.insert(i);                
if (--cnt==0) {
n=0;
std::for_each(S.begin(), S.end(), [&n](int const&) { ++n;});
return n;
}
}
}
n=0;
std::for_each(S.begin(), S.end(), [&n](int const&) { ++n;});
return n;
} while(0);
#endif 
char inspect[]="cat /proc/cpuinfo|egrep 'core id|physical id'|tr -d '\n'|sed 's/physical/\\nphysical/g'|grep -v ^$|sort|uniq|wc -l";
#endif 
#elif defined(__APPLE__)
char inspect[] = "sysctl hw.physicalcpu | awk '{print $2}'";
#else
char inspect[]="";
n=1;
#pragma message ("ff_realNumCores not supported on this platform")
#endif
if (strlen(inspect)) {
FILE *f;
f = popen(inspect, "r");
if (f) {
if (fscanf(f, "%ld", &n) == EOF) {
perror("fscanf");
}
pclose(f);
} else
perror("popen");
}
#endif 
return n;
}


static inline ssize_t ff_numSockets() {
ssize_t  n=-1;
#if defined(_WIN32)
n = 1;
#else
#if defined(__linux__)
char inspect[]="cat /proc/cpuinfo|grep 'physical id'|sort|uniq|wc -l";
#if defined(MAMMUT)
mammut::Mammut m;
std::vector<mammut::topology::Cpu*> cpus = m.getInstanceTopology()->getCpus();
if (cpus.size()>0) n = cpus.size();
return n;    
#endif 
#elif defined (__APPLE__)
char inspect[]="sysctl hw.packages | awk '{print $2}'";
#else 
char inspect[]="";
n=1;
#pragma message ("ff_realNumCores not supported on this platform")
#endif
FILE       *f; 
f = popen(inspect, "r");
if (f) {
if (fscanf(f, "%ld", &n) == EOF) { 
perror("fscanf");
}
pclose(f);
} else perror("popen");
#endif 
return n;
}



static inline ssize_t ff_setPriority(ssize_t priority_level=0) {
ssize_t ret=0;
#if (defined(__GNUC__) && defined(__linux))
if (setpriority(PRIO_PROCESS, ff_gettid(), priority_level) != 0) {
perror("setpriority:");
ret = EINVAL;
}
#elif defined(__APPLE__) 
if (setpriority(PRIO_PROCESS, 0  ,priority_level) != 0) {
perror("setpriority:");
ret = EINVAL;
}
#elif defined(_WIN32)
ssize_t pri = (priority_level + 20)/10;
ssize_t subpri = ((priority_level + 20)%10)/2;
switch (pri) {
case 0: ret = !(SetPriorityClass(GetCurrentThread(),HIGH_PRIORITY_CLASS));
break;
case 1: ret = !(SetPriorityClass(GetCurrentThread(),ABOVE_NORMAL_PRIORITY_CLASS));
break;
case 2: ret = !(SetPriorityClass(GetCurrentThread(),NORMAL_PRIORITY_CLASS));
break;
case 3: ret = !(SetPriorityClass(GetCurrentThread(),BELOW_NORMAL_PRIORITY_CLASS));
break;
default: ret = EINVAL;
}

switch (pri) {
case 0: ret |= !(SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST));
break;
case 1: ret |= !(SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_ABOVE_NORMAL));
break;
case 2: ret |= !(SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_NORMAL));
break;
case 3: ret |= !(SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_BELOW_NORMAL));
break;
case 4: ret |= !(SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_LOWEST));
break;
default: ret = EINVAL;
}
#else
#endif
if (ret!=0) perror("ff_setPriority");
return ret;
}


static inline ssize_t ff_getMyCore() {
#if defined(__linux__) && defined(CPU_SET)
cpu_set_t mask;
CPU_ZERO(&mask);
if (sched_getaffinity(ff_gettid(), sizeof(mask), &mask) != 0) {
perror("sched_getaffinity");
return EINVAL;
}
for(int i=0;i<CPU_SETSIZE;++i) 
if (CPU_ISSET(i,&mask)) return i;
#elif defined(__APPLE__) && MAC_OS_X_HAS_AFFINITY
struct thread_affinity_policy mypolicy;
boolean_t get_default;
mach_msg_type_number_t thread_info_count = THREAD_AFFINITY_POLICY_COUNT;
thread_policy_get(mach_thread_self(), THREAD_AFFINITY_POLICY,
(integer_t*) &mypolicy,
&thread_info_count, &get_default);
ssize_t res = mypolicy.affinity_tag;
return(res);
#else
#if __GNUC__
#warning "ff_getMyCpu not supported"
#else 
#pragma message( "ff_getMyCpu not supported")
#endif
#endif
return -1;
}
static inline ssize_t ff_getMyCpu() { return ff_getMyCore(); }


static inline ssize_t ff_mapThreadToCpu(int cpu_id, int priority_level=0) {
#if defined(__linux__) && defined(CPU_SET)
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET(cpu_id, &mask);
if (sched_setaffinity(ff_gettid(), sizeof(mask), &mask) != 0) 
return EINVAL;
return (ff_setPriority(priority_level));
#elif defined(__APPLE__) && MAC_OS_X_HAS_AFFINITY
#define CACHE_LEVELS 3
#define CACHE_L2     2
size_t len;

if (sysctlbyname("hw.cacheconfig",NULL, &len, NULL, 0) != 0) {
perror("sysctl");
} else {
std::vector<int64_t> cacheconfig(len);
if (sysctlbyname("hw.cacheconfig", &cacheconfig[0], &len, NULL, 0) != 0)
perror("sysctl: unable to get hw.cacheconfig");
else {

struct thread_affinity_policy mypolicy;
mypolicy.affinity_tag = cpu_id/cacheconfig[CACHE_L2];
if ( thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY, (integer_t*) &mypolicy, THREAD_AFFINITY_POLICY_COUNT) != KERN_SUCCESS ) {
perror("thread_policy_set: unable to set affinity of thread");
return EINVAL;
} 
}
}
return(ff_setPriority(priority_level));
#elif defined(_WIN32)
if (-1==SetThreadIdealProcessor(GetCurrentThread(),cpu_id)) {
perror("ff_mapThreadToCpu:SetThreadIdealProcessor");
return EINVAL;
}
#else 
#warning "CPU_SET not defined, cannot map thread to specific CPU"
#endif
return 0;
}


#include <stddef.h>

#if defined(__APPLE__)

static inline size_t cache_line_size() {

u_int64_t line_size = 0;
size_t sizeof_line_size = sizeof(line_size);
if (sysctlbyname("hw.cachelinesize", &line_size, &sizeof_line_size, NULL, 0) !=0) {
perror("cachelinesize:");
line_size=0;
}
return line_size;
}

#elif defined(_WIN32)


static inline size_t cache_line_size() {
size_t line_size = 0;
DWORD buffer_size = 0;
DWORD i = 0;
SYSTEM_LOGICAL_PROCESSOR_INFORMATION * buffer = 0;

GetLogicalProcessorInformation(0, &buffer_size);
buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(buffer_size);
GetLogicalProcessorInformation(&buffer[0], &buffer_size);

for (i = 0; i != buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i) {
if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == 1) {
line_size = buffer[i].Cache.LineSize;
break;
}
}

free(buffer);
return line_size;
}

#elif defined(__linux__)



static inline size_t cache_line_size() {
FILE * p = 0;
p = fopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
unsigned int i = 0;
if (p) {
if (fscanf(p, "%ud", &i) == EOF) { 
perror("fscanf");
if (fclose(p) != 0) perror("fclose"); 
return 0;
}
if (fclose(p) != 0) {
perror("fclose");
return 0;
}
}
return i;
}

#else
#error Unrecognized platform
#endif

#endif 
