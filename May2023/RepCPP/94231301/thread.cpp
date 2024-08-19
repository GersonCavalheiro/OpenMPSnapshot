
#include "thread.h"
#include "sysinfo.h"
#include "string.h"

#include <iostream>
#include <xmmintrin.h>

#if defined(PTHREADS_WIN32)
#pragma comment (lib, "pthreadVC.lib")
#endif


#if defined(__WIN32__)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace embree
{

void setAffinity(HANDLE thread, ssize_t affinity)
{
typedef WORD (WINAPI *GetActiveProcessorGroupCountFunc)();
typedef DWORD (WINAPI *GetActiveProcessorCountFunc)(WORD);
typedef BOOL (WINAPI *SetThreadGroupAffinityFunc)(HANDLE, const GROUP_AFFINITY *, PGROUP_AFFINITY);
typedef BOOL (WINAPI *SetThreadIdealProcessorExFunc)(HANDLE, PPROCESSOR_NUMBER, PPROCESSOR_NUMBER);
HMODULE hlib = LoadLibrary("Kernel32");
GetActiveProcessorGroupCountFunc pGetActiveProcessorGroupCount = (GetActiveProcessorGroupCountFunc)GetProcAddress(hlib, "GetActiveProcessorGroupCount");
GetActiveProcessorCountFunc pGetActiveProcessorCount = (GetActiveProcessorCountFunc)GetProcAddress(hlib, "GetActiveProcessorCount");
SetThreadGroupAffinityFunc pSetThreadGroupAffinity = (SetThreadGroupAffinityFunc)GetProcAddress(hlib, "SetThreadGroupAffinity");
SetThreadIdealProcessorExFunc pSetThreadIdealProcessorEx = (SetThreadIdealProcessorExFunc)GetProcAddress(hlib, "SetThreadIdealProcessorEx");
if (pGetActiveProcessorGroupCount && pGetActiveProcessorCount && pSetThreadGroupAffinity && pSetThreadIdealProcessorEx) 
{
int groups = pGetActiveProcessorGroupCount();
int totalProcessors = 0, group = 0, number = 0;
for (int i = 0; i<groups; i++) {
int processors = pGetActiveProcessorCount(i);
if (totalProcessors + processors > affinity) {
group = i;
number = (int)affinity - totalProcessors;
break;
}
totalProcessors += processors;
}

GROUP_AFFINITY groupAffinity;
groupAffinity.Group = (WORD)group;
groupAffinity.Mask = (KAFFINITY)(uint64_t(1) << number);
groupAffinity.Reserved[0] = 0;
groupAffinity.Reserved[1] = 0;
groupAffinity.Reserved[2] = 0;
if (!pSetThreadGroupAffinity(thread, &groupAffinity, nullptr))
WARNING("SetThreadGroupAffinity failed"); 

PROCESSOR_NUMBER processorNumber;
processorNumber.Group = group;
processorNumber.Number = number;
processorNumber.Reserved = 0;
if (!pSetThreadIdealProcessorEx(thread, &processorNumber, nullptr))
WARNING("SetThreadIdealProcessorEx failed"); 
} 
else 
{
if (!SetThreadAffinityMask(thread, DWORD_PTR(uint64_t(1) << affinity)))
WARNING("SetThreadAffinityMask failed"); 
if (SetThreadIdealProcessor(thread, (DWORD)affinity) == (DWORD)-1)
WARNING("SetThreadIdealProcessor failed"); 
}
}


void setAffinity(ssize_t affinity) {
setAffinity(GetCurrentThread(), affinity);
}

struct ThreadStartupData 
{
public:
ThreadStartupData (thread_func f, void* arg) 
: f(f), arg(arg) {}
public:
thread_func f;
void* arg;
};

DWORD WINAPI threadStartup(LPVOID ptr)
{
ThreadStartupData* parg = (ThreadStartupData*) ptr;
_mm_setcsr(_mm_getcsr() |  (1<<15) |  (1<<6));
parg->f(parg->arg);
delete parg;
return 0;
}

#if !defined(PTHREADS_WIN32)


thread_t createThread(thread_func f, void* arg, size_t stack_size, ssize_t threadID)
{
HANDLE thread = CreateThread(nullptr, stack_size, threadStartup, new ThreadStartupData(f,arg), 0, nullptr);
if (thread == nullptr) FATAL("CreateThread failed");
if (threadID >= 0) setAffinity(thread, threadID);
return thread_t(thread);
}


void yield() {
SwitchToThread();
}


void join(thread_t tid) {
WaitForSingleObject(HANDLE(tid), INFINITE);
CloseHandle(HANDLE(tid));
}


void destroyThread(thread_t tid) {
TerminateThread(HANDLE(tid),0);
CloseHandle(HANDLE(tid));
}


tls_t createTls() {
return tls_t(size_t(TlsAlloc()));
}


void setTls(tls_t tls, void* const ptr) {
TlsSetValue(DWORD(size_t(tls)), ptr);
}


void* getTls(tls_t tls) {
return TlsGetValue(DWORD(size_t(tls)));
}


void destroyTls(tls_t tls) {
TlsFree(DWORD(size_t(tls)));
}
#endif
}

#endif


#if defined(__LINUX__)

#include <fstream>
#include <sstream>
#include <algorithm>

namespace embree
{

size_t mapThreadID(size_t threadID)
{
static MutexSys mutex;
Lock<MutexSys> lock(mutex);
static std::vector<size_t> threadIDs;

if (threadIDs.size() == 0)
{

for (size_t cpuID=0;;cpuID++)
{
std::fstream fs;
std::string cpu = std::string("/sys/devices/system/cpu/cpu") + std::to_string((long long)cpuID) + std::string("/topology/thread_siblings_list");
fs.open (cpu.c_str(), std::fstream::in);
if (fs.fail()) break;

int i;
while (fs >> i) 
{
if (std::none_of(threadIDs.begin(),threadIDs.end(),[&] (int id) { return id == i; }))
threadIDs.push_back(i);
if (fs.peek() == ',') 
fs.ignore();
}
fs.close();
}

#if 0
for (size_t i=0;i<threadIDs.size();i++)
std::cout << i << " -> " << threadIDs[i] << std::endl;
#endif


for (size_t i=0;i<threadIDs.size();i++) {
for (size_t j=0;j<threadIDs.size();j++) {
if (i != j && threadIDs[i] == threadIDs[j]) {
threadIDs.clear();
}
}
}
}


size_t ID = threadID;
if (threadID < threadIDs.size())
ID = threadIDs[threadID];

return ID;
}


void setAffinity(ssize_t affinity)
{
cpu_set_t cset;
CPU_ZERO(&cset);
CPU_SET(mapThreadID(affinity), &cset);

if (pthread_setaffinity_np(pthread_self(), sizeof(cset), &cset) != 0)
WARNING("pthread_setaffinity_np failed"); 
}
}
#endif


#if defined(__FreeBSD__)

#include <pthread_np.h>

namespace embree
{

void setAffinity(ssize_t affinity)
{
cpuset_t cset;
CPU_ZERO(&cset);
CPU_SET(affinity, &cset);

if (pthread_setaffinity_np(pthread_self(), sizeof(cset), &cset) != 0)
WARNING("pthread_setaffinity_np failed"); 
}
}
#endif


#if defined(__MACOSX__)

#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <mach/mach_init.h>

namespace embree
{

void setAffinity(ssize_t affinity)
{
thread_affinity_policy ap;
ap.affinity_tag = affinity;
if (thread_policy_set(mach_thread_self(),THREAD_AFFINITY_POLICY,(thread_policy_t)&ap,THREAD_AFFINITY_POLICY_COUNT) != KERN_SUCCESS)
WARNING("setting thread affinity failed"); 
}
}
#endif


#if defined(__UNIX__) || defined(PTHREADS_WIN32)

#include <pthread.h>
#include <sched.h>

#if defined(__USE_NUMA__)
#include <numa.h>
#endif

namespace embree
{
struct ThreadStartupData 
{
public:
ThreadStartupData (thread_func f, void* arg, int affinity) 
: f(f), arg(arg), affinity(affinity) {}
public: 
thread_func f;
void* arg;
ssize_t affinity;
};

static void* threadStartup(ThreadStartupData* parg)
{
_mm_setcsr(_mm_getcsr() |  (1<<15) |  (1<<6));


#if defined(__MACOSX__)
if (parg->affinity >= 0)
setAffinity(parg->affinity);
#endif

parg->f(parg->arg);
delete parg;
return nullptr;
}


thread_t createThread(thread_func f, void* arg, size_t stack_size, ssize_t threadID)
{

pthread_attr_t attr;
pthread_attr_init(&attr);
if (stack_size > 0) pthread_attr_setstacksize (&attr, stack_size);


pthread_t* tid = new pthread_t;
if (pthread_create(tid,&attr,(void*(*)(void*))threadStartup,new ThreadStartupData(f,arg,threadID)) != 0) {
pthread_attr_destroy(&attr);
delete tid; 
FATAL("pthread_create failed");
}
pthread_attr_destroy(&attr);


#if defined(__LINUX__)
if (threadID >= 0) {
cpu_set_t cset;
CPU_ZERO(&cset);
CPU_SET(mapThreadID(threadID), &cset);
if (pthread_setaffinity_np(*tid, sizeof(cset), &cset))
WARNING("pthread_setaffinity_np failed"); 
}
#elif defined(__FreeBSD__)
if (threadID >= 0) {
cpuset_t cset;
CPU_ZERO(&cset);
CPU_SET(threadID, &cset);
if (pthread_setaffinity_np(*tid, sizeof(cset), &cset))
WARNING("pthread_setaffinity_np failed"); 
}
#endif

return thread_t(tid);
}


void yield() {
sched_yield();
}


void join(thread_t tid) {
if (pthread_join(*(pthread_t*)tid, nullptr) != 0)
FATAL("pthread_join failed");
delete (pthread_t*)tid;
}


void destroyThread(thread_t tid) {
pthread_cancel(*(pthread_t*)tid);
delete (pthread_t*)tid;
}


tls_t createTls() 
{
pthread_key_t* key = new pthread_key_t;
if (pthread_key_create(key,nullptr) != 0) {
delete key;
FATAL("pthread_key_create failed");
}

return tls_t(key);
}


void* getTls(tls_t tls) 
{
assert(tls);
return pthread_getspecific(*(pthread_key_t*)tls);
}


void setTls(tls_t tls, void* const ptr) 
{
assert(tls);
if (pthread_setspecific(*(pthread_key_t*)tls, ptr) != 0)
FATAL("pthread_setspecific failed");
}


void destroyTls(tls_t tls) 
{
assert(tls);
if (pthread_key_delete(*(pthread_key_t*)tls) != 0)
FATAL("pthread_key_delete failed");
delete (pthread_key_t*)tls;
}
}

#endif
