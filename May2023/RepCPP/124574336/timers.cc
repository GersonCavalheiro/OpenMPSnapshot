
#include "timers.h"
#include "internal_macros.h"

#ifdef BENCHMARK_OS_WINDOWS
#include <shlwapi.h>
#undef StrCat  
#include <versionhelpers.h>
#include <windows.h>
#else
#include <fcntl.h>
#ifndef BENCHMARK_OS_FUCHSIA
#include <sys/resource.h>
#endif
#include <sys/time.h>
#include <sys/types.h>  
#include <unistd.h>
#if defined BENCHMARK_OS_FREEBSD || defined BENCHMARK_OS_MACOSX
#include <sys/sysctl.h>
#endif
#if defined(BENCHMARK_OS_MACOSX)
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/thread_act.h>
#endif
#endif

#ifdef BENCHMARK_OS_EMSCRIPTEN
#include <emscripten.h>
#endif

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <mutex>

#include "check.h"
#include "log.h"
#include "sleep.h"
#include "string_util.h"

namespace benchmark {

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace {
#if defined(BENCHMARK_OS_WINDOWS)
double MakeTime(FILETIME const& kernel_time, FILETIME const& user_time) {
ULARGE_INTEGER kernel;
ULARGE_INTEGER user;
kernel.HighPart = kernel_time.dwHighDateTime;
kernel.LowPart = kernel_time.dwLowDateTime;
user.HighPart = user_time.dwHighDateTime;
user.LowPart = user_time.dwLowDateTime;
return (static_cast<double>(kernel.QuadPart) +
static_cast<double>(user.QuadPart)) *
1e-7;
}
#elif !defined(BENCHMARK_OS_FUCHSIA)
double MakeTime(struct rusage const& ru) {
return (static_cast<double>(ru.ru_utime.tv_sec) +
static_cast<double>(ru.ru_utime.tv_usec) * 1e-6 +
static_cast<double>(ru.ru_stime.tv_sec) +
static_cast<double>(ru.ru_stime.tv_usec) * 1e-6);
}
#endif
#if defined(BENCHMARK_OS_MACOSX)
double MakeTime(thread_basic_info_data_t const& info) {
return (static_cast<double>(info.user_time.seconds) +
static_cast<double>(info.user_time.microseconds) * 1e-6 +
static_cast<double>(info.system_time.seconds) +
static_cast<double>(info.system_time.microseconds) * 1e-6);
}
#endif
#if defined(CLOCK_PROCESS_CPUTIME_ID) || defined(CLOCK_THREAD_CPUTIME_ID)
double MakeTime(struct timespec const& ts) {
return ts.tv_sec + (static_cast<double>(ts.tv_nsec) * 1e-9);
}
#endif

BENCHMARK_NORETURN static void DiagnoseAndExit(const char* msg) {
std::cerr << "ERROR: " << msg << std::endl;
std::exit(EXIT_FAILURE);
}

}  

double ProcessCPUUsage() {
#if defined(BENCHMARK_OS_WINDOWS)
HANDLE proc = GetCurrentProcess();
FILETIME creation_time;
FILETIME exit_time;
FILETIME kernel_time;
FILETIME user_time;
if (GetProcessTimes(proc, &creation_time, &exit_time, &kernel_time,
&user_time))
return MakeTime(kernel_time, user_time);
DiagnoseAndExit("GetProccessTimes() failed");
#elif defined(BENCHMARK_OS_EMSCRIPTEN)
return emscripten_get_now() * 1e-3;
#elif defined(CLOCK_PROCESS_CPUTIME_ID) && !defined(BENCHMARK_OS_MACOSX)
struct timespec spec;
if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &spec) == 0)
return MakeTime(spec);
DiagnoseAndExit("clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ...) failed");
#else
struct rusage ru;
if (getrusage(RUSAGE_SELF, &ru) == 0) return MakeTime(ru);
DiagnoseAndExit("getrusage(RUSAGE_SELF, ...) failed");
#endif
}

double ThreadCPUUsage() {
#if defined(BENCHMARK_OS_WINDOWS)
HANDLE this_thread = GetCurrentThread();
FILETIME creation_time;
FILETIME exit_time;
FILETIME kernel_time;
FILETIME user_time;
GetThreadTimes(this_thread, &creation_time, &exit_time, &kernel_time,
&user_time);
return MakeTime(kernel_time, user_time);
#elif defined(BENCHMARK_OS_MACOSX)
mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
thread_basic_info_data_t info;
mach_port_t thread = pthread_mach_thread_np(pthread_self());
if (thread_info(thread, THREAD_BASIC_INFO, (thread_info_t)&info, &count) ==
KERN_SUCCESS) {
return MakeTime(info);
}
DiagnoseAndExit("ThreadCPUUsage() failed when evaluating thread_info");
#elif defined(BENCHMARK_OS_EMSCRIPTEN)
return ProcessCPUUsage();
#elif defined(BENCHMARK_OS_RTEMS)
return ProcessCPUUsage();
#elif defined(BENCHMARK_OS_SOLARIS)
struct rusage ru;
if (getrusage(RUSAGE_LWP, &ru) == 0) return MakeTime(ru);
DiagnoseAndExit("getrusage(RUSAGE_LWP, ...) failed");
#elif defined(CLOCK_THREAD_CPUTIME_ID)
struct timespec ts;
if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) return MakeTime(ts);
DiagnoseAndExit("clock_gettime(CLOCK_THREAD_CPUTIME_ID, ...) failed");
#else
#error Per-thread timing is not available on your system.
#endif
}

namespace {

std::string DateTimeString(bool local) {
typedef std::chrono::system_clock Clock;
std::time_t now = Clock::to_time_t(Clock::now());
const std::size_t kStorageSize = 128;
char storage[kStorageSize];
std::size_t written;

if (local) {
#if defined(BENCHMARK_OS_WINDOWS)
written =
std::strftime(storage, sizeof(storage), "%x %X", ::localtime(&now));
#else
std::tm timeinfo;
::localtime_r(&now, &timeinfo);
written = std::strftime(storage, sizeof(storage), "%F %T", &timeinfo);
#endif
} else {
#if defined(BENCHMARK_OS_WINDOWS)
written = std::strftime(storage, sizeof(storage), "%x %X", ::gmtime(&now));
#else
std::tm timeinfo;
::gmtime_r(&now, &timeinfo);
written = std::strftime(storage, sizeof(storage), "%F %T", &timeinfo);
#endif
}
CHECK(written < kStorageSize);
((void)written);  
return std::string(storage);
}

}  

std::string LocalDateTimeString() { return DateTimeString(true); }

}  
