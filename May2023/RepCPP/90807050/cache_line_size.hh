

#ifndef cache_line_size_hh_
#define cache_line_size_hh_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"


#include <stddef.h>

namespace mad { namespace cache { static size_t cache_line_size(); } }

#if defined(__APPLE__)

#include <cstdbool>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/sysctl.h>

static size_t mad::cache::cache_line_size()
{
size_t line_size = 0;
size_t sizeof_line_size = sizeof(line_size);
sysctlbyname("hw.cachelinesize", &line_size, &sizeof_line_size, 0, 0);
return line_size;
}

#elif defined(_WIN32)

#include <stdlib.h>
#include <windows.h>
static size_t mad::cache::cache_line_size()
{
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

#include <stdio.h>
static inline size_t mad::cache::cache_line_size()
{
FILE * p = 0;
p = fopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
int i = 32;
int ret = -1;
if (p)
{
ret = fscanf(p, "%d", &i);
fclose(p);
if(ret != 1)
ret = printf("Unable to read cache line size\n");
}
return i;
}

#else
#error Unrecognized platform
#endif


#pragma GCC diagnostic pop

#endif
