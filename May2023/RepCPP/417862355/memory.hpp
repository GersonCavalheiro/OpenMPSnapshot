

#pragma once

namespace quids::utils {
#ifdef __CYGWIN__ 

#include <windows.h>

size_t inline get_free_mem() {
MEMORYSTATUSEX statex;
statex.dwLength = sizeof (statex);
GlobalMemoryStatusEx (&statex);

return statex.AvailPageFile; 
}

#elif defined(__linux__) 

size_t inline get_free_mem() {
char buff[128];
char useless[128];
unsigned long free_mem = 0;

FILE *fd = fopen("/proc/meminfo", "r");

fgets(buff, sizeof(buff), fd); 
fgets(buff, sizeof(buff), fd); 
sscanf(buff, "%s %lu ", useless, &free_mem); 

return free_mem * 1000; 
}

#elif defined(__unix__) 
#error "UNIX system other than LINUX aren't supported for now"
#elif defined(__MACH__) 
#error "macos isn't supported for now !"
#else 
#error "system isn't supported"

size_t inline get_free_mem() { return 0; }
#endif
}