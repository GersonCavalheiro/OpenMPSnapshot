#ifndef HELPERS_CUH
#define HELPERS_CUH

#pragma once

#define errorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
if (code != hipSuccess)
{
fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
if (abort) exit(code);
}
}

#endif 
