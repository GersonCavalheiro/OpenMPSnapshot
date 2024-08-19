#pragma once
#include "stdafx.h"
#ifdef _WIN32
#include <Windows.h>
#define PFDG_TIMESTAMP LARGE_INTEGER
PFDG_TIMESTAMP freq;
SINLINE void pfdg_timestamp_init()
{
QueryPerformanceFrequency(&freq);
}
SINLINE void pfdg_timestamp_get(PFDG_TIMESTAMP* ts)
{
QueryPerformanceCounter(ts);
}
SINLINE PFDG_TIMESTAMP pfdg_timestamp_diff(const PFDG_TIMESTAMP start, const PFDG_TIMESTAMP end)
{
PFDG_TIMESTAMP res;
res.QuadPart = end.QuadPart - start.QuadPart;
return res;
}
SINLINE uint64_t pfdg_timestamp_microseconds(const PFDG_TIMESTAMP ts)
{
return (ts.QuadPart * 1000000Ui64) / freq.QuadPart;
}
#else
#define PFDG_TIMESTAMP struct timespec
PFDG_TIMESTAMP freq;
SINLINE void pfdg_timestamp_init()
{
int err = clock_getres(CLOCK_MONOTONIC_RAW, &freq);
if (err != 0)
printf("clock error");
}
SINLINE void pfdg_timestamp_get(PFDG_TIMESTAMP* ts)
{
int err = clock_gettime(CLOCK_MONOTONIC_RAW, ts);
if (err != 0)
printf("clock error");
}
SINLINE PFDG_TIMESTAMP pfdg_timestamp_diff(const PFDG_TIMESTAMP start, const PFDG_TIMESTAMP end)
{
PFDG_TIMESTAMP res;
res.tv_sec = end.tv_sec - start.tv_sec;
res.tv_nsec = end.tv_nsec - start.tv_nsec;
return res;
}
SINLINE uint64_t pfdg_timestamp_microseconds(const PFDG_TIMESTAMP ts)
{
return ((ts.tv_sec * 1000000000ULL) + ts.tv_nsec) / 1000ULL;
}
#endif
