
#pragma once

#if _MSC_VER
#include <Windows.h>
#elif __GNUC__
#include "time.h"
#endif

using namespace std;

#if _MSC_VER
struct Timer { 
double pc_frequency = 0.0;
double elapsed_time_milliseconds = 0.0;
LARGE_INTEGER start_time, end_time = { 0 };

inline Timer() {
LARGE_INTEGER li;
QueryPerformanceFrequency(&li);
pc_frequency = static_cast<double>(li.QuadPart) / 1000.0;
}

inline void reset() {
elapsed_time_milliseconds = 0.0;
}

inline void start() {
QueryPerformanceCounter(&start_time);
}

inline void stop() {
QueryPerformanceCounter(&end_time);
elapsed_time_milliseconds += static_cast<double>((end_time.QuadPart - start_time.QuadPart) / pc_frequency);
}
};
#else

#define MILLION 1000000.0f

struct Timer { 
double elapsed_time_milliseconds = 0;
timespec t1;
timespec t2;

inline Timer() {
}

inline void start() {
clock_gettime(CLOCK_REALTIME, &t1);
}

inline void stop() {
clock_gettime(CLOCK_REALTIME, &t2);
elapsed_time_milliseconds += (t2.tv_sec - t1.tv_sec) * 1000.0f;
elapsed_time_milliseconds += ((float)(t2.tv_nsec - t1.tv_nsec)) / MILLION;
}
};
#endif
