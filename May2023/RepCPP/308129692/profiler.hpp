#pragma once
#include <iostream>

#ifdef COSTA_WITH_PROFILING
#include <semiprof.hpp>

#define PP() std::cout << semiprof::profiler_summary() << "\n"

#define PC() semiprof::profiler_clear()

#else
#define PE(name)
#define PL()
#define PP()
#define PC()
#endif
