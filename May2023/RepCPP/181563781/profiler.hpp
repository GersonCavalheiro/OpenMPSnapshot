#pragma once

#ifdef GRID2GRID_WITH_PROFILING
#include <semiprof/semiprof.hpp>
#else
#define PE(name)
#define PL()
#define PP()
#define PC()
#endif
