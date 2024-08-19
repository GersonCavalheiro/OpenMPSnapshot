

#ifndef EDGE_PARALLEL_GLOBAL_H_
#define EDGE_PARALLEL_GLOBAL_H_

#include "constants.hpp"
#include <string>

namespace edge {
namespace parallel {
extern int           g_thread;
extern char          g_threadStr[10];
#ifdef PP_SCRATCH_MEMORY
extern t_scratchMem* g_scratchMem;
#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_scratchMem)
#endif

#endif

#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_thread, edge::parallel::g_threadStr)
#endif
extern int         g_nThreads;

extern int         g_rank;
extern std::string g_rankStr;
extern int         g_nRanks;
}
}

#endif
