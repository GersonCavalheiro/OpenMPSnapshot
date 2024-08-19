
#include "global.h"

#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_thread, edge::parallel::g_threadStr)
#endif
int           edge::parallel::g_nThreads = 1;
int           edge::parallel::g_thread = 0;
char          edge::parallel::g_threadStr[10] = "0\000000000";
#ifdef PP_SCRATCH_MEMORY
#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_scratchMem)
#endif
t_scratchMem* edge::parallel::g_scratchMem = nullptr;
#endif

int           edge::parallel::g_nRanks = 1;
int           edge::parallel::g_rank = 0;
std::string   edge::parallel::g_rankStr = std::to_string(0);