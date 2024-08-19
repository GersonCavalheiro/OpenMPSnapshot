
#ifndef SEQAN_PARALLEL_PARALLEL_MACROS_H_
#define SEQAN_PARALLEL_PARALLEL_MACROS_H_



#ifdef _OPENMP

#include <omp.h>

#if defined(COMPILER_MSVC) || defined(COMPILER_WINTEL)
#define SEQAN_OMP_PRAGMA(x) __pragma(omp x)
#else
#define SEQAN_DO_PRAGMA(x) _Pragma(# x)
#define SEQAN_OMP_PRAGMA(x) SEQAN_DO_PRAGMA(omp x)
#endif

#else  

#define SEQAN_OMP_PRAGMA(x)

inline void omp_set_num_threads(int)
{}

inline int omp_get_num_threads()
{
return 1;
}

inline int omp_get_max_threads()
{
return 1;
}

inline int omp_get_thread_num()
{
return 0;
}

inline double omp_get_wtime()
{
return seqan::sysTime();
}

#endif  


inline unsigned getThreadId()
{
#if defined(_OPENMP)
return omp_get_thread_num();

#else
return 0;

#endif
}

#endif  
