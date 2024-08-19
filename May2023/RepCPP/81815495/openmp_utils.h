
#ifndef KRATOS_OPENMP_UTILS_H
#define	KRATOS_OPENMP_UTILS_H

#include <stdio.h>
#include <vector>
#include <iostream>
#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#else
#include <ctime>
#endif
#include "parallel_utilities.h"

namespace Kratos
{



class OpenMPUtils
{
public:



typedef std::vector<int> PartitionVector;



KRATOS_DEPRECATED_MESSAGE("This is legacy version, please use the \"ParallelUtilities\" instead") static inline int GetNumThreads()
{
return ParallelUtilities::GetNumThreads();
}


static int GetCurrentNumberOfThreads()
{
#ifdef _OPENMP
return omp_get_num_threads();
#else
return 1;
#endif
}


KRATOS_DEPRECATED_MESSAGE("This is legacy version, please use the \"ParallelUtilities\" instead") static int GetNumberOfProcessors()
{
return ParallelUtilities::GetNumProcs();
}


static int IsDynamic()
{
#ifdef _OPENMP
return omp_get_dynamic();
#else
return 0;
#endif
}


static inline int ThisThread()
{
#ifdef _OPENMP
return omp_get_thread_num();
#else
return 0;
#endif
}


static inline int IsInParallel()
{
#ifdef _OPENMP
return omp_in_parallel();
#else
return 0;
#endif
}


KRATOS_DEPRECATED_MESSAGE("This is legacy version, please use the \"utilities/builtin_timer.h\" instead") static double GetCurrentTime()
{
#ifndef _OPENMP
return std::clock()/static_cast<double>(CLOCKS_PER_SEC);
#else
return  omp_get_wtime();
#endif
}


KRATOS_DEPRECATED_MESSAGE("This is legacy, please use the \"ParallelUtilities\" instead")
static inline void DivideInPartitions(
const int NumTerms,
const int NumThreads,
PartitionVector& Partitions)
{
Partitions.resize(NumThreads + 1);
int PartitionSize = NumTerms / NumThreads;
Partitions[0] = 0;
Partitions[NumThreads] = NumTerms;
for(int i = 1; i < NumThreads; i++)
Partitions[i] = Partitions[i-1] + PartitionSize ;
}


template< class TVector >
KRATOS_DEPRECATED_MESSAGE("This is legacy, please use the \"ParallelUtilities\" instead")
static void PartitionedIterators(TVector& rVector,
typename TVector::iterator& rBegin,
typename TVector::iterator& rEnd)
{
#ifdef _OPENMP
int NumTerms = rVector.size();
int ThreadNum = omp_get_thread_num();
int NumThreads = omp_get_max_threads();
int PartitionSize = NumTerms / NumThreads;
rBegin = rVector.begin() + ThreadNum * PartitionSize;
if ( (ThreadNum + 1) != NumThreads )
rEnd = rBegin + PartitionSize;
else
rEnd = rVector.end();
#else
rBegin = rVector.begin();
rEnd = rVector.end();
#endif
}


KRATOS_DEPRECATED_MESSAGE("This is legacy version, please use the \"ParallelUtilities\" instead") static inline void SetNumThreads(int NumThreads = 1)
{
ParallelUtilities::SetNumThreads(NumThreads);
}


static inline void PrintOMPInfo()
{
#ifdef _OPENMP

int nthreads,tid, procs, maxt, inpar, dynamic, nested;



#pragma omp parallel private(nthreads, tid)
{

tid = omp_get_thread_num();


if (tid == 0)
{
printf("  Thread %d getting environment info...\n", tid);


procs    = omp_get_num_procs();
nthreads = omp_get_num_threads();
maxt     = omp_get_max_threads();
inpar    = omp_in_parallel();
dynamic  = omp_get_dynamic();
nested   = omp_get_nested();


printf( "  | ------------ OMP IN USE --------- |\n");
printf( "  | Machine number of processors  = %d |\n", procs);
printf( "  | Number of threads set         = %d |\n", nthreads);
printf( "  | Max threads in use            = %d |\n", maxt);
printf( "  | In parallel?                  = %d |\n", inpar);
printf( "  | Dynamic threads enabled?      = %d |\n", dynamic);
printf( "  | Nested parallelism supported? = %d |\n", nested);
printf( "  | --------------------------------- |\n");


if( procs < nthreads )
std::cout<<" ( WARNING: Maximimun number of threads is EXCEEDED )"<<std::endl;

}

}

#endif
}

template<class T>
KRATOS_DEPRECATED_MESSAGE("This is legacy, please use the \"ParallelUtilities\" instead")
static inline void CreatePartition(unsigned int number_of_threads, const int number_of_rows, T& partitions)
{
partitions.resize(number_of_threads+1);
int partition_size = number_of_rows / number_of_threads;
partitions[0] = 0;
partitions[number_of_threads] = number_of_rows;
for(unsigned int i = 1; i<number_of_threads; i++)
partitions[i] = partitions[i-1] + partition_size ;
}

};


}

#endif	

