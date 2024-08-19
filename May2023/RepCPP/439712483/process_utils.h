#pragma once

#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#elif USE_OMP
#include <omp.h>
#endif


inline int compute_max_depth(int n_processes) {
return log2((double)n_processes);
}


inline int compute_n_surplus_processes(int n_processes, int max_depth) {
return n_processes - (int)pow(2.0, (double)max_depth);
}


inline int compute_next_process_rank(int rank, int max_depth, int next_depth,
int surplus_processes, int n_processes) {
if (next_depth < max_depth + 1)
return rank + pow(2.0, max_depth - next_depth);
else if (next_depth == max_depth + 1 && rank < surplus_processes)
return n_processes - surplus_processes + rank;
else
return -1;
}


inline int get_n_parallel_workers() {
#ifdef USE_MPI
int n_processes;
MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
return n_processes;
#elif USE_OMP
return omp_get_num_threads();
#else
return 1;
#endif
}


inline int get_rank() {
#ifdef USE_MPI
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
return rank;
#elif USE_OMP
return omp_get_thread_num();
#else
return 0;
#endif
}
