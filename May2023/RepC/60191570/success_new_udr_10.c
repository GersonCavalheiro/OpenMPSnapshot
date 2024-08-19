#pragma omp declare reduction(foo : float : omp_in *= omp_out )
