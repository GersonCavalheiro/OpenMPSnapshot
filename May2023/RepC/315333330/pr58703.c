#pragma omp declare reduction (+ : char[] : omp_out += omp_in) 
#pragma omp declare reduction (+ : char() : omp_out += omp_in) 
