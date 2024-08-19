#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <omp.h>
#include "bellard_recursive_power_cyclic.h"
void mpfr_bellard_bit_shift_power_cyclic_algorithm(mpfr_t pi, int num_iterations, int num_threads, int precision_bits){
mpfr_t ONE; 
mpfr_init_set_ui(ONE, 1, MPFR_RNDN); 
omp_set_num_threads(num_threads);
#pragma omp parallel 
{
int thread_id, i, dep_a, dep_b, jump_dep_a, jump_dep_b, next_i;
mpfr_t local_pi, dep_m, a, b, c, d, e, f, g, aux;
thread_id = omp_get_thread_num();
mpfr_inits2(precision_bits, local_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);
mpfr_set_ui(local_pi, 0, MPFR_RNDN);
dep_a = thread_id * 4;
dep_b = thread_id * 10;
jump_dep_a = 4 * num_threads;
jump_dep_b = 10 * num_threads;
mpfr_mul_2exp(dep_m, ONE, 10 * thread_id, MPFR_RNDN);
mpfr_div(dep_m, ONE, dep_m, MPFR_RNDN);
if(thread_id % 2 != 0) mpfr_neg(dep_m, dep_m, MPFR_RNDN);                   
for(i = thread_id; i < num_iterations; i+=num_threads){
mpfr_bellard_iteration(local_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
next_i = i + num_threads;
mpfr_mul_2exp(dep_m, ONE, 10 * next_i, MPFR_RNDN);
mpfr_div(dep_m, ONE, dep_m, MPFR_RNDN);
if (next_i % 2 != 0) mpfr_neg(dep_m, dep_m, MPFR_RNDN); 
dep_a += jump_dep_a;
dep_b += jump_dep_b;  
}
#pragma omp critical
mpfr_add(pi, pi, local_pi, MPFR_RNDN);
mpfr_free_cache();
mpfr_clears(local_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);   
}
mpfr_div_2exp(pi, pi, 6, MPFR_RNDN); 
mpfr_clear(ONE);
}
