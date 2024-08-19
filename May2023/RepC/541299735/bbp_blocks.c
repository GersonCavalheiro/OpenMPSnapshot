#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <omp.h>
#define QUOTIENT 0.0625
void mpfr_bbp_iteration(mpfr_t pi, int n, mpfr_t dep_m, mpfr_t quot_a, mpfr_t quot_b, mpfr_t quot_c, mpfr_t quot_d, mpfr_t aux){
mpfr_set_ui(quot_a, 4, MPFR_RNDN);              
mpfr_set_ui(quot_b, 2, MPFR_RNDN);              
mpfr_set_ui(quot_c, 1, MPFR_RNDN);              
mpfr_set_ui(quot_d, 1, MPFR_RNDN);              
mpfr_set_ui(aux, 0, MPFR_RNDN);                 
int i = n << 3;                     
mpfr_div_ui(quot_a, quot_a, i | 1, MPFR_RNDN);  
mpfr_div_ui(quot_b, quot_b, i | 4, MPFR_RNDN);  
mpfr_div_ui(quot_c, quot_c, i | 5, MPFR_RNDN);  
mpfr_div_ui(quot_d, quot_d, i | 6, MPFR_RNDN);  
mpfr_sub(aux, quot_a, quot_b, MPFR_RNDN);
mpfr_sub(aux, aux, quot_c, MPFR_RNDN);
mpfr_sub(aux, aux, quot_d, MPFR_RNDN);
mpfr_mul(aux, aux, dep_m, MPFR_RNDN);   
mpfr_add(pi, pi, aux, MPFR_RNDN);  
}
void mpfr_bbp_blocks_algorithm(mpfr_t pi, int num_iterations, int num_threads, int precision_bits){
mpfr_t quotient; 
mpfr_init_set_d(quotient, QUOTIENT, MPFR_RNDN);         
omp_set_num_threads(num_threads);
#pragma omp parallel 
{
int thread_id, i, block_size, block_start, block_end;
mpfr_t local_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux;
thread_id = omp_get_thread_num();
block_size = (num_iterations + num_threads - 1) / num_threads;
block_start = thread_id * block_size;
block_end = block_start + block_size;
if (block_end > num_iterations) block_end = num_iterations;
mpfr_inits2(precision_bits, local_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux, NULL);
mpfr_set_ui(local_pi, 0, MPFR_RNDN);
mpfr_pow_ui(dep_m, quotient, block_start, MPFR_RNDN);    
for(i = block_start; i < block_end; i++){
mpfr_bbp_iteration(local_pi, i, dep_m, quot_a, quot_b, quot_c, quot_d, aux);
mpfr_mul(dep_m, dep_m, quotient, MPFR_RNDN);
}
#pragma omp critical
mpfr_add(pi, pi, local_pi, MPFR_RNDN);
mpfr_free_cache();
mpfr_clears(local_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux, NULL);   
}
mpfr_clear(quotient);
}