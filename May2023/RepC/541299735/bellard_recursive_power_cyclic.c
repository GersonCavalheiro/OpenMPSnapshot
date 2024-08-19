#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <omp.h>
void mpfr_bellard_iteration(mpfr_t pi, int n, mpfr_t m, mpfr_t a, mpfr_t b, mpfr_t c, mpfr_t d, 
mpfr_t e, mpfr_t f, mpfr_t g, mpfr_t aux, int dep_a, int dep_b){
mpfr_set_ui(a, 32, MPFR_RNDN);              
mpfr_set_ui(b, 1, MPFR_RNDN);               
mpfr_set_ui(c, 256, MPFR_RNDN);             
mpfr_set_ui(d, 64, MPFR_RNDN);              
mpfr_set_ui(e, 4, MPFR_RNDN);               
mpfr_set_ui(f, 4, MPFR_RNDN);               
mpfr_set_ui(g, 1, MPFR_RNDN);               
mpfr_set_ui(aux, 0, MPFR_RNDN);             
mpfr_div_ui(a, a, dep_a + 1, MPFR_RNDN);    
mpfr_div_ui(b, b, dep_a + 3, MPFR_RNDN);    
mpfr_div_ui(c, c, dep_b + 1, MPFR_RNDN);    
mpfr_div_ui(d, d, dep_b + 3, MPFR_RNDN);    
mpfr_div_ui(e, e, dep_b + 5, MPFR_RNDN);    
mpfr_div_ui(f, f, dep_b + 7, MPFR_RNDN);    
mpfr_div_ui(g, g, dep_b + 9, MPFR_RNDN);    
mpfr_neg(a, a, MPFR_RNDN);
mpfr_sub(aux, a, b, MPFR_RNDN);
mpfr_sub(c, c, d, MPFR_RNDN);
mpfr_sub(c, c, e, MPFR_RNDN);
mpfr_sub(c, c, f, MPFR_RNDN);
mpfr_add(c, c, g, MPFR_RNDN);
mpfr_add(aux, aux, c, MPFR_RNDN);
mpfr_mul(aux, aux, m, MPFR_RNDN);   
mpfr_add(pi, pi, aux, MPFR_RNDN); 
}
void mpfr_bellard_recursive_power_cyclic_algorithm(mpfr_t pi, int num_iterations, int num_threads, int precision_bits){
mpfr_t jump; 
mpfr_init_set_ui(jump, 1, MPFR_RNDN);
mpfr_div_ui(jump, jump, 1024, MPFR_RNDN);
mpfr_pow_ui(jump, jump, num_threads, MPFR_RNDN);
omp_set_num_threads(num_threads);
#pragma omp parallel 
{
int thread_id, i, dep_a, dep_b, jump_dep_a, jump_dep_b;
mpfr_t local_pi, dep_m, a, b, c, d, e, f, g, aux;
thread_id = omp_get_thread_num();
mpfr_inits2(precision_bits, local_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);
mpfr_set_ui(local_pi, 0, MPFR_RNDN);               
dep_a = thread_id * 4;
dep_b = thread_id * 10;
jump_dep_a = 4 * num_threads;
jump_dep_b = 10 * num_threads;
mpfr_set_ui(dep_m, 1, MPFR_RNDN);
mpfr_div_ui(dep_m, dep_m, 1024, MPFR_RNDN);
mpfr_pow_ui(dep_m, dep_m, thread_id, MPFR_RNDN);        
if(thread_id % 2 != 0) mpfr_neg(dep_m, dep_m, MPFR_RNDN);                   
if(num_threads % 2 != 0){
for(i = thread_id; i < num_iterations; i+=num_threads){
mpfr_bellard_iteration(local_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
mpfr_mul(dep_m, dep_m, jump, MPFR_RNDN); 
mpfr_neg(dep_m, dep_m, MPFR_RNDN); 
dep_a += jump_dep_a;
dep_b += jump_dep_b;  
}
} else {
for(i = thread_id; i < num_iterations; i+=num_threads){
mpfr_bellard_iteration(local_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
mpfr_mul(dep_m, dep_m, jump, MPFR_RNDN);    
dep_a += jump_dep_a;
dep_b += jump_dep_b;  
}
}
#pragma omp critical
mpfr_add(pi, pi, local_pi, MPFR_RNDN);
mpfr_free_cache();
mpfr_clears(local_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);   
}
mpfr_div_ui(pi, pi, 64, MPFR_RNDN);
mpfr_clear(jump);
}
