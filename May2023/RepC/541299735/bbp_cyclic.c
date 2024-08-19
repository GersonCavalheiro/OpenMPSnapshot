#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#define QUOTIENT 0.0625
void gmp_bbp_iteration(mpf_t pi, int n, mpf_t dep_m, mpf_t quot_a, mpf_t quot_b, mpf_t quot_c, mpf_t quot_d, mpf_t aux){
mpf_set_ui(quot_a, 4);              
mpf_set_ui(quot_b, 2);              
mpf_set_ui(quot_c, 1);              
mpf_set_ui(quot_d, 1);              
mpf_set_ui(aux, 0);                 
int i = n << 3;                     
mpf_div_ui(quot_a, quot_a, i | 1);  
mpf_div_ui(quot_b, quot_b, i | 4);  
mpf_div_ui(quot_c, quot_c, i | 5);  
mpf_div_ui(quot_d, quot_d, i | 6);  
mpf_sub(aux, quot_a, quot_b);
mpf_sub(aux, aux, quot_c);
mpf_sub(aux, aux, quot_d);
mpf_mul(aux, aux, dep_m);   
mpf_add(pi, pi, aux);  
}
void gmp_bbp_cyclic_algorithm(mpf_t pi, int num_iterations, int num_threads){
mpf_t jump, quotient; 
mpf_init_set_d(quotient, QUOTIENT);         
mpf_init_set_ui(jump, 1);        
mpf_pow_ui(jump, quotient, num_threads);    
omp_set_num_threads(num_threads);
#pragma omp parallel 
{
int thread_id, i;
mpf_t local_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux;
thread_id = omp_get_thread_num();
mpf_init_set_ui(local_pi, 0);               
mpf_init(dep_m);
mpf_pow_ui(dep_m, quotient, thread_id);    
mpf_inits(quot_a, quot_b, quot_c, quot_d, aux, NULL);
for(i = thread_id; i < num_iterations; i+=num_threads){
gmp_bbp_iteration(local_pi, i, dep_m, quot_a, quot_b, quot_c, quot_d, aux);
mpf_mul(dep_m, dep_m, jump);
}
#pragma omp critical
mpf_add(pi, pi, local_pi);
mpf_clears(local_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux, NULL);   
}
mpf_clears(jump, quotient, NULL);
}