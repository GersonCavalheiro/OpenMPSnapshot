#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005
void gmp_get_factorials(mpf_t * factorials, int num_factorials){
int i;
mpf_t f;
mpf_init_set_ui(f, 1);
mpf_init_set_ui(factorials[0], 1);
for(i = 1; i <= num_factorials; i++){
mpf_mul_ui(f, f, i);
mpf_init_set(factorials[i], f);
}
mpf_clear(f);
}
void gmp_clear_factorials(mpf_t * factorials, int num_factorials){
int i;
for(i = 0; i <= num_factorials; i++){
mpf_clear(factorials[i]);
}
}
void gmp_chudnovsky_all_factorials_iteration(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, mpf_t dep_c, 
mpf_t dep_d, mpf_t dep_e, mpf_t dividend, mpf_t divisor){
mpf_mul(dividend, dep_a, dep_e);
mpf_mul(divisor, dep_b, dep_c);
mpf_mul(divisor, divisor, dep_d);
mpf_div(dividend, dividend, divisor);
mpf_add(pi, pi, dividend);
}
void gmp_chudnovsky_all_factorials_blocks_algorithm(mpf_t pi, int num_iterations, int num_threads){
mpf_t e, c;
int num_factorials, block_size;
num_factorials = num_iterations * 6;
mpf_t factorials[num_factorials + 1];
gmp_get_factorials(factorials, num_factorials);
block_size = (num_iterations + num_threads - 1) / num_threads;
mpf_init_set_ui(e, E);
mpf_init_set_ui(c, C);
mpf_neg(c, c);
mpf_pow_ui(c, c, 3);
omp_set_num_threads(num_threads);
#pragma omp parallel 
{   
int thread_id, i, block_start, block_end;
mpf_t local_pi, dep_a, dep_b, dep_c, dep_d, dep_e, dividend, divisor;
thread_id = omp_get_thread_num();
block_start = thread_id * block_size;
block_end = block_start + block_size;
if (block_end > num_iterations) block_end = num_iterations;
mpf_init_set_ui(local_pi, 0);    
mpf_inits(dividend, divisor, NULL);
mpf_init_set(dep_a, factorials[block_start * 6]);
mpf_init_set(dep_b, factorials[block_start]);
mpf_pow_ui(dep_b, dep_b, 3);
mpf_init_set(dep_c, factorials[block_start * 3]);
mpf_init_set_ui(dep_d, C);
mpf_neg(dep_d, dep_d);
mpf_pow_ui(dep_d, dep_d, block_start * 3);
mpf_init_set_ui(dep_e, B);
mpf_mul_ui(dep_e, dep_e, block_start);
mpf_add_ui(dep_e, dep_e, A);
for(i = block_start; i < block_end; i++){
gmp_chudnovsky_all_factorials_iteration(local_pi, i, dep_a, dep_b, dep_c, dep_d, dep_e, dividend, divisor);
mpf_set(dep_a, factorials[6 * (i + 1)]);
mpf_pow_ui(dep_b, factorials[i + 1], 3);
mpf_set(dep_c, factorials[3 * (i + 1)]);
mpf_mul(dep_d, dep_d, c);
mpf_add_ui(dep_e, dep_e, B);
}
#pragma omp critical
mpf_add(pi, pi, local_pi);
mpf_clears(local_pi, dep_a, dep_b, dep_c, dep_d, dep_e, dividend, divisor, NULL);   
}
mpf_sqrt(e, e);
mpf_mul_ui(e, e, D);
mpf_div(pi, e, pi);    
gmp_clear_factorials(factorials, num_factorials);
mpf_clears(c, e, NULL);
}
