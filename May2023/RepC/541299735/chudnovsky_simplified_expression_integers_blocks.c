#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005
void gmp_chudnovsky_simplified_expression_integers_iteration(mpf_t pi, int n, mpz_t dep_a, mpz_t dep_b, mpz_t dep_c, mpz_t aux, 
mpf_t float_aux, mpf_t float_dep_b){
mpz_mul(aux, dep_a, dep_c);
mpf_set_z(float_aux, aux);
mpf_set_z(float_dep_b, dep_b);   
mpf_div(float_aux, float_aux, float_dep_b); 
mpf_add(pi, pi, float_aux);
}
void mpz_compute_dep_a_gmp(mpz_t dep_a, int n){
mpz_t factorial_n, dividend, divisor;
mpz_inits(factorial_n, dividend, divisor, NULL);
mpz_fac_ui(factorial_n, n);
mpz_fac_ui(divisor, 3 * n);
mpz_fac_ui(dividend, 6 * n);
mpz_pow_ui(factorial_n, factorial_n, 3);
mpz_mul(divisor, divisor, factorial_n);
mpz_div(dep_a, dividend, divisor);
mpz_clears(factorial_n, dividend, divisor, NULL);
}
void gmp_chudnovsky_simplified_expression_integers_blocks_algorithm(mpf_t pi, int num_iterations, int num_threads){
mpf_t e;
mpz_t c;
mpf_init_set_ui(e, E);
mpz_init_set_ui(c, C);
mpz_neg(c, c);
mpz_pow_ui(c, c, 3);
omp_set_num_threads(num_threads);
#pragma omp parallel 
{   
int thread_id, i, block_size, block_start, block_end, factor_a;
mpf_t local_pi, float_aux, float_dep_b;
mpz_t dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux;
thread_id = omp_get_thread_num();
block_size = (num_iterations + num_threads - 1) / num_threads;
block_start = thread_id * block_size;
block_end = block_start + block_size;
if (block_end > num_iterations) block_end = num_iterations;
mpf_init_set_ui(local_pi, 0);    
mpf_inits(float_aux, float_dep_b, NULL);
mpz_inits(dep_a, dep_b, dep_a_dividend, dep_a_divisor, aux, NULL);
mpz_compute_dep_a_gmp(dep_a, block_start);
mpz_pow_ui(dep_b, c, block_start);
mpz_init_set_ui(dep_c, B);
mpz_mul_ui(dep_c, dep_c, block_start);
mpz_add_ui(dep_c, dep_c, A);
factor_a = 12 * block_start;
for(i = block_start; i < block_end; i++){
gmp_chudnovsky_simplified_expression_integers_iteration(local_pi, i, dep_a, dep_b, dep_c, aux, float_aux, float_dep_b);
mpz_set_ui(dep_a_dividend, factor_a + 10);
mpz_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
mpz_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
mpz_mul(dep_a_dividend, dep_a_dividend, dep_a);
mpz_set_ui(dep_a_divisor, i + 1);
mpz_pow_ui(dep_a_divisor, dep_a_divisor ,3);
mpz_div(dep_a, dep_a_dividend, dep_a_divisor);
factor_a += 12;
mpz_mul(dep_b, dep_b, c);
mpz_add_ui(dep_c, dep_c, B);
}
#pragma omp critical
mpf_add(pi, pi, local_pi);
mpf_clears(local_pi, float_aux, float_dep_b, NULL);  
mpz_clears(dep_a, dep_b, dep_c, dep_a_dividend, dep_a_divisor, aux, NULL);  
}
mpf_sqrt(e, e);
mpf_mul_ui(e, e, D);
mpf_div(pi, e, pi);    
mpf_clear(e);
mpz_clear(c);
}