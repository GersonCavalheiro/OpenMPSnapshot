#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <mpfr.h>
#include <omp.h>
#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005
void mpfr_chudnovsky_iteration(mpfr_t pi, int n, mpfr_t dep_a, mpfr_t dep_b, 
mpfr_t dep_c, mpfr_t aux){
mpfr_mul(aux, dep_a, dep_c, MPFR_RNDN);
mpfr_div(aux, aux, dep_b, MPFR_RNDN);
mpfr_add(pi, pi, aux, MPFR_RNDN);
}
void mpfr_init_dep_a(mpfr_t dep_a, int block_start, int precision_bits){
mpz_t factorial_n, dividend, divisor;
mpfr_t float_dividend, float_divisor;
mpz_inits(factorial_n, dividend, divisor, NULL);
mpfr_inits2(precision_bits, float_dividend, float_divisor, NULL);
mpz_fac_ui(factorial_n, block_start);
mpz_fac_ui(divisor, 3 * block_start);
mpz_fac_ui(dividend, 6 * block_start);
mpz_pow_ui(factorial_n, factorial_n, 3);
mpz_mul(divisor, divisor, factorial_n);
mpfr_set_z(float_dividend, dividend, MPFR_RNDN);
mpfr_set_z(float_divisor, divisor, MPFR_RNDN);
mpfr_div(dep_a, float_dividend, float_divisor, MPFR_RNDN);
mpz_clears(factorial_n, dividend, divisor, NULL);
mpfr_clears(float_dividend, float_divisor, NULL);
}
void mpfr_chudnovsky_simplified_expression_blocks_algorithm(mpfr_t pi, int num_iterations, int num_threads, int precision_bits){
mpfr_t e, c;
mpfr_inits2(precision_bits, e, c, NULL);
mpfr_set_ui(e, E, MPFR_RNDN);
mpfr_set_ui(c, C, MPFR_RNDN);
mpfr_neg(c, c, MPFR_RNDN);
mpfr_pow_ui(c, c, 3, MPFR_RNDN);
omp_set_num_threads(num_threads);
#pragma omp parallel 
{   
int thread_id, i, block_size, block_start, block_end, factor_a, * distribution;
mpfr_t local_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux;
thread_id = omp_get_thread_num();
block_size = (num_iterations + num_threads - 1) / num_threads;
block_start = thread_id * block_size;
block_end = block_start + block_size;
if (block_end > num_iterations) block_end = num_iterations;
mpfr_inits2(precision_bits, local_pi, dep_a, dep_b, dep_c, dep_a_dividend, dep_a_divisor, aux, NULL);
mpfr_set_ui(local_pi, 0, MPFR_RNDN);    
mpfr_init_dep_a(dep_a, block_start, precision_bits);
mpfr_pow_ui(dep_b, c, block_start, MPFR_RNDN);
mpfr_set_ui(dep_c, B, MPFR_RNDN);
mpfr_mul_ui(dep_c, dep_c, block_start, MPFR_RNDN);
mpfr_add_ui(dep_c, dep_c, A, MPFR_RNDN);
factor_a = 12 * block_start;
for(i = block_start; i < block_end; i++){
mpfr_chudnovsky_iteration(local_pi, i, dep_a, dep_b, dep_c, aux);
mpfr_set_ui(dep_a_dividend, factor_a + 10, MPFR_RNDN);
mpfr_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6, MPFR_RNDN);
mpfr_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2, MPFR_RNDN);
mpfr_mul(dep_a_dividend, dep_a_dividend, dep_a, MPFR_RNDN);
mpfr_set_ui(dep_a_divisor, i + 1, MPFR_RNDN);
mpfr_pow_ui(dep_a_divisor, dep_a_divisor , 3, MPFR_RNDN);
mpfr_div(dep_a, dep_a_dividend, dep_a_divisor, MPFR_RNDN);
factor_a += 12;
mpfr_mul(dep_b, dep_b, c, MPFR_RNDN);
mpfr_add_ui(dep_c, dep_c, B, MPFR_RNDN);
}
#pragma omp critical
mpfr_add(pi, pi, local_pi, MPFR_RNDN);
mpfr_clears(local_pi, dep_a, dep_b, dep_c, dep_a_dividend, dep_a_divisor, aux, NULL);   
}
mpfr_sqrt(e, e, MPFR_RNDN);
mpfr_mul_ui(e, e, D, MPFR_RNDN);
mpfr_div(pi, e, pi, MPFR_RNDN);    
mpfr_clears(c, e, NULL);
}