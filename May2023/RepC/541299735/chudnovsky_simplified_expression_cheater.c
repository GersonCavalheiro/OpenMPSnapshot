#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "chudnovsky_simplified_expression_blocks.h"
#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005
int * get_thread_distribution(int num_threads, int thread_id, int num_iterations){
int * distribution, i, block_size, block_start, block_end, row, column; 
FILE * ratios_file;
float working_ratios[160][41];
ratios_file = fopen("resources/working_ratios.txt", "r");
if(ratios_file == NULL){
printf("working_ratios.txt not found \n");
exit(-1);
} 
row = 0;
while (fscanf(ratios_file, "%f", &working_ratios[row][0]) == 1){
for (column = 1; column < 41; column++){
fscanf(ratios_file, "%f", &working_ratios[row][column]);
}
row++;
}
distribution = malloc(sizeof(int) * 3);
if(num_threads == 1){
distribution[0] = num_iterations;
distribution[1] = 0;
distribution[2] = num_iterations;
return distribution; 
}
block_size = working_ratios[thread_id][num_threads / 4] * num_iterations / 100;
block_start = 0;
for(i = 0; i < thread_id; i ++){
block_start += working_ratios[i][num_threads / 4] * num_iterations / 100;
}
block_end = block_start + block_size;
if (thread_id == num_threads -1) block_end = num_iterations;
distribution[0] = block_size;
distribution[1] = block_start;
distribution[2] = block_end;
return distribution;
}
void gmp_chudnovsky_simplified_expression_cheater_algorithm(mpf_t pi, int num_iterations, int num_threads){
mpf_t e, c;
mpf_init_set_ui(e, E);
mpf_init_set_ui(c, C);
mpf_neg(c, c);
mpf_pow_ui(c, c, 3);
omp_set_num_threads(num_threads);
#pragma omp parallel 
{   
int thread_id, i, block_size, block_start, block_end, factor_a, * distribution;
mpf_t local_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux;
thread_id = omp_get_thread_num();
distribution = get_thread_distribution(num_threads, thread_id, num_iterations);
block_size = distribution[0];
block_start = distribution[1];
block_end = distribution[2];
mpf_init_set_ui(local_pi, 0);    
mpf_inits(dep_a, dep_b, dep_a_dividend, dep_a_divisor, aux, NULL);
gmp_init_dep_a(dep_a, block_start);
mpf_pow_ui(dep_b, c, block_start);
mpf_init_set_ui(dep_c, B);
mpf_mul_ui(dep_c, dep_c, block_start);
mpf_add_ui(dep_c, dep_c, A);
factor_a = 12 * block_start;
for(i = block_start; i < block_end; i++){
gmp_chudnovsky_iteration(local_pi, i, dep_a, dep_b, dep_c, aux);
mpf_set_ui(dep_a_dividend, factor_a + 10);
mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
mpf_mul(dep_a_dividend, dep_a_dividend, dep_a);
mpf_set_ui(dep_a_divisor, i + 1);
mpf_pow_ui(dep_a_divisor, dep_a_divisor ,3);
mpf_div(dep_a, dep_a_dividend, dep_a_divisor);
factor_a += 12;
mpf_mul(dep_b, dep_b, c);
mpf_add_ui(dep_c, dep_c, B);
}
#pragma omp critical
mpf_add(pi, pi, local_pi);
mpf_clears(local_pi, dep_a, dep_b, dep_c, dep_a_dividend, dep_a_divisor, aux, NULL);   
}
mpf_sqrt(e, e);
mpf_mul_ui(e, e, D);
mpf_div(pi, e, pi);    
mpf_clears(c, e, NULL);
}