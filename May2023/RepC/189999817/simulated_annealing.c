#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "CSA_Problem.h"
#ifndef PI
#define PI 3.14159265358979323846264338327
#endif
double maxValue(double myArray[], int size) {
assert(myArray && size);
int i;
double maxValue = myArray[0];
for (i = 1; i < size; ++i) {
if ( myArray[i] > maxValue ) {
maxValue = myArray[i];
}
}
return maxValue;
}
double minValue(double myArray[], int size) {
assert(myArray && size);
int i, j = 0;
double minValue = myArray[0];
for (i = 1; i < size; i++) {
if ( myArray[i] < minValue ) {
minValue = myArray[i];
j = i;
}
}
return j;
}
int main(int argc, char* argv[]) {
double t_gen = 100; 
double t_ac = 100; 
double num_aleatorio = 0.0; 
double custo_sol_corrente, custo_sol_nova, melhor_custo; 
int dim; 
int i; 
double num_otimizadores = 0; 
int num_function_choices[3] = {2001,2003,2006}; 
int num_function; 
int avaliacoes = 0; 
int menor_custo_total; 
num_otimizadores = (double) atoi(argv[1]); 
dim = atoi(argv[2]); 
num_function = num_function_choices[atoi(argv[3])]; 
double var_desejada = 0.99 * ((num_otimizadores - 1)/(num_otimizadores * num_otimizadores )); 
double *sol_corrente; 
double *sol_nova; 
double *tmp = NULL; 
double *vetor_func_prob = (double *)malloc(num_otimizadores * sizeof(double)); 
double *atuais_custos = (double *)malloc(num_otimizadores * sizeof(double)); 
double termo_acoplamento; 
double sigma;
double total_time; 
double start; 
struct drand48_data buffer; 
#pragma omp parallel num_threads((int)num_otimizadores) default(none) shared(start, total_time, vetor_func_prob, sigma, menor_custo_total, k, termo_acoplamento, avaliacoes, t_gen, t_ac, atuais_custos, num_otimizadores, dim, num_function, var_desejada) private(num_aleatorio, tmp, buffer, melhor_custo, custo_sol_corrente, sol_corrente, sol_nova, custo_sol_nova, i)
{
#pragma omp single
{
start = omp_get_wtime();
}
sol_corrente = (double *)malloc(dim * sizeof(double)); 
sol_nova = (double *)malloc(dim * sizeof(double)); 
int my_rank = omp_get_thread_num(); 
srand48_r(time(NULL)+my_rank*my_rank,&buffer); 
for (i = 0; i < dim; i++){
drand48_r(&buffer, &num_aleatorio); 
sol_corrente[i] = 2.0*num_aleatorio-1.0;
if (sol_corrente[i] < -1.0 || sol_corrente[i] > 1.0) {
printf("Erro no limite da primeira solucao corrente: \n");
exit(0);
}
}
custo_sol_corrente = CSA_EvalCost(sol_corrente, dim, num_function); 
#pragma omp atomic
avaliacoes++;
atuais_custos[my_rank] = custo_sol_corrente;
melhor_custo = custo_sol_corrente;
#pragma omp barrier
#pragma omp single private(i)
{
termo_acoplamento = 0;
for (i = 0; i < (int) num_otimizadores; i++) {
termo_acoplamento += pow(EULLER, ((atuais_custos[i] - maxValue(atuais_custos, (int)num_otimizadores))/t_ac));
}
} 
double func_prob = pow(EULLER, ((custo_sol_corrente - maxValue(atuais_custos, (int)num_otimizadores))/t_ac))/termo_acoplamento;
if (func_prob < 0 || func_prob > 1){
printf("Limite errado da função de probabilidade\n");
exit(0);
}
vetor_func_prob[my_rank] = func_prob;
while(avaliacoes < 1000000){
for (i = 0; i < dim; i++) {
drand48_r(&buffer, &num_aleatorio); 
sol_nova[i] = fmod((sol_corrente[i] + t_gen * tan(PI*(num_aleatorio-0.5))), 1.0);
if (sol_nova[i] > 1.0 || sol_nova[i] < -1.0) {
printf("Intervalo de soluções mal definido!\n");
exit(0);
}
}
custo_sol_nova = CSA_EvalCost(sol_nova, dim, num_function);
#pragma omp atomic
avaliacoes++;
drand48_r(&buffer, &num_aleatorio);
if (num_aleatorio > 1 || num_aleatorio < 0) {
printf("ERRO NO LIMITE DE R = %f\n", num_aleatorio);
exit(0);
}
if (custo_sol_nova <= custo_sol_corrente || func_prob > num_aleatorio){
tmp = sol_corrente;
sol_corrente = sol_nova;
sol_nova = tmp;
custo_sol_corrente = custo_sol_nova;
atuais_custos[my_rank] = custo_sol_nova;
if (melhor_custo > custo_sol_nova) {
melhor_custo = custo_sol_nova;
}
}
#pragma omp barrier
#pragma omp single private(i)
{
termo_acoplamento = 0;
for (i = 0; i < (int) num_otimizadores; i++) {
termo_acoplamento += pow(EULLER, ((atuais_custos[i] - maxValue(atuais_custos, (int) num_otimizadores))/t_ac));
}
} 
func_prob = pow(EULLER, ((custo_sol_corrente - maxValue(atuais_custos, (int) num_otimizadores))/t_ac))/termo_acoplamento;
if (func_prob < 0 || func_prob > 1){
printf("Limite errado da função de probabilidade\n");
exit(0);
}
vetor_func_prob[my_rank] = func_prob;
#pragma omp barrier
#pragma omp single private(i)
{
sigma = 0;
for (i = 0; i < (int) num_otimizadores; i++) {
sigma += (double) pow(vetor_func_prob[i], 2);
}
sigma = ((1/num_otimizadores) * sigma) - 1/(num_otimizadores * num_otimizadores);
double sigma_limit = ((num_otimizadores - 1)/(num_otimizadores * num_otimizadores));
if (sigma < 0 || sigma > sigma_limit){
printf("Limite errado de sigma. sigma = %f, iteracao = %d\n", sigma, k);
exit(0);
}
if (sigma < var_desejada){
t_ac = t_ac * (1 - 0.01);
} else if (sigma > var_desejada){
t_ac = t_ac * (1 + 0.01);
}
t_gen = 0.99992 * t_gen;
} 
}
atuais_custos[my_rank] = melhor_custo;
#pragma omp barrier
#pragma omp single private(i)
{
menor_custo_total = minValue(atuais_custos, (int) num_otimizadores);
} 
if (menor_custo_total == my_rank) {
total_time = omp_get_wtime() - start;
FILE *tempo;
tempo = fopen("/home/vgdmenezes/csa-openmp/runtimes/tempo_de_exec_omp_csa.txt", "a");
fprintf(tempo, "Melhor custo = %1.5e\t\tRuntime = %1.2e\n", melhor_custo, total_time);
fclose(tempo);
}
} 
return 0;
}  
