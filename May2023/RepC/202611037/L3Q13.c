#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
int primo (int n);
int main(int argc, char *argv[]){
int n, i, total = 1, num_threads = 1;     
double tempo_inicial, tempo_final;
if (argc < 2) {
printf("Valor inválido! Entre com o valor do maior inteiro\n");
return 0;
} else {
num_threads = strtol(argv[1], (char **) NULL, 10);
n = strtol(argv[2], (char **) NULL, 10);
}
omp_set_num_threads(num_threads);
tempo_inicial = omp_get_wtime();
#pragma omp parallel for reduction(+:total) schedule(guided,10)
for (i = 3; i <= n; i += 2) {
if(primo(i) == 1) total++;
}
tempo_final = omp_get_wtime();
printf("São %d primos até %d.\nThreads: %d | Tempo gasto: %lf segundos\n", total, n, num_threads, (double) (tempo_final - tempo_inicial));
return 0;
}
int primo (int n) {
int i; 
for (i = 3; i < (int)(sqrt(n) + 1); i+=2){
if(n%i == 0) return 0;
}
return 1;
}