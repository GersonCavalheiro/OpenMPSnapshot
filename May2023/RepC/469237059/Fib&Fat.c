#include <time.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
int fibonacci(int n) {
if (n == 1){
return 1; 
} else                                    
if (n == 2) {
return 1;  
} else 
return fibonacci(n - 1) + fibonacci(n - 2); 
}
int fatorial(int n) 
{
int num;
if (num == 1 || num == 0){
return 1;
}
return fatorial(num - 1) * num;
}
int main(void){
omp_set_num_threads(10);   
int n = 49;
clock_t inicio = clock();
clock_t fim = clock();
#pragma omp parallel sections
{
#pragma omp section
{
double duracao = (double)(fim - inicio) / CLOCKS_PER_SEC;
printf("Fibonacci de %d = %d\n", n, fibonacci(n));
printf("Calculado em %.3f segundos \n",duracao);
}
#pragma omp section
{
double duracao = (double)(fim - inicio) / CLOCKS_PER_SEC;
printf("Fatorial de %d = %d\n", n, fatorial(n));
printf("Calculado em %.3f segundos \n",duracao);
}
}
return 0;
}
