#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
int main(int argc, char* argv[]) {
int i, j;
int **a;
int **b;
int **c;
double total_time;
double start;
int thread_count = strtol(argv[1], NULL, 10);
int size_matrix = atoi(argv[2]);
a = (int**)malloc(size_matrix * sizeof(int*));
b = (int**)malloc(size_matrix * sizeof(int*));
c = (int**)malloc(size_matrix * sizeof(int*));
for(i = 0; i < size_matrix; i++){
a[i] = (int*)malloc(size_matrix * sizeof(int));
b[i] = (int*)malloc(size_matrix * sizeof(int));
c[i] = (int*)malloc(size_matrix * sizeof(int));
} 
srand(time(NULL));
#pragma omp parallel num_threads(thread_count) default(none) shared(a,b,c,start) private(i,j) firstprivate(size_matrix)
{
#pragma omp parallel for default(none) firstprivate(size_matrix) private(i,j) shared(a,b,c)
for (i = 0; i < size_matrix; i++) {
for (j = 0; j < size_matrix; j++) {
a[i][j] = rand() % 100;
b[i][j] = rand() % 100;
c[i][j] = -1;
}
}
#pragma omp single
{
start = omp_get_wtime(); 
}    
#pragma omp single
{
for (i = 0; i < size_matrix; i++) {
for (j = 0; j < size_matrix; j++) {
#pragma omp task firstprivate(i,j)
{
int k, mult_result = 0;
for (k = 0; k < size_matrix; k++){
mult_result += a[i][k] * b[k][j];
}
c[i][j] = mult_result;
} 
}
}
} 
} 
total_time = omp_get_wtime() - start;
FILE *tempo;
tempo = fopen("/home/vgdmenezes/openmp/runtimes/tempo_de_exec_omp.txt", "a");
fprintf(tempo,"Problem Size = %d ----- Thread number = %d ----- Runtime = %f\n", size_matrix, thread_count, total_time);
fclose(tempo);
return 0;
}  
