#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/types.h>
#define row_matriz_A 800
#define col_matriz_A 800
#define upper 1000
#define lower 0
#define THREADS_NUM 4
void gerarMatrizes(int r, int c, int mA[], int vX[]){
int i;               
int j;               
srandom(time(0));    
for(i = 0; i < r; i++){
for(j = 0; j < c; j++){
mA[i * c + j] = (random() % (upper - lower + 1)) + lower;
}
}
for(j = 0; j < c; j++) vX[j] = (random() % (upper - lower + 1)) + lower; 
return;
}
void multSequencial(int r, int c, int mA[], int vX[], int vBS[]){
int i;               
int j;               
int sum;             
struct timeval tv1;  
struct timeval tv2;  
double t1;           
double t2;           
gettimeofday(&tv1, NULL);
t1 = (double)(tv1.tv_sec) + (double)(tv1.tv_usec)/ 1000000.00;
for(i = 0; i < r; i++){
sum = 0;
for(j = 0; j < c; j++){
sum += mA[i * c + j] * vX[j];
}
vBS[i] = sum;
}
gettimeofday(&tv2, NULL);
t2 = (double)(tv2.tv_sec) + (double)(tv2.tv_usec)/ 1000000.00;
printf("%lf;", (t2 - t1));
return;
}
void matvecHost(int r, int c, int mA[], int vX[], int vBP[]){
int i;               
int j;               
int sum;             
struct timeval tv1;  
struct timeval tv2;  
double t1;           
double t2;           
gettimeofday(&tv1, NULL);
t1 = (double)(tv1.tv_sec) + (double)(tv1.tv_usec)/ 1000000.00;
omp_set_num_threads(THREADS_NUM);
#pragma omp parallel private(i, j, sum) shared(mA, vBP, vX)
{
#pragma omp for
for(i = 0; i < r; i++){
sum = 0;
for(j = 0; j < c; j++){
sum += mA[i * c + j] * vX[j];
}
vBP[i] = sum;
}
}
gettimeofday(&tv2, NULL);
t2 = (double)(tv2.tv_sec) + (double)(tv2.tv_usec)/ 1000000.00;  
printf("%lf\n", (t2 - t1));
return;
}
void matvecDevice(int r, int c, int mA[], int vX[], int vBP[]) {
printf("matvecDevice\n");
return;
}
int main(int argc, char * argv[]) {
int i;
int b;
int row;
int col;
int verifyB = 0;
int benchmarks = 10;
row = row_matriz_A;
col = col_matriz_A;
int *matrizA = (int *)calloc(row * col, sizeof(int));
int *vectorX = (int *)calloc(col * 1, sizeof(int));
int *vectorBS = (int *)calloc(row * 1, sizeof(int));
int *vectorBP = (int *)calloc(row * 1, sizeof(int));
int opcao = atoi(argv[1]);
printf("linhas - %d | colunas - %d\nbenchmark;serial;paralelismo\n",row,col);
for(b = 0; b < benchmarks; b++){
gerarMatrizes(row, col, matrizA, vectorX);
printf("%d;",b);
multSequencial(row, col, matrizA, vectorX, vectorBS);
if(opcao == 1) matvecHost(row, col, matrizA, vectorX, vectorBP);
else if(opcao == 2) matvecDevice(row, col, matrizA, vectorX, vectorBP);
for(i=0;i<row;i++){
if(vectorBS[i] != vectorBP[i]){
printf("Elemento distinto | i = %d | BS[i] = %d | BP[i] = %d\n",i,vectorBS[i],vectorBP[i]);
verifyB = 1;
}
}
if(verifyB == 1) printf("ERROR - Vetores B distintos!\n");
}
return 0;
}