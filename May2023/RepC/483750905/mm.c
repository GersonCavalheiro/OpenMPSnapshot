#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <time.h>
#define N 5
#define RAND_MAX 10
int**generateMatrix(int n, bool is_out);
void showMatrix(int**matrix , int n);
int** multMatrix(int**matrix_A , int**matrix_B , int n, int p);
void generadorMultiplicadorDeMatrix(int p);
int main(){
generadorMultiplicadorDeMatrix(1);
return 0;
}
void generadorMultiplicadorDeMatrix(int p){
float tiempo_inicial= omp_get_wtime();
srand(time(NULL));
int**matrix_A = generateMatrix(N,0); 
int**matrix_B = generateMatrix(N,0); 
int**matrix_Out = generateMatrix(N,1); 
showMatrix(matrix_A , N);  
showMatrix(matrix_B , N);  
matrix_Out = multMatrix(matrix_A , matrix_B , N, p);
showMatrix(matrix_Out , N);
float tiempo_final= omp_get_wtime();
printf(" %f \n",tiempo_final-tiempo_inicial);
}
void showMatrix(int**matrix , int n){
for (int i = 0 ; i < n ; i++){
for(int j = 0 ; j < n ; j++)
printf("%i  ",matrix[i][j]);
printf("\n");
}
printf("\n");
}
int*generateRow(int n, bool is_out){
int*row = (int*)malloc(n*sizeof(int));
if(!is_out){
for(int i = 0 ; i < n ; i++)
row[i] = rand()%10;
}
return row;
}
int**generateMatrix(int n,bool is_out){
int ** matrix= (int**)malloc(n*sizeof(int*));
for(int i = 0 ; i < n ; i++) {
matrix[i]= generateRow(n,is_out);
}
return matrix;
}
int** multMatrix(int**matrix_A , int**matrix_B ,  int n, int p){
omp_set_num_threads(p);
int**matrix_Out = generateMatrix(N,1);
#pragma omp parallel for
for (int a = 0; a < n; a++) {
for (int i = 0; i < n; i++) {
int adder = 0;
#pragma omp parallel for reduction(+:adder) 
for (int j = 0; j < n; j++) {
adder += matrix_A[i][j] * matrix_B[j][a];
}
matrix_Out[i][a] = adder;
}
}
return matrix_Out;
}
