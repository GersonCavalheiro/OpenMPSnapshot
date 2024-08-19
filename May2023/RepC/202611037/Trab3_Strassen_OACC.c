#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
int** alocarMatriz(int Tam);
void soma(double **A, double **B, double **C, int Tam);
void subtracao(double **A, double **B, double **C, int Tam);
void algoritmoDeStrassen(double **A, double **B, double **D, int Tam);
void liberamemoria(double **A, int Tam);
int main (int argc, char *argv[]){
int i, j;
double **A, **B, **C; 
double t_final, t_inicial;
int Tamanho = atoi(argv[1]);
srand(time(NULL));
A = (double**) malloc(sizeof(double*)*Tamanho);
B = (double**) malloc(sizeof(double*)*Tamanho);
C = (double**) malloc(sizeof(double*)*Tamanho);
for(i=0; i<Tamanho; i++){
A[i] = (double*) malloc(sizeof(double)*Tamanho);
B[i] = (double*) malloc(sizeof(double)*Tamanho);
C[i] = (double*) malloc(sizeof(double)*Tamanho);
}
for(i=0; i<Tamanho; i++){
for(j=0; j<Tamanho; j++){
A[i][j] = rand()%Tamanho;
B[i][j] = rand()%Tamanho;
C[i][j] = 0.0;
}
}
printf("_____________Método de Strassen_______________\n\n\n");
t_inicial = omp_get_wtime();
algoritmoDeStrassen(A,B,C,Tamanho); 
t_final = omp_get_wtime();
printf("Tempo de execução: %lf\n", t_final-t_inicial);
for(i=0; i<Tamanho; i++){
free(A[i]);
free(B[i]);
free(C[i]);
}
free(A);
free(B);
free(C);
return 0;
}
void soma(double **A, double **B, double **C, int Tam){ 
int i,j;
for(i=0; i<Tam; i++){
for(j=0; j<Tam; j++){
C[i][j] = A[i][j]+B[i][j]; 
}
}
}
void subtracao(double **A, double **B, double **C, int Tam){ 
int i,j;
for(i=0; i<Tam; i++){
for(j=0; j<Tam; j++){
C[i][j] = A[i][j]-B[i][j]; 
}
}
}
void algoritmoDeStrassen(double **A, double **B, double **D, int Tam){ 
double **A11, **A12, **A21, **A22, **B11, **B12, **B21, **B22, **C11, **C12, **C21, **C22,
**M1, **M2, **M3, **M4, **M5, **M6, **M7, **aux1, **aux2;
int i, j;
int newTam = Tam/2;
if(Tam == 1){ 
D[0][0] = A[0][0] * B[0][0];
return;
}
A11 = (double**) malloc(sizeof(double*)*newTam);
A12 = (double**) malloc(sizeof(double*)*newTam);
A21 = (double**) malloc(sizeof(double*)*newTam);
A22 = (double**) malloc(sizeof(double*)*newTam);
B11 = (double**) malloc(sizeof(double*)*newTam);
B12 = (double**) malloc(sizeof(double*)*newTam);
B21 = (double**) malloc(sizeof(double*)*newTam);
B22 = (double**) malloc(sizeof(double*)*newTam);
C11 = (double**) malloc(sizeof(double*)*newTam);
C12 = (double**) malloc(sizeof(double*)*newTam);
C21 = (double**) malloc(sizeof(double*)*newTam);
C22 = (double**) malloc(sizeof(double*)*newTam);
M1 = (double**) malloc(sizeof(double*)*newTam);
M2 = (double**) malloc(sizeof(double*)*newTam);
M3 = (double**) malloc(sizeof(double*)*newTam);
M4 = (double**) malloc(sizeof(double*)*newTam);
M5 = (double**) malloc(sizeof(double*)*newTam);
M6 = (double**) malloc(sizeof(double*)*newTam);
M7 = (double**) malloc(sizeof(double*)*newTam);
aux1 = (double**) malloc(sizeof(double*)*newTam);
aux2 = (double**) malloc(sizeof(double*)*newTam);
for(i=0; i<newTam; i++){
A11[i] = (double*) malloc(sizeof(double)*newTam);
A12[i] = (double*) malloc(sizeof(double)*newTam);
A21[i] = (double*) malloc(sizeof(double)*newTam);
A22[i] = (double*) malloc(sizeof(double)*newTam);
B11[i] = (double*) malloc(sizeof(double)*newTam);
B12[i] = (double*) malloc(sizeof(double)*newTam);
B21[i] = (double*) malloc(sizeof(double)*newTam);
B22[i] = (double*) malloc(sizeof(double)*newTam);
C11[i] = (double*) malloc(sizeof(double)*newTam);
C12[i] = (double*) malloc(sizeof(double)*newTam);
C21[i] = (double*) malloc(sizeof(double)*newTam);
C22[i] = (double*) malloc(sizeof(double)*newTam);
M1[i] = (double*) malloc(sizeof(double)*newTam);
M2[i] = (double*) malloc(sizeof(double)*newTam);
M3[i] = (double*) malloc(sizeof(double)*newTam);
M4[i] = (double*) malloc(sizeof(double)*newTam);
M5[i] = (double*) malloc(sizeof(double)*newTam);
M6[i] = (double*) malloc(sizeof(double)*newTam);
M7[i] = (double*) malloc(sizeof(double)*newTam);
aux1[i] = (double*) malloc(sizeof(double)*newTam);
aux2[i] = (double*) malloc(sizeof(double)*newTam); 
}
#pragma acc parallel loop collapse(2)
for(i = 0; i<newTam; i++){
for(j = 0; j<newTam; j++){
A11[i][j] = A[i][j];
A12[i][j] = A[i][newTam+j];
A21[i][j] = A[newTam+i][j];
A22[i][j] = A[newTam+i][newTam+j];
B11[i][j] = B[i][j];
B12[i][j] = B[i][newTam+j];
B21[i][j] = B[newTam+i][j];
B22[i][j] = B[newTam+i][newTam+j];
}
}
soma(A11,A22,aux1,newTam);
soma(B11,B22,aux2,newTam);
algoritmoDeStrassen(aux1,aux2,M1,newTam); 
soma(A21,A22,aux1,newTam);
algoritmoDeStrassen(aux1,B11,M2,newTam); 
subtracao(B12,B22,aux1,newTam);
algoritmoDeStrassen(A11,aux1,M3,newTam); 
subtracao(B21,B11,aux1,newTam);
algoritmoDeStrassen(A22,aux1,M4,newTam); 
soma(A11,A12,aux1,newTam);
algoritmoDeStrassen(aux1,B22,M5,newTam); 
subtracao(A21,A11,aux1,newTam);
soma(B11,B12,aux2,newTam);
algoritmoDeStrassen(aux1,aux2,M6,newTam); 
subtracao(A12,A22,aux1,newTam);
soma(B21,B22,aux2,newTam);
algoritmoDeStrassen(aux1,aux2,M7,newTam); 
soma(M1,M4,aux1,newTam);
soma(aux1,M7,aux2,newTam);
subtracao(aux2,M5,C11,newTam); 
soma(M2,M4,C21,newTam); 
soma(M3,M5,C12,newTam); 
soma(M1,M3,aux1,newTam);
soma(aux1,M6,aux2,newTam);
subtracao(aux2,M2,C22,newTam);  
for(i = 0; i<newTam; i++){
for(j = 0; j<newTam; j++){
D[i][j] = C11[i][j];
D[i][newTam+j] = C12[i][j];
D[newTam+i][j] = C21[i][j];
D[newTam+i][newTam+j] = C22[i][j];
}
}
for(i=0; i<newTam; i++){
free(A11[i]);
free(A12[i]);
free(A21[i]);
free(A22[i]);
free(B11[i]);
free(B12[i]);
free(B21[i]);
free(B22[i]);
free(C11[i]);
free(C12[i]);
free(C21[i]);
free(C22[i]);
free(M1[i]);
free(M2[i]);
free(M3[i]);
free(M4[i]);
free(M5[i]);
free(M6[i]);
free(M7[i]);
free(aux1[i]);
free(aux2[i]);
}
free(A11);
free(A12);
free(A21);
free(A22);
free(B11);
free(B12);
free(B21);
free(B22);
free(C11);
free(C12);
free(C21);
free(C22);
free(M1);
free(M2);
free(M3);
free(M4);
free(M5);
free(M6);
free(M7);
free(aux1);
free(aux2);
}