#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include"leerBin.h"
double maximo_local(double *A,int i, int j,int n){
int k,l;
double max = -DBL_MAX;
for(k=-1;k<2;k++)
for(l=-1;l<2;l++)
if(A[(i+k)*n+j+l]>max)
max = A[(i+k)*n+j+l];
return max;
}
int main(int argc,char *argv[]){
int i,j,m,n;
double *A;
double start,end;
leerMatriz("Matriz.dat",&A,&m,&n);
double *B1 = (double*) calloc(m*n,sizeof(double));
double *B2 = (double*) calloc(m*n,sizeof(double));  
start = omp_get_wtime();
for(i=1;i<m-1;i++)
for(j=1;j<n-1;j++)
B1[i*n+j] = maximo_local(A,i,j,n);
end = omp_get_wtime();
printf("T. de ejecución serial: %f s\n",end - start);
start = omp_get_wtime();
#pragma omp parallel for collapse(2) 
for(i=1;i<m-1;i++)
for(j=1;j<n-1;j++)
B2[i*n+j] = maximo_local(A,i,j,n);
end = omp_get_wtime();
printf("T. de ejecución paralelo: %f s\n",end - start);
return 0;
}
