#include <omp.h>
#include<stdio.h>
#include<stdlib.h>
#include"leerBin.h"
void intercambiar(double *a, double *b){ 
double t = *a; 
*a = *b; 
*b = t; 
} 
int bubbleSort_Serial(double v[],int N){
int i,j;
for (i = 0; i < N-1; i++) 
for (j = 0; j < N-i-1; j++) 
if (v[j] > v[j+1]){
intercambiar(&v[j+1],&v[j]);                   
}
}
int bubbleSort_Paralelo(double v[],int N){
int i=0, j=0; 
int inicio;
for(i=0;i<N-1;i++){
inicio = i % 2; 
#pragma omp parallel for default(none),shared(v,inicio,N)
for(j=inicio;j<N-1;j+=2){
if(v[j] > v[j+1]){
intercambiar(&v[j],&v[j+1]);
}
}
}      
}
int main(int argc,char *argv[]){
int i,N;
double *v,*u;
double start,end,T,Tsum;
leerVector("Vector.dat",&v,&N);
leerVector("Vector.dat",&u,&N);
start = omp_get_wtime();
bubbleSort_Serial(v,N);
end = omp_get_wtime();
printf("T. de ejecución serial: %f s\n",end-start);
start = omp_get_wtime();
bubbleSort_Paralelo(u,N);
end = omp_get_wtime();
printf("T. de ejecución paralelo: %f s\n",end - start);
return 0;
}
