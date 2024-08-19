#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
int main(int argc, char** argv){
int i, j, f, c;
double t1, t2, total;
srand(time(NULL));
if (argc<2){
printf("Falta tamaño de matriz y vector\n");
exit(-1);
}
unsigned int N = atoi(argv[1]); 
double *v1, *v2, **M;
v1 = (double*) malloc(N*sizeof(double));
v2 = (double*) malloc(N*sizeof(double)); 
M = (double**) malloc(N*sizeof(double *));
if ( (v1==NULL) || (v2==NULL) || (M==NULL) ){
printf("Error en la reserva de espacio para los vectores\n");
exit(-2);
}
for (i=0; i<N; i++){
M[i] = (double*) malloc(N*sizeof(double));
if ( M[i]==NULL ){
printf("Error en la reserva de espacio para los vectores\n");
exit(-2);
}
}
#pragma omp parallel for
for (i=0; i<N; i++)
v1[i]=rand()%(1-10 + 1) + 1;
#pragma omp parallel for
for (f=0; f<N; f++)
for (c=0; c<N; c++)
M[f][c] = rand()%(1-10 + 1) + 1;
double suma;
t1 = omp_get_wtime();
for (f=0; f<N; f++)
{
suma=0;
#pragma omp parallel for reduction(+:suma)
for (c=0; c<N; c++)
suma += M[f][c] * v1[c];
v2[f] += suma;		
}
t2 = omp_get_wtime();
total = t2 - t1;
printf("Tiempo(seg.):%11.9f\t / Tamaño:%u\t/ V2[0]=%8.6f V2[%d]=%8.6f\n", total,N,v2[0],N-1,v2[N-1]);
if (N<15)
{
printf("\nv2=[");
for (i=0; i<N; i++)
printf("%.0lf ",v2[i]);
printf("]\n");
}
free(v1); 
free(v2); 
for (i=0; i<N; i++)
free(M[i]);
free(M);
return 0;
}
