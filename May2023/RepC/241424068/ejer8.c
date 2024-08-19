#include <stdlib.h>	
#include <stdio.h>	
#include <time.h>	
#include <omp.h>
#define VECTOR_DYNAMIC	
#ifdef VECTOR_GLOBAL
#define MAX 33554432	
double v1[MAX], v2[MAX], v3[MAX]; 
#endif
int main(int argc, char** argv){ 
int i; 
double t1, tf;
if (argc<2){	
printf("Faltan nº componentes del vector\n");
exit(-1);
}
unsigned int N = atoi(argv[1]);	
printf("Tamaño Vectores:%u (%u B)\n",N, sizeof(unsigned int)); 
#ifdef VECTOR_LOCAL
double v1[N], v2[N], v3[N];   
#endif
#ifdef VECTOR_GLOBAL
if (N>MAX) N=MAX;
#endif
#ifdef VECTOR_DYNAMIC
double *v1, *v2, *v3;
v1 = (double*) malloc(N*sizeof(double));
v2 = (double*) malloc(N*sizeof(double));
v3 = (double*) malloc(N*sizeof(double));
if ((v1 == NULL) || (v2 == NULL) || (v2 == NULL)) {	
printf("No hay suficiente espacio para los vectores \n");
exit(-2);
}
#endif
#pragma omp parallel sections
{
#pragma omp section
for(i=0; i<N/2; i++){ 
v1[i] = i; 
v2[i] = i*0.5+N;
}
#pragma omp section
for(i=N/2; i<N; i++){
v1[i] = i; 
v2[i] = i*0.5+N;
}
}
t1 = omp_get_wtime();
#pragma omp parallel sections
{
#pragma omp section
for(i=0; i<N/2; i++) 
v3[i] = v1[i] + v2[i];
#pragma omp section
for(i=N/2; i<N; i++) 
v3[i] = v1[i] + v2[i];
}
tf = omp_get_wtime() - t1;
if (N<10) {
printf("Tiempo:%11.9f\t / Tamaño Vectores:%u\n",tf,N); 
for(i=0; i<N; i++) 
printf("/ V1[%d]+V2[%d]=V3[%d](%8.6f+%8.6f=%8.6f) /\n",
i,i,i,v1[i],v2[i],v3[i]); 
}
else
printf("Tiempo:%11.9f\t / Tamaño Vectores:%u\t/ V1[0]+V2[0]=V3[0](%8.6f+%8.6f=%8.6f) / / V1[%d]+V2[%d]=V3[%d](%8.6f+%8.6f=%8.6f) /\n",
tf,N,v1[0],v2[0],v3[0],N-1,N-1,N-1,v1[N-1],v2[N-1],v3[N-1]); 
#ifdef VECTOR_DYNAMIC
free(v1); 
free(v2); 
free(v3); 
#endif
return 0; 
}