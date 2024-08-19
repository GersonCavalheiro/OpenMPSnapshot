#include <stdlib.h> 
#include <stdio.h> 
#include <time.h> 
#include <omp.h> 
#define VECTOR_GLOBAL
#ifdef VECTOR_GLOBAL
#define MAX 33554432 
double v1[MAX], v2[MAX], v3[MAX];
#endif
int main(int argc, char** argv){
int i;
struct timespec cgt1,cgt2; double ncgt; 
if (argc<2){
printf("Faltan n componentes del vector\n");
exit(-1);
}
unsigned int N = atoi(argv[1]); 
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
if ( (v1==NULL) || (v2==NULL) || (v3==NULL) ){
printf("Error en la reserva de espacio para los vectores\n");
exit(-2);
}
#endif
#pragma omp parallel for 
for(i=0; i<N; i++){
v1[i] = N*0.1+i*0.1;
v2[i] = N*0.1-i*0.1; 
}
double start = omp_get_wtime();
#pragma omp parallel for
for(i=0; i<N; i++)
v3[i] = v1[i] + v2[i];
double end = omp_get_wtime();
double diff = end - start;
#ifdef PRINTF_ALL
printf("Tiempo(seg.):%11.9f\t / Tamao Vectores:%u\n",diff,N);
for(i=0; i<N; i++)
printf("/ V1[%d]+V2[%d]=V3[%d](%8.6f+%8.6f=%8.6f) /\n",
i,i,i,v1[i],v2[i],v3[i]);
#else
printf("%f\n",diff);
#endif
#ifdef VECTOR_DYNAMIC
free(v1); 
free(v2); 
free(v3); 
#endif
return 0;
}
