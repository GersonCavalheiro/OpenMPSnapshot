#include <stdlib.h> 
#include <stdio.h> 
#include <time.h> 
#include <omp.h> 
int main(int argc, char** argv){ 
int i, j, f, c, intkind, chunk; 
double t1, t2, total; 
srand(time(NULL));
omp_sched_t kind;
if (argc<4){
printf("Formato: programa tamaño_matriz sched_var chunk\n"); 
exit(-1); 
} 
unsigned int N = atoi(argv[1]); intkind=atoi(argv[2]); chunk=atoi(argv[3]); 
double *v1, *v2, **M; 
v1 = (double*) malloc(N*sizeof(double));
v2 = (double*) malloc(N*sizeof(double)); 
M = (double**) malloc(N*sizeof(double *)); 
if ( (v1==NULL) || (v2==NULL) || (M==NULL) ){ 
printf("Error en la reserva de espacio para los vectores\n"); 
exit(-2); 
}
switch(intkind){ 
case 1: kind = omp_sched_static; 
break; 
case 2: kind = omp_sched_dynamic; 
break; 
case 3: kind = omp_sched_guided; 
break; 
case 4: kind = omp_sched_auto; 
break; 
} 
omp_set_schedule(kind,chunk);
for (i=0; i<N; i++){ 
M[i] = (double*) malloc(N*sizeof(double)); 
if ( M[i]==NULL ){ 
printf("Error en la reserva de espacio para los vectores\n"); 
exit(-2); 
} 
} 
#pragma omp parallel for schedule(runtime)
for (i=0; i<N; i++) 
{	 
v1[i]=i; 
} 
#pragma omp parallel for schedule(runtime)
for (f=0; f<N; f++) 
{ 
for (c=0; c<f; c++){
M[f][c] = 0;
}
for (c=f; c<N; c++) 
{ 
M[f][c] = rand()%(1-10 + 1) + 1; 
}
}	 
t1 = omp_get_wtime(); 
#pragma omp parallel for schedule(runtime)
for (f=0; f<N; f++) 
for (c=f; c<N; c++) 
v2[f] += M[f][c] * v1[c]; 	
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
