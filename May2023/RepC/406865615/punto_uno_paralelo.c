#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#define COORDINATOR 0
double dwalltime();
void validation(int, double *, double);
int main(int argc, char** argv){
int miID, nrProcesos, N, stripSize, i, j, k;
double *A, *C, *AC,*B, *D, *BD, *R, timetick, totalTime;
register double aux = 0;
double *aa, *ac, *bb, *bd, *rr;
double promedio_parcial_a, promedio_parcial_b;
double *promedio;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &miID);
MPI_Comm_size(MPI_COMM_WORLD, &nrProcesos);
if ((argc != 3) || ((N = atoi(argv[1])) <= 0) ) {
printf("\nUsar: %s size num_threads\n  size: Dimension de la matriz num_threads: Numero de hilos\n", argv[0]);
exit(1);
}
if (N % nrProcesos != 0) {
printf("El tamaño de la matriz debe ser multiplo del numero de procesos.\n");
exit(1);
}
int numThreads = atoi(argv[2]);
omp_set_num_threads(numThreads);
stripSize = N / nrProcesos;
if (miID == COORDINATOR) {
A = (double*) malloc(sizeof(double)*N*N);
AC = (double*) malloc(sizeof(double)*N*N);
B = (double*) malloc(sizeof(double)*N*N);
BD = (double*) malloc(sizeof(double)*N*N);
}
else  {
A = (double*) malloc(sizeof(double)*N*stripSize);
AC = (double*) malloc(sizeof(double)*N*stripSize);
B = (double*) malloc(sizeof(double)*N*stripSize);
BD = (double*) malloc(sizeof(double)*N*stripSize);
}
C = (double*) malloc(sizeof(double)*N*N);
D = (double*) malloc(sizeof(double)*N*N);
R = (double*) malloc(sizeof(double)*N*N);
aa = (double*) malloc(sizeof(double)*N*N);
ac = (double*) malloc(sizeof(double)*N*N);
bb = (double*) malloc(sizeof(double)*N*N);
bd = (double*) malloc(sizeof(double)*N*N);
rr = (double*) malloc(sizeof(double)*N*N);		
promedio = (double*) malloc(sizeof(double)*2);
if (miID == COORDINATOR) {
for (i=0; i<N ; i++)
for (j=0; j<N ; j++){
A[i*N+j] = 1;
C[i*N+j] = 1;
B[i*N+j] = 1;
D[i*N+j] = 1;
}	
}
promedio_parcial_a = 0;
promedio_parcial_b = 0;
timetick = dwalltime();
MPI_Scatter(A, N*stripSize, MPI_DOUBLE, aa, N*stripSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Scatter(B, N*stripSize, MPI_DOUBLE, bb, N*stripSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(C, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(D, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#pragma omp parallel
{
#pragma omp for nowait private(i, j, k, aux) reduction(+:promedio_parcial_a)
for (i = 0; i < stripSize; i++) {
for (j = 0; j < N; j++) {
aux  = 0;
promedio_parcial_a += aa[i * N + j];
for (k = 0; k < N; k++) {
aux += (aa[i*N+k]*C[j*N+k]);
}
ac[i*N+j]=aux;
}
}
#pragma omp for private(i, j, k, aux) reduction(+:promedio_parcial_b) 
for (i = 0; i < stripSize; i++) {
for (j = 0; j < N; j++) {
aux  = 0;
promedio_parcial_b += bb[i * N + j];
for (k = 0; k < N; k++) {
aux += (bb[i*N+k]*D[j*N+k]);
}
bd[i*N+j]=aux;
}
}
}
MPI_Allreduce(&promedio_parcial_a, &promedio[0], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(&promedio_parcial_b, &promedio[1], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
promedio[0] /= (N*N);
promedio[1] /= (N*N);
#pragma omp parallel for private(i,j) firstprivate(stripSize, promedio, ac, bd)   
for (i = 0; i < stripSize; i++)
{
for (j = 0; j < N; j++)
{
rr[i * N + j] = (promedio[1] * ac[i * N + j]) + (promedio[0] * bd[i * N + j]);
}
}
MPI_Gather(rr, N*stripSize, MPI_DOUBLE, R, N*stripSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
totalTime = dwalltime() - timetick;
MPI_Finalize();
if (miID == COORDINATOR) {
validation(N, R, totalTime);
}
free(A);
free(C);
free(AC);
free(aa);
free(ac);
free(B);
free(BD);
free(D);
free(bb);
free(bd);
free(rr);
free(promedio);
return(0);
}
double dwalltime()
{
double sec;
struct timeval tv;
gettimeofday(&tv,NULL);
sec = tv.tv_sec + tv.tv_usec/1000000.0;
return sec;
}
void validation(int N, double *c, double totalTime)
{
int i, j, k, check=1;
for(i=0;i<N;i++)
for(j=0;j<N;j++)
check=check&&(c[i*N+j]==N+N);
if(check){
printf("Multiplicación de matrices resultado correcto\n");
}else{
printf("Multiplicación de matrices resultado erroneo\n");
}
printf("****************\nMultiplicación de matrices (N=%d)\nTiempo total: %lf\n", N, totalTime);
}
