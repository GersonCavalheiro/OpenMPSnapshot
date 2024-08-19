#include <stdio.h>
#include <omp.h>
#include <math.h>
FILE *fptr;
#define NUM_THREADS 6
int main()
{
int N = 100000;
fptr=fopen("Euler_n_02.txt","w");
printf("Numero de pasos:%d \n", N);
fprintf(fptr, "Datos que encuentra el metodo de Euler(variable ind.\t variable dep.\t numero de thread)\n");
double h,t[N],w[N],ab;
int thrd[N];
double w0=0.5,a=0,b=2;
double t1, t2, tiempo;
int i;
omp_set_num_threads(NUM_THREADS);
t1 = omp_get_wtime();
#pragma omp parallel
{ 
h=(b-a)/N;
w[0] = w0;
t[0] = a;
for(i=0;i<N;i++){
thrd[i] = omp_get_thread_num();
t[i]=a+(h*i);
w[i]=w[i-1]+h*(1+t[i-1]*t[i-1]-w[i-1]);
}
}
t2 = omp_get_wtime();
tiempo =t2-t1;
for(i=0;i<N;i++){
fprintf(fptr, "%f\t %lf\t thread: %d\n", t[i], w[i], thrd[i]);
}
printf("Tiempo (%lf) segundos \n", tiempo);
fclose(fptr);
}