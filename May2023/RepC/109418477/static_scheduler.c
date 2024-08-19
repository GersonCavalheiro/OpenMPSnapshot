#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
int main(int argc, char* argv[]){
const int N= 28;
int nthreads, threadid, i;
double a[N], b[N], result[N];
for(i=0;i<N;i++)
{
a[i] = 1.0*i;
b[i] = 2.0*i;
}
int chunk = 7;
#pragma omp parallel private(threadid)
{
threadid = omp_get_thread_num();
#pragma omp for schedule(dynamic, chunk)
for(i=0;i<N;i++){
result[i] = a[i] + b[i];
printf("Thread id: %d working on index %d\n", threadid, i);
}
}
printf(" TEST result[19] = %g\n",result[19]);
return 0;
}
