#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
int fibonacci(int n){
if(n == 0){
return 0;
} 
else if(n == 1){
return 1;
} 
else{
return (fibonacci(n-1) + fibonacci(n-2));
}
}
int main(int argc,char *argv){
int tid,i;
float thilo,tglobal,start = omp_get_wtime();
puts("--- Sin schedule ---");
#pragma omp parallel private(tid,thilo)
{
tid = omp_get_thread_num();
#pragma omp for nowait
for(i=0;i<42;i++){
fibonacci(i);
}
thilo = omp_get_wtime()-start;
printf("Hilo %d: %f s \n",tid,thilo);
}
tglobal = omp_get_wtime()-start;
printf("Tiempo global: %f s\n",tglobal);
puts("--- Con schedule ---");
start = omp_get_wtime();
#pragma omp parallel private(tid,thilo)
{
tid = omp_get_thread_num();
#pragma omp for nowait schedule(static,1)
for(i=0;i<42;i++){
fibonacci(i);
}
thilo = omp_get_wtime()-start;
printf("Hilo %d: %f s \n",tid,thilo);
}
tglobal = omp_get_wtime()-start;
printf("Tiempo global: %f s\n",tglobal);
return 0;
}
