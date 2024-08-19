#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>
#include<time.h>
int randint(int minimo, int maximo,int *sd){  
int num = (rand_r(sd) % (minimo - maximo + 1)) + minimo; 
return num;
}
void rand_delay(int *rseed){
int t = randint(1,25,rseed);
sleep(t);
}
void f(int x){
int T[16] = {1,20,30,5,4,20,7,2,1,2,8,5,4,2,8,2};
sleep(T[x]);
}
int main(int argc,char *argv){
int tid,i,seed = time(NULL);
float thilo,tglobal,start = omp_get_wtime();
puts("--- Sin schedule ---");
#pragma omp parallel private(seed,tid,thilo)
{
tid = omp_get_thread_num();
seed += omp_get_thread_num();
#pragma omp for nowait
for(i=0;i<16;i++){
f(i);
}
thilo = omp_get_wtime()-start;
printf("Hilo %d: %f s \n",tid,thilo);
}
tglobal = omp_get_wtime()-start;
printf("Tiempo global: %f s\n",tglobal);
puts("--- Con schedule ---");
start = omp_get_wtime();
#pragma omp parallel private(seed,tid,thilo)
{
tid = omp_get_thread_num();
seed += omp_get_thread_num();
#pragma omp for nowait schedule(dynamic,1)
for(i=0;i<16;i++){
f(i);
}
thilo = omp_get_wtime()-start;
printf("Hilo %d: %f s \n",tid,thilo);
}
tglobal = omp_get_wtime()-start;
printf("Tiempo global: %f s\n",tglobal);
return 0;
}
