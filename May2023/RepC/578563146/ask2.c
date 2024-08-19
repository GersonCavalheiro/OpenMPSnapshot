#include<stdio.h>
#include<omp.h>
#include<time.h>
#include<string.h>
#include<stdlib.h>
#define N 100000000
int main(int argc, char *argv[]){
double sum = 0, part_sum = 0, dx, xi;
int i, chunk, num_thr;
double cpuStart, cpuEnd;
char *scheduling;
printf("max: %d\n", omp_get_max_threads());
if(argc == 4) {
chunk = atoi(argv[1]);
scheduling = argv[2];
num_thr = atoi(argv[3]);
}
else {
printf("Wrong arguments...\n");
return(1);
}
dx = 1/(double)N;
cpuStart = omp_get_wtime();
sum += 4;
sum += 4/(1+((double)N*(double)N));
if(!strcmp(scheduling,"static")) {
#pragma omp parallel
{
#pragma omp for schedule(static,chunk) reduction(+:part_sum)
for(i = 1; i < N; i++) {
xi = (double)i*dx;
if((i%2) == 0) {
part_sum += 2*(4/(1+(xi*xi)));
}
else{
part_sum += 4*(4/(1+(xi*xi)));
}
}
}
}
else {
#pragma omp parallel
{
#pragma omp for schedule(dynamic,chunk) reduction(+:part_sum)
for(i = 1; i < N; i++) {
xi = (double)i*dx;
if((i%2) == 0) {
part_sum += 2*(4/(1+(xi*xi)));
}
else{
part_sum += 4*(4/(1+(xi*xi)));
}
}
}	
}
part_sum *= dx/3;
cpuEnd = omp_get_wtime();
double cpuElapsedTime = cpuEnd - cpuStart ;
printf("pi is approximately %.8lf,computation time=%lf,number of threads=%d,FLOPS=%lf,chunk=%d,scheduling=%s\n", part_sum, cpuElapsedTime, num_thr, (double)(6*N)/cpuElapsedTime,chunk,scheduling);
return(0);
}
