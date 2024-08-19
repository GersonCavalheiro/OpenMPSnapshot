#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main (int argc, char *argv[]){
FILE* f = fopen("scale.txt", "a");
long long int Ntests = atoll(argv[1]);
double tic = omp_get_wtime();
long long int Ninside = 0;
long long n;
double estpi = 0;
struct drand48_data randBuffer;  
srand48_r(omp_get_thread_num(),&randBuffer);
#pragma omp parallel private(n,randBuffer) reduction(+:Ninside)
for(n=0;n<Ntests;++n){
double x;
double y;
drand48_r(&randBuffer, &x);
drand48_r(&randBuffer, &y);
if(sqrt((x*x) + (y*y)) < 1){ 
Ninside++;    
}
}
estpi = (double)(4*Ninside) / (double)(Ntests * 8);
double toc = omp_get_wtime();
double elapsedTime = toc - tic;
fprintf(f, "estPi = %lf\n", estpi);
fprintf(f, "dt = %f\n", elapsedTime);
fclose(f);
}
