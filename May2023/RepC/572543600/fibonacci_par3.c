#include "stdio.h" 
#include "stdlib.h" 
#include "time.h"   
#include "math.h"  
#include <omp.h>
int parallel_fibonacci(int number);
int serial_fibonacci(int number);
int main(int argc, char* argv[]){
int res;
if(argc != 2){
printf("Error: You have to insert the input number for fibonacci!\n");
return 0;
}
printf("The input number for fibonacci is: %s\n", argv[1]);
int maxthreads = omp_get_max_threads();
printf("Max available threads = %d\n", maxthreads);
int nthreads_par;
int nthreads_sin;
double start_time = omp_get_wtime();
#pragma omp parallel num_threads(128)
{
nthreads_par = omp_get_num_threads();
#pragma omp single
{
nthreads_sin = omp_get_num_threads();
res = parallel_fibonacci(atoi(argv[1]));
}
}
double run_time = omp_get_wtime() - start_time;
printf("Total DFTW computation in %f seconds\n",run_time);
printf("Total threads used after parallel declaration = %d\n",nthreads_par);
printf("Total threads used after single declaration = %d\n",nthreads_sin);
printf("The fibonacci of %s is %d\n", argv[1], res);
}
int parallel_fibonacci(int number){
if(number == 0){
return 0;
}else if( number == 1){
return 1;
}
if(number <= 30){
return serial_fibonacci(number);
}
int x, y;
#pragma omp task shared(x)
{
x = parallel_fibonacci(number - 1);
}
#pragma omp task shared(y)
{
y = parallel_fibonacci(number - 2);
}
#pragma omp taskwait
return x + y;
}
int serial_fibonacci(int number){
if(number == 0){
return 0;
}else if( number == 1){
return 1;
}
int x = serial_fibonacci(number - 1);
int y = serial_fibonacci(number - 2);
return x + y;
}