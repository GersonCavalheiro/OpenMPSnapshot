#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
typedef struct Range Range;
struct Range{
int minval;
int maxval;
};
bool isPrime(int number)
{
for(int i = 2; i*i <= number; i++)
{
if(number%i == 0) 
{
return false;
}
}
return true;
}
int prime_range(int lowerlimit, int upperlimit){
int total = 0;
for(int i = lowerlimit; i<upperlimit; i++)
{
total=total+isPrime(i);
}
return total;
}
void main(){
int minval, maxval;
int n_threads = 8;
minval=2;
maxval=10000000;
int total_primes = 0;
Range tasks[10000];
for(int i = 0; i<10000; i++){
tasks[i].minval = minval + i*1000;
if(minval + i * 1000 + 1000 <= maxval)
{
tasks[i].maxval = minval + i*1000 + 1000;
}
else 
{
tasks[i].maxval = maxval;
}
}
double start_time = clock();
#pragma omp parallel shared(total_primes)
{
#pragma omp for
for(int i = 0; i<10000; i++){
#pragma omp atomic update
total_primes += prime_range(tasks[i].minval, tasks[i].maxval);
}
}
double end_time = clock();
double timeelapsed= ((end_time - start_time) / CLOCKS_PER_SEC) / n_threads;
printf(" Number of primes in range 1 %d are :\n ",maxval);
printf("%d \n", total_primes);
printf(" Time Elapsed: %.5f secs \n", timeelapsed);
}