#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define B 10000000000L;
static int is_prime(long n)
{
if(n == 0) 
return 0;
else if(n == 1) 
return 0;
else if(n == 2) 
return 1;
int temp = (int)(sqrt((double)n));
int i;
for(i=2;i<=temp;i++)
if(n%i==0)
return 0;
return 1;
}
int main() 
{
long n  = 100000000;
int numprimes = 0;
struct timespec start, end;
clock_gettime(CLOCK_REALTIME,&start);
#pragma omp parallel forschedule(dynamic, 1) reduction(+:numprimes) 
for(long i = 1; i <= n; i++) 
{
if(is_prime(i) == 1)
numprimes ++;
}
clock_gettime(CLOCK_REALTIME,&end);
double accum=(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)/B;
printf("time = %f\n",accum);
printf("Number of prime within %d: %d\n", n, numprimes);
return 0;
}
