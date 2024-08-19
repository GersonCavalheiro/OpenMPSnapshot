#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#define UPTO 10000000
long int count,      
lastprime;  
void serial_primes(long int n) {
long int i, num, divisor, quotient, remainder;
if (n < 2) return;
count = 1;                         
lastprime = 2;
for (i = 0; i < (n-1)/2; ++i) {    
num = 2*i + 3;
divisor = 1;
do 
{
divisor += 2;                  
quotient  = num / divisor;  
remainder = num % divisor;  
} while (remainder && divisor <= quotient);  
if (remainder || divisor == num) 
{
count++;
lastprime = num;
}
}
}
void openmp_primes(long int n) {
long int i, num, divisor, quotient, remainder;
if (n < 2) return;
count = 1;                         
lastprime = 2;
omp_set_num_threads(4);
#pragma omp parallel private(num, divisor, quotient, remainder) reduction(max: lastprime) reduction(+:count)
{
#pragma omp for schedule(static, 1000)
for (i = 0; i < (n-1)/2; ++i) {    
num = 2*i + 3;
divisor = 1;
do 
{
divisor += 2;                  
quotient  = num / divisor;  
remainder = num % divisor;  
} while (remainder && divisor <= quotient);  
if (remainder || divisor == num) 
{
count++;
lastprime = num;
}
}
}
}
int main()
{
double exectime, exectimepar;
struct timeval start, end;
printf("Serial and parallel prime number calculations:\n\n");
gettimeofday(&start, NULL);
serial_primes(UPTO);        
gettimeofday(&end, NULL);
exectime = (double) (end.tv_usec - start.tv_usec) / 1000000 + (double) (end.tv_sec - start.tv_sec);
printf("[serial] count = %ld, last = %ld, execution time = %lf sec\n", count, lastprime, exectime);
gettimeofday(&start, NULL);
openmp_primes(UPTO);        
gettimeofday(&end, NULL);
exectimepar = (double) (end.tv_usec - start.tv_usec) / 1000000 + (double) (end.tv_sec - start.tv_sec); 
printf("[openmp] count = %ld, last = %ld, execution time = %lf sec\n", count, lastprime, exectimepar);
return 0;
}
