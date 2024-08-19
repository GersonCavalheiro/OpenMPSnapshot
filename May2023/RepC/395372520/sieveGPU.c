#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
int sieveOfEratosthenes(int n)
{
int primes = 0; 
bool *prime = (bool*) malloc((n+1)*sizeof(bool));
int sqrt_n = sqrt(n);
memset(prime, true,(n+1)*sizeof(bool));
int i, p;
#pragma omp target map(tofrom:prime[1:n])
#pragma omp teams distribute parallel for simd schedule (dynamic,100)
for (p=2; p <= sqrt_n; p++)
{
if (prime[p] == true)
{
for(i=p*2; i<=n; i += p)
prime[i] = false;
}
}
#pragma omp target map(from:prime[1:n]) map(to:primes)
#pragma omp teams distribute parallel for reduction(+:primes) schedule(guided)
for (int p=2; p<=n; p++)
if (prime[p])
primes++;
return(primes);
}
int main()
{
int n = 100000000;
double start; 
double end; 
start = omp_get_wtime(); 
printf("%d\n",sieveOfEratosthenes(n));
end = omp_get_wtime(); 
printf("Total time: %f seconds\n", end - start);
return 0;
} 
