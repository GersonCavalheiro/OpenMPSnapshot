#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
int sieveOfEratosthenes(int n)
{
int primes = 0; 
bool *prime = (bool*) malloc((n+1)*sizeof(bool));
int sqrt_n = sqrt(n);
memset(prime, true,(n+1)*sizeof(bool));
int i, p;
#pragma omp parallel for schedule (dynamic,100)
for (p=2; p <= sqrt_n; p++)
{
if (prime[p] == true)
{
#pragma omp parallel for 
for(i=p*2; i<=n; i += p)
prime[i] = false;
}
}
#pragma omp parallel for reduction(+:primes)   
for (int p=2; p<=n; p++)
if (prime[p])
primes++;
return(primes);
}
int main()
{
int n = 100000000;
printf("%d\n",sieveOfEratosthenes(n));
return 0;
} 
