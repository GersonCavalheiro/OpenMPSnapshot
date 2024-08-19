#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int isprime(int n) {
int i, sq;
sq = (int) sqrt(n);
for (i=3; i<=sq; i+=2)
if ((n%i) == 0) return 0;
return 1;
}
int main(int argc, char *argv[])
{
int i, p=0, max;
max=100000000;   
printf("Numbers to be scanned = %d\n",max);
for (i=1; i<=max; i+=2) {
if (isprime(i)) p++;
}
printf("Total primes = %d\n", p);
}
