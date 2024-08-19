#include <stdio.h>
#include <omp.h>
#include <limits.h>
int main()
{
long long N= 10000, sum= -1, verify= 0;
int m= INT_MAX, n= INT_MIN;
verify= (N*(N+1)*(2*N+1))/6;
printf("%d %d\n",m,n);
#pragma omp parallel reduction(+:sum)
{
#pragma omp for
for(long long i= 1; i <= N; i++)
sum+= i*i;
}
printf("Calculated sum= %lld, Verified with %lld\n",sum,verify);
}
