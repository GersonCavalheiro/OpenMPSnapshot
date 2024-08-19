#include <stdio.h>
#include <omp.h>
int main()
{
long long N= 10000, sum= 0, verify= 0;
verify= (N*(N+1)*(2*N+1))/6;
#pragma omp parallel for
for(long long i= 1; i <= N; i++)
sum+= i*i;
printf("Calculated sum= %lld, Verified with %lld\n",sum,verify);
}
