#include <stdio.h>
#include <omp.h>
int main()
{
long long N= 10000, sum= 0, verify= 0;
verify= (N*(N+1)*(2*N+1))/6;
#pragma omp parallel
{
long long psum= 0;
#pragma omp for
for(long long i= 1; i <= N; i++)
psum+= i*i;
sum+= psum;
}
printf("Calculated sum= %lld, Verified with %lld\n",sum,verify);
}
