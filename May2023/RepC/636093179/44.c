#include <stdio.h>
#include <omp.h>
#include <unistd.h>
int main()
{
long long N= 10000, sum= 0, verifiedSum= 0, max= -1, verifiedMax= 0;
verifiedSum= (N*(N+1)*(2*N+1))/6;
verifiedMax= N*N;
#pragma omp parallel reduction(+:verifiedSum) reduction(max:verifiedMax)
{
#pragma omp for
for(long long i= 1; i <= N; i++){
sum+= i*i;
if(i*i > max)
max= i*i;
}
}
printf("Calculated sum= %lld, Verified with %lld\n",sum,verifiedSum);
printf("Calculated max= %lld, Verified with %lld\n",max,verifiedMax);
}
