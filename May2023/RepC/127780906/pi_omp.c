#include <stdio.h>
#include <time.h>
#include <sys/time.h>     
#include <omp.h>          
#define N_TERMS 2000000000LL
double pi(long long N) {
double sum = 0.0;
#pragma omp parallel 
{
#pragma omp for reduction(+:sum)
for (long long  k = 0; k < N; ++k) {
sum += (k % 2 ? -1.0 : 1.0) / (2 * k + 1);
}
}
return 4 * sum;
}
int main()
{
#if defined(_OPENMP)
omp_set_num_threads(96);
#endif
hrtime_t start, end;
start = gethrvtime();
double pi_estimate = pi(N_TERMS);
end = gethrvtime();
printf("Time: %lld nsec\n",end - start);
printf("Leibniz Pi estimate with %lld terms: \n %.9lf\n"
, N_TERMS, pi_estimate);
return 0;
}
