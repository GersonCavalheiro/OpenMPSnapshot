#include <stdio.h>
#include <time.h>
#include <sys/time.h>     
#include <omp.h>          
#define N_TERMS 2000000000LL
double pi(long long N) {
double sum = 0.0;
int n_threads = omp_get_num_threads();
double local_sum  = 0.0;
#pragma omp parallel shared(sum)
{
int thread_num = omp_get_thread_num();
if(thread_num==0){
printf("Computing Pi using %d threads\n",omp_get_num_threads());
}
#pragma omp for nowait
for (long long  k = 0; k < N; ++k) {
local_sum+= (k % 2 ? -1.0 : 1.0) / (2 * k + 1);
}
#pragma omp atomic
sum += local_sum;
}
return 4 * sum;
}
int main()
{
hrtime_t start, end;
start = gethrvtime();
double pi_estimate = pi(N_TERMS);
end = gethrvtime();
printf("Time: %lld nsec\n",end - start);
printf("Leibniz Pi estimate with %lld terms: \n %.9lf\n"
, N_TERMS, pi_estimate);
return 0;
}
