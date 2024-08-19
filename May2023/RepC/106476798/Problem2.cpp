#include <Rcpp.h>
#include <omp.h>
using namespace Rcpp;
NumericVector omp_rollmean(NumericVector x, int k)
{
omp_set_num_threads(4);
int totalNum = x.size() - k + 1;
NumericVector result(totalNum);
#pragma omp parallel
{
int i;
int numThreads = omp_get_num_threads();
int partition = totalNum / numThreads + 1;
int me = omp_get_thread_num();
int start = me * partition;
double sum;	
sum = 0;
for (i = start; i < start + k; i++)
sum += x[i];
result[start] = sum / k;
for (i = start; i < (me + 1) * partition && i <= x.size() - k; i++) {
sum -= x[i];
sum += x[i + k];
result[i + 1] = sum / k;
}
}
return result;
}
