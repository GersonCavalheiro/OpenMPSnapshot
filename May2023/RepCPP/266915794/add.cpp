#include <Rcpp.h>
#include <omp.h>
using namespace Rcpp;


int add_sugar(NumericVector x, NumericVector y, NumericVector & ans) {

ans = exp(x + y);

return 1;

}

int add_for(NumericVector x, NumericVector y, NumericVector & ans) {

for (int i = 0; i < (int) x.size(); ++i)
ans[i] = exp(x[i] + y[i]);

return 1;

}

int add_simd(NumericVector x, NumericVector y, NumericVector & ans,int ncores = 4) {

omp_set_num_threads(ncores);

#pragma omp simd
for (int i = 0; i < (int) x.size(); ++i)
ans[i] = exp(x[i] + y[i]);

return 1;

}

int add_omp(NumericVector x, NumericVector y, NumericVector & ans, int ncores = 4) {

omp_set_num_threads(ncores);

#pragma omp parallel for shared(ans) 
for (int i = 0; i < (int) x.size(); ++i)
ans[i] = exp(x[i] + y[i]);

return 1;

}

int add_omp_simd(NumericVector x, NumericVector y, NumericVector & ans, int ncores = 4) {

omp_set_num_threads(ncores);

#pragma omp distribute parallel for simd
for (int i = 0; i < (int) x.size(); ++i)
ans[i] = exp(x[i] + y[i]);

return 1;

}


