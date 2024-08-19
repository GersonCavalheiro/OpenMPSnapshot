#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <mkl.h>
#include <vector>
#include <algorithm>

void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind);

void filter_ref(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {
float sum;
for(long i = 0; i < n; i++){
sum = 0.0f;
for(long j = 0; j < m; j++) {
sum+=data[i*m+j];
} 
if(sum > threshold) 
result_row_ind.push_back(i);
}
std::sort(result_row_ind.begin(),result_row_ind.end());
}

int main(int argc, char** argv) {
float threshold = 0.5;
if(argc < 2) {
threshold = 0.5;
} else {
threshold = atof(argv[1]);
} 


const long n = 1<<15; 
const long m = 1<<18; 

float *data = (float *) malloc((long)sizeof(float)*n*m);
long random_seed = (long)(omp_get_wtime()*1000.0) % 1000L;
VSLStreamStatePtr rnStream;
vslNewStream( &rnStream, VSL_BRNG_MT19937, random_seed);

#pragma omp parallel for 
for(long i =0; i < n; i++)
vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, m, &data[m*i], -1.0, 1.0);

std::vector<long> ref_result_row_ind; 
filter_ref(n, m, data, threshold, ref_result_row_ind);

std::vector<long> result_row_ind; 
const double t0 = omp_get_wtime();
filter(n, m, data, threshold, result_row_ind);
const double t1 = omp_get_wtime();

if(ref_result_row_ind.size() != result_row_ind.size()) {
printf("Error: The reference and result vectors have different sizes: %ld %ld",ref_result_row_ind.size(), result_row_ind.size());
} else {
bool passed = true;
for(long i = 0; i < ref_result_row_ind.size(); i++) {
passed &= (ref_result_row_ind[i] == result_row_ind[i]);
} 
if(passed) {
printf("Time: %f\n", t1-t0);
} else  {
printf("Error: The reference and result vectors did not match");
}
}
}