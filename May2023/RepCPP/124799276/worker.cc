#include <vector>
#include <algorithm>
#include <cstdio>

void append_vec(std::vector<long> &v1, std::vector<long> &v2) {
v1.insert(v1.end(),v2.begin(),v2.end());
}


void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {

std::vector<long> shared_vec;  
#pragma omp parallel 
{
std::vector<long> a_vec;
#pragma omp for 
for (long i = 0; i < n; i++) {
float sum = 0.0f;

for (long j = 0; j < m; j++) {
sum += data[i*m + j];
} 

if (sum > threshold) { 
a_vec.push_back(i);
}
}
#pragma omp critical
{
append_vec(result_row_ind, a_vec);
}
}

std::sort(result_row_ind.begin(),
result_row_ind.end());
}
