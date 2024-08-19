#include <vector>
#include <algorithm>

void append_vec(std::vector<long> &v1, std::vector<long> &v2) {
v1.insert(v1.end(),v2.begin(),v2.end());
}


void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {

#pragma omp parallel for
for (long i = 0; i < n; i++){
float sum = 0.0f;
for (long j = 0; j < m; j++) {
sum += data[i*m + j];
} 

#pragma omp critical
{
if (sum > threshold) 
result_row_ind.push_back(i);
}
}

std::sort(result_row_ind.begin(),
result_row_ind.end());
}