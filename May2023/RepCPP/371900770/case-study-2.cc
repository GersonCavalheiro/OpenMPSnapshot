void case_study_2(int *__restrict__ a, int *__restrict__ b, int *__restrict__ d,
int N) {
const int s = 10;
#pragma omp simd
for (int i = 0; i < N; i++) {
int j = d[i];
a[j] += s * b[i];
}
}

