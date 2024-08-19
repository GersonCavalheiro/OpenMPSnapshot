#include <iostream>
#include <omp.h>

constexpr int MAX = 1000;
constexpr int SIZE = 12;

using namespace std;
double A[MAX], B[MAX];
long long C;

double getResult(double* time) {
double start;

omp_set_num_threads(SIZE);
start = omp_get_wtime();

#pragma omp parallel for reduction(+:C)
for (int i = 0; i < MAX; i++) {
C += A[i] * B[i] * 1ll;
}

*time = omp_get_wtime() - start;
return C;
}

int main()
{
double t;
for (int i = 0; i < MAX; i++) {
A[i] = (i + 1) / 10000.;
B[i] = (i + 1) / 10000.;
}
double result = getResult(&t);
cout << "Result: " << result << '\n';
cout << "Execute time(" << SIZE << ") " << t << " seconds\n";

return 0;
}

