#include <iostream>
#include <omp.h>
using namespace std;

int main() {
int m = 4, n = 16384;
int i, j;
double a[m][n];
double s[m];

#pragma omp parallel for schedule(static, 1)
for(i = 0; i < m; ++i) {
s[i] = 1.0;
}

#pragma omp parallel for private(i,j), shared(s,a), schedule(static, 1), collapse(2)
for(i = 0; i < m; i++) {
for(j = 0; j < n; j++){
s[i] = s[i] + a[i][j];
}
}
printf("%f", s[1]);
}