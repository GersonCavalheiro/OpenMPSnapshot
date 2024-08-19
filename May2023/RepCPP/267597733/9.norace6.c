#include <omp.h>
#define N 40

int main() {
int F[N], x = 1, y = 1;
#pragma omp parallel
{
#pragma omp single firstprivate(x, y)
for (int i = 3; i < N; i++) {
F[i] = x + y;
y = x;
x = F[i];
}
}
return 0;
}
