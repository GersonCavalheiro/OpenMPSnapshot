#include <cstdio>
#include <omp.h>

int main() {
int a[30] = {1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1};
int decimal = 0;

#pragma omp parallel for reduction(+: decimal)
for (int i = 0; i < 30; i++) {
if (a[i] == 1) {
decimal += 1 << i;
}
}

printf("%d\n", decimal);
}
