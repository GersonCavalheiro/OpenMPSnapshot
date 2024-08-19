#include <cstdio>
#include <omp.h>

int main() {
int a[] = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1};

int s = 0;
#pragma omp parallel for reduction(+: s)
for (int i = 0; i < 10; i++) {
s += a[i];
}
double avg1 = 1.0 * s / 10;
printf("a average is %f\n", avg1);

int b[] = {0, 2, 6, 8, 10, 12, 9, 5, 1, 0};

s = 0;
#pragma omp parallel for reduction(+: s)
for (int i = 0; i < 10; i++) {
s += b[i];
}
double avg2 = 1.0 * s / 10;
printf("b average is %f\n", avg2);

printf("avg of a ");
if (avg1 > avg2) {
printf(">");
} else if (avg1 == avg2) {
printf("=");
} else {
printf("<");
}
printf(" avg of b\n");
}
