#include <omp.h>
#include <stdio.h>

int main() {

const int ARRAY_SIZE = 10;

int a[ARRAY_SIZE] = {1, 2, 3, 48, 587, 6, 71, 8, 19, 10};
int b[ARRAY_SIZE] = {11, 12, 3, 14, 15, 758, 17, 18, 65, 26};

float avg_a = 0, avg_b = 0;

#pragma omp parallel for reduction(+: avg_a) reduction(+: avg_b)
for (int i = 0; i < ARRAY_SIZE; i++) {
avg_a += a[i];
avg_b += b[i];
}

#pragma omp sections
{
#pragma omp section
{
avg_a /= ARRAY_SIZE;
}
#pragma omp section
{
avg_b /= ARRAY_SIZE;
}
}

printf("Average value for array a = %.3f\n", avg_a);
printf("Average value for array b = %.3f\n", avg_b);

if(avg_a > avg_b) {
printf("%.3f > %.3f\n", avg_a, avg_b);
} 
else {
printf("%.3f < %.3f\n", avg_a, avg_b);
}
}
