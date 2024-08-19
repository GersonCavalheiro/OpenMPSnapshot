#include <stdio.h>
#include <omp.h>
int main() {
int x = 0, y = 0, z = 0;
#pragma omp parallel
{
#pragma omp single
{
#pragma omp task depend(out: x)
{
x = 1;
printf("Task 1: x = %d\n", x);
}
#pragma omp task depend(in: x) depend(out: y)
{
y = x + 1;
printf("Task 2: y = %d\n", y);
}
#pragma omp task depend(in: x) depend(out: z)
{
z = x + 2;
printf("Task 3: z = %d\n", z);
}
#pragma omp task depend(in: y, z)
{
printf("Task 4: x = %d, y = %d, z = %d\n", x, y, z);
}
}
}
return 0;
}
