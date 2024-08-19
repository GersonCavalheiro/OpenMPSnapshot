#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
int main(void) {
int x = 10, y = 10, z = 10;
printf("x before parallel region = %d\n", x);
printf("y before parallel region = %d\n", y);
printf("z before parallel region = %d\n", z);
#pragma omp parallel shared(x) private(y, z)
{    
x = x + 30;
y = y + 20;
z = z + 5;
#pragma omp single private(z)
{
z = z + 100;
printf("z inside omp single = %d\n", z);
}
printf("z outside omp single = %d\n", z);
}
printf("x after parallel region = %d\n", x);
printf("y after parallel region = %d\n", y);
printf("z after parallel region = %d\n", z);
}
