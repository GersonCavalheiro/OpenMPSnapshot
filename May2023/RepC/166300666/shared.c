#include <stdio.h>
#include "omp.h"
int main(void) {
int toplam = 0;
#pragma omp parallel shared(toplam) 
{
#pragma omp critical
toplam++;
}
printf("Toplam = %d\n", toplam);
return 0;
}
