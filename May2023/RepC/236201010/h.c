#include <stdio.h>
int main(void)
{
#pragma omp parallel
printf("Hellow, world.\n");
return 0;
}