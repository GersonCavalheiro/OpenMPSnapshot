#include <stdio.h>
int main(void) {
#pragma omp parallel num_threads(3)
printf("Hello World!\n");
return 0;
}