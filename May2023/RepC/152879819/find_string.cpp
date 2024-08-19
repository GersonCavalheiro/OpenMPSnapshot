#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define N 1000000000
int main(void)
{
char *array = (char*)malloc(sizeof(char) * N);
for (int x=0; x<N-3; x++) {
array[x] = 'a';
}
array[N-3] = 'T';
array[N-2] = 'h';
array[N-1] = 'e';
int i, j;
double start_t = omp_get_wtime();
#pragma omp parallel num_threads(4)
{
char str[] = "The";
#pragma omp for private(j)
for (i=0; i<=N-3; i++) {
for (j=0; j<strlen(str); j++) {
if (str[j] != array[i+j]){
break;
}
}
if (j == strlen(str)) {
printf("found at %d\n",i);
}
}
}
double end_t = omp_get_wtime();
printf("%f\n", end_t - start_t);
}
