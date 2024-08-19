#include <stdio.h>
#include <stdlib.h>
#include <omp.h>	
long result=0, result_even=0, result_odd=0;
void foo() {
#pragma omp parallel 
{
for (long i = 0; i < 10; i++) {
if (i%2) {
#pragma omp critical(even)
result_even++;
}
else {
#pragma omp critical(odd)
result_odd++;
}
}
#pragma omp barrier
printf("Values for even and odd are %ld and %ld, respectively\n", result_even, result_odd);
#pragma omp critical
result += (result_even + result_odd);
#pragma omp atomic
result++;
#pragma omp flush(result)
#pragma omp barrier
printf("result = %ld\n", result);
#pragma omp barrier
printf("To double check ... result = %ld\n", result);
}
}
int main(int argc, char *argv[]) {
foo();
}
