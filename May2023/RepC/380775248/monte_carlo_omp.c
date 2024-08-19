#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "time.h"

#define N 1000000000

int main(void) {

srand(time(NULL));

int A = 0, i;
double x, y;



int P = 0;
#pragma omp parallel private(i, x, y, A) shared(P)
{ 
#pragma omp for schedule(dynamic) 
for (i = 0; i < N; i++) {
x = (double)rand() / RAND_MAX;
y = (double)rand() / RAND_MAX;
if ( x * x + y * y <= 1 )
A++;
}

#pragma omp critical
P += A;

} 



printf("Approx. PI = %.16f\n", 4.0 * (double)P / N);
printf("P = %d\n", P);

}