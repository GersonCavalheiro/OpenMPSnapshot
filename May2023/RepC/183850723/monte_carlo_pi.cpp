#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
double r1, r2, pi;
double m, n;
long t, j;
m = n = j = 0;
printf("Enter the number of iterations used to approximate pi: ");
scanf("%ld", &t);
printf("\nRAND_MAX: %d\n\n", RAND_MAX);
srand(time(NULL ));
while (1)
{
#pragma omp parallel for reduction(+:m,n) private(r1,r2)
for (long i = 0; i < t; i++)
{
m++;
r1 = (double)rand() / 16384.0 - 1.0;
r2 = (double)rand() / 16384.0 - 1.0;
if ( (r1*r1+r2*r2) < 1.0 )
n++;
}
pi = 4.0 * n / m;
printf("[%10ld] %.14f\r", ++j, pi);
}
return 0;
}
