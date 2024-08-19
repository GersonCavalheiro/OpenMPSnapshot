#include <omp.h>
#include <stdio.h>
int main(){
int i;
double q[10], qq[10];
for (i = 0; i < 10; i++) qq[i] = (double)i;
for (i = 0; i < 10; i++) q[i] = (double)i;
#pragma omp parallel default(shared)
{
#pragma omp for private(i)
for (i = 0; i < 10; i++)
q[i] += qq[i];
#pragma omp critical
{
q[9] += 1.0;
}
#pragma omp barrier
#pragma omp single
{
q[9] = q[9] - 1.0;
}
} 
for (i = 0; i < 10; i++)printf("%f %f\n",qq[i],q[i]);
return 0;
}
