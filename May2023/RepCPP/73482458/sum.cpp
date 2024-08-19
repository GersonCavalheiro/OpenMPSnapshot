#include <cstdio>

#define NX 102400

int main(void)
{
long vecA[NX];
long sum, psum, sumex;
int i;


for (i = 0; i < NX; i++) {
vecA[i] = (long) i + 1;
}

sumex = (long) NX * (NX + 1) / ((long) 2);
printf("Arithmetic sum formula (exact):                  %ld\n",
sumex);

sum = 0.0;

#pragma omp parallel for default(shared) private(i)
for (i = 0; i < NX; i++) {
sum += vecA[i];
}
printf("Sum with data race:                              %ld\n",
sum);

sum = 0.0;

#pragma omp parallel for default(shared) private(i)
for (i = 0; i < NX; i++) {
#pragma omp critical(dummy)
sum += vecA[i];
}
printf("Sum using critical section:                      %ld\n",
sum);

sum = 0.0;

#pragma omp parallel default(shared) private(i, psum)
{
psum = 0.0;
#pragma omp for
for (i = 0; i < NX; i++) {
psum += vecA[i];
}
#pragma omp critical(par_sum)
sum += psum;
}
printf("Sum using private variable and critical section: %ld\n",
sum);

sum = 0.0;

#pragma omp parallel for default(shared) private(i) reduction(+:sum)
for (i = 0; i < NX; i++) {
sum += vecA[i];
}
printf("Reduction sum:                                   %ld\n",
sum);

return 0;
}
