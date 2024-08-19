#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
int main( int argc, char *argv[] )
{
time_t t;
time_t tnew;
unsigned long long int i;
double sp = 5;
long double j;
unsigned long long int m = 0;
int ex = 6;
unsigned long long int n = 0;
unsigned long long int nIn = 0;
int r = 1;
long double rndX = 0;
long double rndY = 0;
long double rndR = 0;
long double pi = 0;
if( argc >= 2 )
{
ex = atol(argv[1]);
}
n = pow(10, ex);
if( argc >= 3 )
{
sp = atof(argv[2]);
}
printf("Calculating 10 pow %d points ", ex);
#pragma omp parallel private(rndX, rndY, rndR)
{
#pragma omp single
{
time(&t);
printf("with %d threads\n", omp_get_num_threads());
printf("started at %s", ctime(&t));
}
srand((unsigned int) t);
#pragma omp for
for( i = 0; i < n; i++ )
{
rndX = (long double) rand() / RAND_MAX;
rndY = (long double) rand() / RAND_MAX;
rndR = sqrt((pow(rndX, 2) + pow(rndY, 2)));
if( rndR <= r )
{
#pragma omp atomic
nIn++;
}
#pragma omp atomic
m++;
if( m % (int)(n / (100 / sp)) == 0 )
{
#pragma omp critical
{
j = m;
pi = nIn / j * 4.0;
time(&tnew);
printf("%14.10Lf%% in %2d:%02d : %.16Lf\n", (j / n * 100), (int) (difftime(tnew, t) / 60), (int) difftime(tnew, t) % 60, pi);
}
}
}
}
pi = nIn / (long double) m * 4.0;
time(&tnew);
printf("\n%lld / %lld in %2d:%02d: %.16Lf\n", nIn, m, difftime(tnew, t), (int) (difftime(tnew, t) / 60), (int) difftime(tnew, t) % 60, pi);
return 0;
}
