# include <stdlib.h>
# include <stdio.h>
# include <omp.h>
int main ( int argc, char *argv[] ){
int n = 500000;
int not_primes=0;
int i,j;
double  start_time = omp_get_wtime();
#pragma omp parallel for private(j) reduction(+: not_primes) schedule(dynamic)
for ( i = 2; i <= n; i++ ){
for ( j = 2; j < i; j++ ){
if ( i % j == 0 ){
not_primes++;
break;
}
}
}
double  total_time = omp_get_wtime() - start_time;
printf("Primes: %d   Time: %f \n", n - not_primes, total_time);
}
