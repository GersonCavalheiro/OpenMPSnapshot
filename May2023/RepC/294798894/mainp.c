# include <stdlib.h>
# include <stdio.h>
# include <omp.h>
int main ( int argc, char *argv[] );
int prime_guided ( int n );
int prime_static ( int n );
int prime_dynamic ( int n );
int main ( int argc, char *argv[] )
{
int n,primes;
int n_factor = 2;
int n_hi = 131072;
int n_lo = 1;
double t1,t2,t3;
printf ( "Done by Maitreyee\n" );
printf ( "         n      Pi(n)    Guided Time   Static Time   Dynamic Time\n" );
printf ( "\n" );
n = n_lo;
while ( n <= n_hi )
{
t1 = omp_get_wtime ( );
primes = prime_guided ( n );
t1 = omp_get_wtime ( ) - t1;
t2 = omp_get_wtime ( );
primes = prime_static ( n );
t2 = omp_get_wtime ( ) - t2;
t3 = omp_get_wtime ( );
primes = prime_dynamic ( n );
t3 = omp_get_wtime ( ) - t3;
printf ( "  %8d  %8d  %12f  %12f  %12f\n", n, primes, t1, t2, t3 );
n = n * n_factor;
}
return 0;
}
int prime_guided  ( int n )
{
int i,j,prime;
int total = 0;
#pragma omp parallel shared ( n ) private ( i, j, prime )
#pragma omp for reduction ( + : total ) schedule(guided , 100)
for ( i = 2; i <= n; i++ )
{
prime = 1;
for ( j = 2; j < i; j++ )
{
if ( i % j == 0 )
{
prime = 0;
break;
}
}
total = total + prime;
}
return total;
}
int prime_static ( int n )
{
int i,j,prime;
int total = 0;
#pragma omp parallel shared ( n ) private ( i, j, prime )
#pragma omp for reduction ( + : total ) schedule ( static, 100 )
for ( i = 2; i <= n; i++ )
{
prime = 1;
for ( j = 2; j < i; j++ )
{
if ( i % j == 0 )
{
prime = 0;
break;
}
}
total = total + prime;
}
return total;
}
int prime_dynamic ( int n )
{
int i,j,prime;
int total = 0;
#pragma omp parallel shared ( n ) private ( i, j, prime )
#pragma omp for reduction ( + : total ) schedule ( dynamic, 100 )
for ( i = 2; i <= n; i++ )
{
prime = 1;
for ( j = 2; j < i; j++ )
{
if ( i % j == 0 )
{
prime = 0;
break;
}
}
total = total + prime;
}
return total;
}
