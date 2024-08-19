# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <omp.h>
int main (){
double *b;
double d;
int i;
int it;
int m;
int n;
double r;
double t;
double *x;
double *xnew;
double d_total;
double r_total;
m = 20;
n = 1000000;
x = ( double * ) malloc ( n * sizeof ( double ) );
xnew = ( double * ) malloc ( n * sizeof ( double ) );
for ( i = 0; i < n; i++ )
{
x[i] = 0.0;
xnew[i] = 0.0;
}
double start = omp_get_wtime();
#pragma omp parallel private(t,it,r,d) shared(d_total, r_total)
{
for ( it = 0; it < m; it++ )
{
d_total = 0.0;
r_total = 0.0;
#pragma omp for
for ( i = 0; i < n; i++)
{
if ( 0 == i )
{
xnew[i] = x[i+1]/2;
}
else if ( i == n - 1 ){
xnew[i] = ( x[i-1] + (n+1) ) / 2;
}
else
xnew[i] = ( x[i-1] + x[i+1] ) / 2;
}
#pragma omp barrier
r = 0;
#pragma omp for 
for ( i = 0; i < n; i++ )
{
if(i == 0)
r += (2*xnew[i] - xnew[i+1]) * (2*xnew[i] - xnew[i+1]);
else if (i == n - 1){
r += (2*xnew[i] -(n+1) -xnew[i-1]) *(2*xnew[i] -(n+1) -xnew[i-1]); 
} 
else
r += (2*xnew[i] - xnew[i-1] -xnew[i+1])*(2*xnew[i] - xnew[i-1] -xnew[i+1]);
}
d = 0;
#pragma omp for
for ( i = 0; i < n; i++ )
{
d = d + (x[i] - xnew[i]) *(x[i] - xnew[i]);
x[i] = xnew[i];
}
#pragma omp critical
d_total += d;
#pragma omp critical
r_total += r;
#pragma omp barrier
#pragma omp master
{
printf ( "  iteration:%8d  difference: %.3lf  residual: %.10lf\n", it+1, sqrt (d_total), sqrt (r_total) );
}
#pragma omp barrier
}
}
double end = omp_get_wtime();
double elapsed = end - start;
printf("elapsed time: %.3lf\n", elapsed);
free ( b );
free ( x );
free ( xnew );
return 0;
}
