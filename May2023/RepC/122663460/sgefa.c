# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <time.h>
int main ( void );
void test01 ( int n );
void test02 ( int n );
void test03 ( int n );
int isamax ( int n, float x[], int incx );
void matgen ( int lda, int n, float a[], float x[], float b[] );
void msaxpy ( int nr, int nc, float a[], int n, float x[], float y[] );
void msaxpy2 ( int nr, int nc, float a[], int n, float x[], float y[] );
int msgefa ( float a[], int lda, int n, int ipvt[] );
int msgefa2 ( float a[], int lda, int n, int ipvt[] );
void saxpy ( int n, float a, float x[], int incx, float y[], int incy );
float sdot ( int n, float x[], int incx, float y[], int incy );
int sgefa ( float a[], int lda, int n, int ipvt[] );
void sgesl ( float a[], int lda, int n, int ipvt[], float b[], int job );
void sscal ( int n, float a, float x[], int incx );
void sswap ( int n, float x[], int incx, float y[], int incy );
void timestamp ( );
int main ( void )
{
int n;
timestamp ( );
printf ( "\n" );
printf ( "SGEFA_OPENMP\n" );
printf ( "  C + OpenMP version\n" );
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
printf ( "\n" );
printf ( " Algorithm        Mode          N    Error       Time\n" );
printf ( "\n" );
n = 10;
test01 ( n );
test02 ( n );
test03 ( n );
printf ( "\n" );
n = 100;
test01 ( n );
test02 ( n );
test03 ( n );
printf ( "\n" );
n = 1000;
test01 ( n );
test02 ( n );
test03 ( n );
printf ( "\n" );
printf ( "SGEFA_OPENMP\n" );
printf ( "  Normal end of execution.\n" );
printf ( "\n" );
timestamp ( );
return 0;
}
void test01 ( int n )
{
float *a;
float *b;
float err;
int i;
int info;
int *ipvt;
int job;
int lda;
double wtime;
float *x;
lda = n;
a = ( float * ) malloc ( lda * n * sizeof ( float ) );
b = ( float * ) malloc ( n * sizeof ( float ) );
x = ( float * ) malloc ( n * sizeof ( float ) );
matgen ( lda, n, a, x, b );
ipvt = ( int * ) malloc ( n * sizeof ( int ) );
wtime = omp_get_wtime ( );
info = sgefa ( a, lda, n, ipvt );
wtime = omp_get_wtime ( ) - wtime;
if ( info != 0 )
{
printf ( "\n" );
printf ( "TEST01 - Fatal error!\n" );
printf ( "  SGEFA reports the matrix is singular.\n" );
exit ( 1 );
}
job = 0;
sgesl ( a, lda, n, ipvt, b, job );
err = 0.0;
for ( i = 0; i < n; i++ )
{
err = err + fabs ( x[i] - b[i] );
}
printf ( "  Original  Sequential   %8d  %10.4e  %10.4e\n", n, err, wtime );
free ( a );
free ( b );
free ( ipvt );
free ( x );
return;
}
void test02 ( int n )
{
float *a;
float *b;
float err;
int i;
int info;
int *ipvt;
int job;
int lda;
double wtime;
float *x;
lda = n;
a = ( float * ) malloc ( lda * n * sizeof ( float ) );
b = ( float * ) malloc ( n * sizeof ( float ) );
x = ( float * ) malloc ( n * sizeof ( float ) );
matgen ( lda, n, a, x, b );
ipvt = ( int * ) malloc ( n * sizeof ( int ) );
wtime = omp_get_wtime ( );
info = msgefa ( a, lda, n, ipvt );
wtime = omp_get_wtime ( ) - wtime;
if ( info != 0 )
{
printf ( "\n" );
printf ( "TEST02 - Fatal error!\n" );
printf ( "  MSGEFA reports the matrix is singular.\n" );
exit ( 1 );
}
job = 0;
sgesl ( a, lda, n, ipvt, b, job );
err = 0.0;
for ( i = 0; i < n; i++ )
{
err = err + fabs ( x[i] - b[i] );
}
printf ( "  Revised     Parallel   %8d  %10.4e  %10.4e\n", n, err, wtime );
free ( a );
free ( b );
free ( ipvt );
free ( x );
return;
}
void test03 ( int n )
{
float *a;
float *b;
float err;
int i;
int info;
int *ipvt;
int job;
int lda;
double wtime;
float *x;
lda = n;
a = ( float * ) malloc ( lda * n * sizeof ( float ) );
b = ( float * ) malloc ( n * sizeof ( float ) );
x = ( float * ) malloc ( n * sizeof ( float ) );
matgen ( lda, n, a, x, b );
ipvt = ( int * ) malloc ( n * sizeof ( int ) );
wtime = omp_get_wtime ( );
info = msgefa2 ( a, lda, n, ipvt );
wtime = omp_get_wtime ( ) - wtime;
if ( info != 0 )
{
printf ( "\n" );
printf ( "TEST03 - Fatal error!\n" );
printf ( "  MSGEFA2 reports the matrix is singular.\n" );
exit ( 1 );
}
job = 0;
sgesl ( a, lda, n, ipvt, b, job );
err = 0.0;
for ( i = 0; i < n; i++ )
{
err = err + fabs ( x[i] - b[i] );
}
printf ( "  Revised   Sequential   %8d  %10.4e  %10.4e\n", n, err, wtime );
free ( a );
free ( b );
free ( ipvt );
free ( x );
return;
}
int isamax ( int n, float x[], int incx )
{
float xmax;
int i;
int ix;
int value;
value = 0;
if ( n < 1 || incx <= 0 )
{
return value;
}
value = 1;
if ( n == 1 )
{
return value;
}
if ( incx == 1 )
{
xmax = fabs ( x[0] );
for ( i = 1; i < n; i++ )
{
if ( xmax < fabs ( x[i] ) )
{
value = i + 1;
xmax = fabs ( x[i] );
}
}
}
else
{
ix = 0;
xmax = fabs ( x[0] );
ix = ix + incx;
for ( i = 1; i < n; i++ )
{
if ( xmax < fabs ( x[ix] ) )
{
value = i + 1;
xmax = fabs ( x[ix] );
}
ix = ix + incx;
}
}
return value;
}
void matgen ( int lda, int n, float a[], float x[], float b[] )
{
int i;
int j;
int seed;
float value;
seed = 1325;
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < n; i++ )
{
seed = ( 3125 * seed ) % 65536;
value = ( ( float ) seed - 32768.0 ) / 16384.0;
a[i+j*lda] = value;
}
}
for ( i = 0; i < n; i++ )
{
x[i] = ( float ) ( i + 1 ) / ( ( float ) n );
}
for ( i = 0; i < n; i++ ) 
{
b[i] = 0.0;
for ( j = 0; j < n; j++ )
{
b[i] = b[i] + a[i+j*lda] * x[j];
}
}
return;
}
void msaxpy ( int nr, int nc, float a[], int n, float x[], float y[] )
{
int i,j;
#pragma omp parallel shared ( a, nc, nr, x, y ) private ( i, j )
#pragma omp for
for ( j = 0; j < nc; j++)
{
for ( i = 0; i < nr; i++ )
{
y[i+j*n] += a[j*n] * x[i];
}
}
return;
}
void msaxpy2 ( int nr, int nc, float a[], int n, float x[], float y[] )
{
int i,j;
for ( j = 0; j < nc; j++)
{
for ( i = 0; i < nr; i++ )
{
y[i+j*n] += a[j*n] * x[i];
}
}
return;
}
int msgefa ( float a[], int lda, int n, int ipvt[] )
{
int info;
int k,kp1,l,nm1;
float t;
info = 0;
nm1 = n - 1;
for ( k = 0; k < nm1; k++ )
{
kp1 = k + 1;
l = isamax ( n-k, a+k+k*lda, 1 ) + k - 1;
ipvt[k] = l + 1;
if ( a[l+k*lda] == 0.0 )
{
info = k + 1;
return info;
}
if ( l != k )
{
t          = a[l+k*lda];
a[l+k*lda] = a[k+k*lda];
a[k+k*lda] = t;
}
t = -1.0 / a[k+k*lda]; 
sscal ( n-k-1, t, a+kp1+k*lda, 1 );
if ( l != k )
{
sswap ( n-k-1, a+l+kp1*lda, lda, a+k+kp1*lda, lda );
}
msaxpy ( n-k-1, n-k-1, a+k+kp1*lda, n, a+kp1+k*lda, a+kp1+kp1*lda );
}
ipvt[n-1] = n;
if ( a[n-1+(n-1)*lda] == 0.0 )
{
info = n;
}
return info;
}
int msgefa2 ( float a[], int lda, int n, int ipvt[] )
{
int info;
int k,kp1,l,nm1;
float t;
info = 0;
nm1 = n - 1;
for ( k = 0; k < nm1; k++ )
{
kp1 = k + 1;
l = isamax ( n-k, a+k+k*lda, 1 ) + k - 1;
ipvt[k] = l + 1;
if ( a[l+k*lda] == 0.0 )
{
info = k + 1;
return info;
}
if ( l != k )
{
t          = a[l+k*lda];
a[l+k*lda] = a[k+k*lda];
a[k+k*lda] = t;
}
t = -1.0 / a[k+k*lda]; 
sscal ( n-k-1, t, a+kp1+k*lda, 1 );
if ( l != k )
{
sswap ( n-k-1, a+l+kp1*lda, lda, a+k+kp1*lda, lda );
}
msaxpy2 ( n-k-1, n-k-1, a+k+kp1*lda, n, a+kp1+k*lda, a+kp1+kp1*lda );
}
ipvt[n-1] = n;
if ( a[n-1+(n-1)*lda] == 0.0 )
{
info = n;
}
return info;
}
void saxpy ( int n, float a, float x[], int incx, float y[], int incy )
{
int i;
int ix;
int iy;
int m;
if ( n <= 0 )
{
return;
}
if ( a == 0.0 )
{
return;
}
if ( incx != 1 || incy != 1 )
{
if ( 0 <= incx )
{
ix = 0;
}
else
{
ix = ( - n + 1 ) * incx;
}
if ( 0 <= incy )
{
iy = 0;
}
else
{
iy = ( - n + 1 ) * incy;
}
for ( i = 0; i < n; i++ )
{
y[iy] = y[iy] + a * x[ix];
ix = ix + incx;
iy = iy + incy;
}
}
else
{
m = n % 4;
for ( i = 0; i < m; i++ )
{
y[i] = y[i] + a * x[i];
}
for ( i = m; i < n; i = i + 4 )
{
y[i  ] = y[i  ] + a * x[i  ];
y[i+1] = y[i+1] + a * x[i+1];
y[i+2] = y[i+2] + a * x[i+2];
y[i+3] = y[i+3] + a * x[i+3];
}
}
return;
}
float sdot ( int n, float x[], int incx, float y[], int incy )
{
int i;
int ix;
int iy;
int m;
float temp;
temp = 0.0;
if ( n <= 0 )
{
return temp;
}
if ( incx != 1 || incy != 1 )
{
if ( 0 <= incx )
{
ix = 0;
}
else
{
ix = ( - n + 1 ) * incx;
}
if ( 0 <= incy )
{
iy = 0;
}
else
{
iy = ( - n + 1 ) * incy;
}
for ( i = 0; i < n; i++ )
{
temp = temp + x[ix] * y[iy];
ix = ix + incx;
iy = iy + incy;
}
}
else
{
m = n % 5;
for ( i = 0; i < m; i++ )
{
temp = temp + x[i] * y[i];
}
for ( i = m; i < n; i = i + 5 )
{
temp = temp + x[i  ] * y[i  ] 
+ x[i+1] * y[i+1] 
+ x[i+2] * y[i+2] 
+ x[i+3] * y[i+3] 
+ x[i+4] * y[i+4];
}
}
return temp;
}
int sgefa ( float a[], int lda, int n, int ipvt[] )
{
int j;
int info;
int k;
int l;
float t;
info = 0;
for ( k = 1; k <= n - 1; k++ ) 
{
l = isamax ( n-k+1, &a[k-1+(k-1)*lda], 1 ) + k - 1;
ipvt[k-1] = l;
if ( a[l-1+(k-1)*lda] != 0.0 ) 
{
if ( l != k ) 
{
t                = a[l-1+(k-1)*lda];
a[l-1+(k-1)*lda] = a[k-1+(k-1)*lda];
a[k-1+(k-1)*lda] = t; 
}
t = - 1.0 / a[k-1+(k-1)*lda];
sscal ( n-k, t, &a[k+(k-1)*lda], 1 );
for ( j = k + 1; j <= n; j++ ) 
{
t = a[l-1+(j-1)*lda];
if (l != k) 
{
a[l-1+(j-1)*lda] = a[k-1+(j-1)*lda];
a[k-1+(j-1)*lda] = t;
}
saxpy ( n-k, t, &a[k+(k-1)*lda], 1, &a[k+(j-1)*lda], 1 );
} 
}
else
{ 
info = k;
}
} 
ipvt[n-1] = n;
if (a[n-1+(n-1)*lda] == 0.0 ) 
{
info = n - 1;
}
return info;
}
void sgesl ( float a[], int lda, int n, int ipvt[], float b[], int job )
{
int k;
int l;
float t;
if ( job == 0 )
{
for ( k = 1; k <= n-1; k++ )
{
l = ipvt[k-1];
t = b[l-1];
if ( l != k )
{
b[l-1] = b[k-1];
b[k-1] = t;
}
saxpy ( n-k, t, a+k+(k-1)*lda, 1, b+k, 1 );
}
for ( k = n; 1 <= k; k-- )
{
b[k-1] = b[k-1] / a[k-1+(k-1)*lda];
t = -b[k-1];
saxpy ( k-1, t, a+0+(k-1)*lda, 1, b, 1 );
}
}
else
{
for ( k = 1; k <= n; k++ )
{
t = sdot ( k-1, a+0+(k-1)*lda, 1, b, 1 );
b[k-1] = ( b[k-1] - t ) / a[k-1+(k-1)*lda];
}
for ( k = n-1; 1 <= k; k-- )
{
b[k-1] = b[k-1] + sdot ( n-k, a+k+(k-1)*lda, 1, b+k, 1 );
l = ipvt[k-1];
if ( l != k )
{
t = b[l-1];
b[l-1] = b[k-1];
b[k-1] = t;
}
}
}
return;
}
void sscal ( int n, float sa, float x[], int incx )
{
int i;
int ix;
int m;
if ( n <= 0 )
{
}
else if ( incx == 1 )
{
m = n % 5;
for ( i = 0; i < m; i++ )
{
x[i] = sa * x[i];
}
for ( i = m; i < n; i = i + 5 )
{
x[i]   = sa * x[i];
x[i+1] = sa * x[i+1];
x[i+2] = sa * x[i+2];
x[i+3] = sa * x[i+3];
x[i+4] = sa * x[i+4];
}
}
else
{
if ( 0 <= incx )
{
ix = 0;
}
else
{
ix = ( - n + 1 ) * incx;
}
for ( i = 0; i < n; i++ )
{
x[ix] = sa * x[ix];
ix = ix + incx;
}
}
return;
}
void sswap ( int n, float x[], int incx, float y[], int incy )
{
int i;
int ix;
int iy;
int m;
float temp;
if ( n <= 0 )
{
}
else if ( incx == 1 && incy == 1 )
{
m = n % 3;
for ( i = 0; i < m; i++ )
{
temp = x[i];
x[i] = y[i];
y[i] = temp;
}
for ( i = m; i < n; i = i + 3 )
{
temp = x[i];
x[i] = y[i];
y[i] = temp;
temp = x[i+1];
x[i+1] = y[i+1];
y[i+1] = temp;
temp = x[i+2];
x[i+2] = y[i+2];
y[i+2] = temp;
}
}
else
{
if ( 0 <= incx )
{
ix = 0;
}
else
{
ix = ( - n + 1 ) * incx;
}
if ( 0 <= incy )
{
iy = 0;
}
else
{
iy = ( - n + 1 ) * incy;
}
for ( i = 0; i < n; i++ )
{
temp = x[ix];
x[ix] = y[iy];
y[iy] = temp;
ix = ix + incx;
iy = iy + incy;
}
}
return;
}
void timestamp ( )
{
# define TIME_SIZE 40
static char time_buffer[TIME_SIZE];
const struct tm *tm;
time_t now;
now = time ( NULL );
tm = localtime ( &now );
strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );
printf ( "%s\n", time_buffer );
return;
# undef TIME_SIZE
}
