# include <cmath>
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <ctime>
# include <cstring>
# include <omp.h>
using namespace std;
int main ( );
int i4_min ( int i1, int i2 );
void i4pp_delete ( int **a, int m, int n );
int **i4pp_new ( int m, int n );
void timestamp ( );
/
#pragma omp parallel shared ( b, count, count_max, g, r, x_max, x_min, y_max, y_min ) private ( i, j, k, x, x1, x2, y, y1, y2 )
{
#pragma omp for
for ( i = 0; i < m; i++ )
{
for ( j = 0; j < n; j++ )
{
x = ( ( double ) (     j - 1 ) * x_max   
+ ( double ) ( m - j     ) * x_min ) 
/ ( double ) ( m     - 1 );
y = ( ( double ) (     i - 1 ) * y_max   
+ ( double ) ( n - i     ) * y_min ) 
/ ( double ) ( n     - 1 );
count[i][j] = 0;
x1 = x;
y1 = y;
for ( k = 1; k <= count_max; k++ )
{
x2 = x1 * x1 - y1 * y1 + x;
y2 = 2 * x1 * y1 + y;
if ( x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2 )
{
count[i][j] = k;
break;
}
x1 = x2;
y1 = y2;
}
if ( ( count[i][j] % 2 ) == 1 )
{
r[i][j] = 255;
g[i][j] = 255;
b[i][j] = 255;
}
else
{
c = ( int ) ( 255.0 * sqrt ( sqrt ( sqrt ( 
( ( double ) ( count[i][j] ) / ( double ) ( count_max ) ) ) ) ) );
r[i][j] = 3 * c / 5;
g[i][j] = 3 * c / 5;
b[i][j] = c;
}
}
}
}
wtime = omp_get_wtime ( ) - wtime;
cout << "\n";
cout << "  Time = " << wtime << " seconds.\n";
output.open ( filename.c_str ( ) );
output << "P3\n";
output << n << "  " << m << "\n";
output << 255 << "\n";
for ( i = 0; i < m; i++ )
{
for ( jlo = 0; jlo < n; jlo = jlo + 4 )
{
jhi = i4_min ( jlo + 4, n );
for ( j = jlo; j < jhi; j++ )
{
output << "  " << r[i][j]
<< "  " << g[i][j]
<< "  " << b[i][j] << "\n";
}
output << "\n";
}
}
output.close ( );
cout << "\n";
cout << "  Graphics data written to \"" << filename << "\".\n";
i4pp_delete ( b, m, n );
i4pp_delete ( count, m, n );
i4pp_delete ( g, m, n );
i4pp_delete ( r, m, n );
cout << "\n";
cout << "MANDELBROT_OPENMP\n";
cout << "  Normal end of execution.\n";
cout << "\n";
timestamp ( );
return 0;
}
int i4_min ( int i1, int i2 )
{
int value;
if ( i1 < i2 )
{
value = i1;
}
else
{
value = i2;
}
return value;
}
void i4pp_delete ( int **a, int m, int n )
{
int i;
for ( i = 0; i < m; i++ )
{
delete [] a[i];
}
delete [] a;
return;
}
int **i4pp_new ( int m, int n )
{
int **a;
int i;
a = new int *[m];
if ( a == NULL )
{
cerr << "\n";
cerr << "I4PP_NEW - Fatal error!\n";
cerr << "  Unable to allocate row pointer array.\n";
exit ( 1 );
}
for ( i = 0; i < m; i++ )
{
a[i] = new int[n];
if ( a[i] == NULL )
{
cerr << "\n";
cerr << "I4PP_NEW - Fatal error!\n";
cerr << "  Unable to allocate row array.\n";
exit ( 1 );
}
}
return a;
}
void timestamp ( )
{
# define TIME_SIZE 40
static char time_buffer[TIME_SIZE];
const struct std::tm *tm_ptr;
std::time_t now;
now = std::time ( NULL );
tm_ptr = std::localtime ( &now );
std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );
std::cout << time_buffer << "\n";
return;
# undef TIME_SIZE
}
