# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <omp.h>
using namespace std;
int main ( );
void monte_carlo ( int n, int &seed );
double random_value ( int &seed );
void timestamp ( );
int main ( void )
{
int n;
int seed;
timestamp ( );
cout << "\n";
cout << "RANDOM_OPENMP\n";
cout << "  C++ version\n";
cout << "  An OpenMP program using random numbers.\n";
cout << "  The random numbers depend on a seed.\n";
cout << "  We need to insure that each OpenMP thread\n";
cout << "  starts with a different seed.\n";
cout << "\n";
cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
n = 100;
seed = 123456789;
monte_carlo ( n, seed );
cout << "\n";
cout << "RANDOM_OPENMP\n";
cout << "  Normal end of execution.\n";
cout << "\n";
timestamp ( );
return 0;
}
void monte_carlo ( int n, int &seed )
{
int i;
int my_id;
int *my_id_vec;
int my_seed;
int *my_seed_vec;
double *x;
x = new double[n];
my_id_vec = new int[n];
my_seed_vec = new int[n];
#pragma omp master
{
cout << "\n";
cout << "  Thread   Seed  I   X(I)\n";
cout << "\n";
}
#pragma omp parallel private ( i, my_id, my_seed ) shared ( my_id_vec, my_seed_vec, n, x )
{
my_id = omp_get_thread_num ( );
my_seed = seed + my_id;
cout << "  " << setw(6) << my_id
<< "  " << setw(12) << my_seed << "\n";
#pragma omp for
for ( i = 0; i < n; i++ )
{
my_id_vec[i] = my_id;
x[i] = random_value ( my_seed );
my_seed_vec[i] = my_seed;
}
}
for ( i = 0; i < n; i++ )
{
cout << "  " << setw(6) << my_id_vec[i]
<< "  " << setw(12) << my_seed_vec[i]
<< "  " << setw(6) << i
<< "  " << setw(14) << x[i] << "\n";
}
delete [] my_id_vec;
delete [] my_seed_vec;
delete [] x;
return;
}
double random_value ( int &seed )
{
double r;
seed = ( seed % 65536 );
seed = ( ( 3125 * seed ) % 65536 );
r = ( double ) ( seed ) / 65536.0;
return r;
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
cout << time_buffer << "\n";
return;
# undef TIME_SIZE
}
