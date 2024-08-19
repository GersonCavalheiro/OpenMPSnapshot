# include <cmath>
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <ctime>
# include <omp.h>
using namespace std;
int main ( int argc, char *argv[] );
void compute ( int np, int nd, double pos[], double vel[], 
double mass, double f[], double *pot, double *kin );
double dist ( int nd, double r1[], double r2[], double dr[] );
void initialize ( int np, int nd, double box[], int *seed, double pos[], 
double vel[], double acc[] );
double r8_uniform_01 ( int *seed );
void timestamp ( );
void update ( int np, int nd, double pos[], double vel[], double f[], 
double acc[], double mass, double dt );
int main ( int argc, char *argv[] )
{
double *acc;
double *box;
double dt = 0.0001;
double e0;
double *force;
int i;
double kinetic;
double mass = 1.0;
int nd = 3;
int np = 1000;
double *pos;
double potential;
int seed = 123456789;
int step;
int step_num = 400;
int step_print;
int step_print_index;
int step_print_num;
double *vel;
double wtime;
timestamp ( );
acc = new double[nd*np];
box = new double[nd];
force = new double[nd*np];
pos = new double[nd*np];
vel = new double[nd*np];
cout << "\n";
cout << "MD_OPENMP\n";
cout << "  C++/OpenMP version\n";
cout << "\n";
cout << "  A molecular dynamics program.\n";
cout << "\n";
cout << "  NP, the number of particles in the simulation is " << np << "\n";
cout << "  STEP_NUM, the number of time steps, is " << step_num << "\n";
cout << "  DT, the size of each time step, is " << dt << "\n";;
cout << "\n";
cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
for ( i = 0; i < nd; i++ )
{
box[i] = 10.0;
}
cout << "\n";
cout << "  Initializing positions, velocities, and accelerations.\n" << flush;
initialize ( np, nd, box, &seed, pos, vel, acc );
cout << "\n";
cout << "  Computing initial forces and energies.\n";
compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );
e0 = potential + kinetic;
cout << "\n";
cout << "  At each step, we report the potential and kinetic energies.\n";
cout << "  The sum of these energies should be a constant.\n";
cout << "  As an accuracy check, we also print the relative error\n";
cout << "  in the total energy.\n";
cout << "\n";
cout << "      Step      Potential       Kinetic        (P+K-E0)/E0\n";
cout << "                Energy P        Energy K       Relative Energy Error\n";
cout << "\n";
step_print = 0;
step_print_index = 0;
step_print_num = 10;
step = 0;
cout << "  " << setw(8) << step
<< "  " << setw(14) << potential
<< "  " << setw(14) << kinetic
<< "  " << setw(14) << ( potential + kinetic - e0 ) / e0 << "\n";
step_print_index = step_print_index + 1;
step_print = ( step_print_index * step_num ) / step_print_num;
wtime = omp_get_wtime ( );
for ( step = 1; step <= step_num; step++ )
{
compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );
if ( step == step_print )
{
cout << "  " << setw(8) << step
<< "  " << setw(14) << potential
<< "  " << setw(14) << kinetic
<< "  " << setw(14) << ( potential + kinetic - e0 ) / e0 << "\n";
step_print_index = step_print_index + 1;
step_print = ( step_print_index * step_num ) / step_print_num;
}
update ( np, nd, pos, vel, force, acc, mass, dt );
}
wtime = omp_get_wtime ( ) - wtime;
cout << "\n";
cout << "  Elapsed cpu time for main computation:\n";
cout << "  " << wtime << " seconds.\n";
delete [] acc;
delete [] box;
delete [] force;
delete [] pos;
delete [] vel;
cout << "\n";
cout << "MD_OPENMP\n";
cout << "  Normal end of execution.\n";
cout << "\n";
timestamp ( );
return 0;
}
void compute ( int np, int nd, double pos[], double vel[], 
double mass, double f[], double *pot, double *kin )
{
double d;
double d2;
int i;
int j;
int k;
double ke;
double pe;
double PI2 = 3.141592653589793 / 2.0;
double rij[3];
pe = 0.0;
ke = 0.0;
#pragma omp parallel shared ( f, nd, np, pos, vel ) private ( i, j, k, rij, d, d2 )
#pragma omp for reduction ( + : pe, ke )
for ( k = 0; k < np; k++ )
{
for ( i = 0; i < nd; i++ )
{
f[i+k*nd] = 0.0;
}
for ( j = 0; j < np; j++ )
{
if ( k != j )
{
d = dist ( nd, pos+k*nd, pos+j*nd, rij );
if ( d < PI2 )
{
d2 = d;
}
else
{
d2 = PI2;
}
pe = pe + 0.5 * pow ( sin ( d2 ), 2 );
for ( i = 0; i < nd; i++ )
{
f[i+k*nd] = f[i+k*nd] - rij[i] * sin ( 2.0 * d2 ) / d;
}
}
}
for ( i = 0; i < nd; i++ )
{
ke = ke + vel[i+k*nd] * vel[i+k*nd];
}
}
ke = ke * 0.5 * mass;
*pot = pe;
*kin = ke;
return;
}
double dist ( int nd, double r1[], double r2[], double dr[] )
{
double d;
int i;
d = 0.0;
for ( i = 0; i < nd; i++ )
{
dr[i] = r1[i] - r2[i];
d = d + dr[i] * dr[i];
}
d = sqrt ( d );
return d;
}
void initialize ( int np, int nd, double box[], int *seed, double pos[], 
double vel[], double acc[] )
{
int i;
int j;
for ( i = 0; i < nd; i++ )
{
for ( j = 0; j < np; j++ )
{
pos[i+j*nd] = box[i] * r8_uniform_01 ( seed );
}
}
for ( j = 0; j < np; j++ )
{
for ( i = 0; i < nd; i++ )
{
vel[i+j*nd] = 0.0;
}
}
for ( j = 0; j < np; j++ )
{
for ( i = 0; i < nd; i++ )
{
acc[i+j*nd] = 0.0;
}
}
return;
}
double r8_uniform_01 ( int *seed )
{
int k;
double r;
k = *seed / 127773;
*seed = 16807 * ( *seed - k * 127773 ) - k * 2836;
if ( *seed < 0 )
{
*seed = *seed + 2147483647;
}
r = ( double ) ( *seed ) * 4.656612875E-10;
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
void update ( int np, int nd, double pos[], double vel[], double f[], 
double acc[], double mass, double dt )
{
int i;
int j;
double rmass;
rmass = 1.0 / mass;
#pragma omp parallel shared ( acc, dt, nd, np, pos, rmass, vel ) private ( i, j )
#pragma omp for
for ( j = 0; j < np; j++ )
{
for ( i = 0; i < nd; i++ )
{
pos[i+j*nd] = pos[i+j*nd] + vel[i+j*nd] * dt + 0.5 * acc[i+j*nd] * dt * dt;
vel[i+j*nd] = vel[i+j*nd] + 0.5 * dt * ( f[i+j*nd] * rmass + acc[i+j*nd] );
acc[i+j*nd] = f[i+j*nd] * rmass;
}
}
return;
}
