# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <math.h>
# include <omp.h>
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
int proc_num;
timestamp ( );
proc_num = omp_get_num_procs ( );
acc = ( double * ) malloc ( nd * np * sizeof ( double ) );
box = ( double * ) malloc ( nd * sizeof ( double ) );
force = ( double * ) malloc ( nd * np * sizeof ( double ) );
pos = ( double * ) malloc ( nd * np * sizeof ( double ) );
vel = ( double * ) malloc ( nd * np * sizeof ( double ) );
printf ( "\n" );
printf ( "MD_OPENMP\n" );
printf ( "  C/OpenMP version\n" );
printf ( "\n" );
printf ( "  A molecular dynamics program.\n" );
printf ( "\n" );
printf ( "  NP, the number of particles in the simulation is %d\n", np );
printf ( "  STEP_NUM, the number of time steps, is %d\n", step_num );
printf ( "  DT, the size of each time step, is %f\n", dt );
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
for ( i = 0; i < nd; i++ )
{
box[i] = 10.0;
}
printf ( "\n" );
printf ( "  Initializing positions, velocities, and accelerations.\n" );
initialize ( np, nd, box, &seed, pos, vel, acc );
printf ( "\n" );
printf ( "  Computing initial forces and energies.\n" );
compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );
e0 = potential + kinetic;
printf ( "\n" );
printf ( "  At each step, we report the potential and kinetic energies.\n" );
printf ( "  The sum of these energies should be a constant.\n" );
printf ( "  As an accuracy check, we also print the relative error\n" );
printf ( "  in the total energy.\n" );
printf ( "\n" );
printf ( "      Step      Potential       Kinetic        (P+K-E0)/E0\n" );
printf ( "                Energy P        Energy K       Relative Energy Error\n" );
printf ( "\n" );
step_print = 0;
step_print_index = 0;
step_print_num = 10;
step = 0;
printf ( "  %8d  %14f  %14f  %14e\n",
step, potential, kinetic, ( potential + kinetic - e0 ) / e0 );
step_print_index = step_print_index + 1;
step_print = ( step_print_index * step_num ) / step_print_num;
wtime = omp_get_wtime ( );
for ( step = 1; step <= step_num; step++ )
{
compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );
if ( step == step_print )
{
printf ( "  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
( potential + kinetic - e0 ) / e0 );
step_print_index = step_print_index + 1;
step_print = ( step_print_index * step_num ) / step_print_num;
}
update ( np, nd, pos, vel, force, acc, mass, dt );
}
wtime = omp_get_wtime ( ) - wtime;
printf ( "\n" );
printf ( "  Elapsed time for main computation:\n" );
printf ( "  %f seconds.\n", wtime );
free ( acc );
free ( box );
free ( force );
free ( pos );
free ( vel );
printf ( "\n" );
printf ( "MD_OPENMP\n" );
printf ( "  Normal end of execution.\n" );
printf ( "\n" );
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
void timestamp ( void )
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
void update ( int np, int nd, double pos[], double vel[], double f[], 
double acc[], double mass, double dt )
{
int i;
int j;
double rmass;
rmass = 1.0 / mass;
#pragma omp parallel shared ( acc, dt, f, nd, np, pos, rmass, vel ) private ( i, j )
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
