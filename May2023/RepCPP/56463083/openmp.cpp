#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"

int main( int argc, char **argv )
{   
int navg,nabsavg=0,numthreads;
double dmin, absmin=1.0,davg,absavg=0.0;

if( find_option( argc, argv, "-h" ) >= 0 )
{
printf( "Options:\n" );
printf( "-h to see this help\n" );
printf( "-n <int> to set number of particles\n" );
printf( "-o <filename> to specify the output file name\n" );
printf( "-s <filename> to specify a summary file name\n" );
printf( "-no turns off all correctness checks and particle output\n");
return 0;
}

int n = read_int( argc, argv, "-n", 1000 );
char *savename = read_string( argc, argv, "-o", NULL );
char *sumname = read_string( argc, argv, "-s", NULL );

FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
set_size( n );
init_particles( n, particles );

double simulation_time = read_timer( );

#pragma omp parallel private(dmin)
{
numthreads = omp_get_num_threads();
for( int step = 0; step < NSTEPS; step++ )
{
navg = 0;
davg = 0.0;
dmin = 1.0;
#pragma omp for reduction (+:navg) reduction(+:davg)
for( int i = 0; i < n; i++ )
{
particles[i].ax = particles[i].ay = 0;
for (int j = 0; j < n; j++ )
apply_force( particles[i], particles[j],&dmin,&davg,&navg);
}


#pragma omp for
for( int i = 0; i < n; i++ )
move( particles[i] );

if( find_option( argc, argv, "-no" ) == -1 )
{
#pragma omp master
if (navg) {
absavg += davg/navg;
nabsavg++;
}

#pragma omp critical
if (dmin < absmin) absmin = dmin;

#pragma omp master
if( fsave && (step%SAVEFREQ) == 0 )
save( fsave, n, particles );
}
}
}
simulation_time = read_timer() - simulation_time;

printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

if( find_option( argc, argv, "-no" ) == -1 )
{
if (nabsavg) absavg /= nabsavg;
printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
}
printf("\n");

if( fsum)
fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

if( fsum )
fclose( fsum );

free( particles );
if( fsave )
fclose( fsave );

return 0;
}
