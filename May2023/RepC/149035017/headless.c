

#include <stdlib.h>
#include <stdio.h>

#include "timing.h"



#define IX(i,j) ((i)+(N+2)*(j))



extern void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt );
extern void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt );



static int N;
static float dt, diff, visc;
static float force, source;

static float * u, * v, * u_prev, * v_prev;
static float * dens, * dens_prev;





static void free_data ( void )
{
if ( u ) free ( u );
if ( v ) free ( v );
if ( u_prev ) free ( u_prev );
if ( v_prev ) free ( v_prev );
if ( dens ) free ( dens );
if ( dens_prev ) free ( dens_prev );
}

static void clear_data ( void )
{
int i, size=(N+2)*(N+2);

for ( i=0 ; i<size ; i++ ) {
u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
}
}

static int allocate_data ( void )
{
int size = (N+2)*(N+2);

u			= (float *) malloc ( size*sizeof(float) );
v			= (float *) malloc ( size*sizeof(float) );
u_prev		= (float *) malloc ( size*sizeof(float) );
v_prev		= (float *) malloc ( size*sizeof(float) );
dens		= (float *) malloc ( size*sizeof(float) );
dens_prev	= (float *) malloc ( size*sizeof(float) );

if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) {
fprintf ( stderr, "cannot allocate data\n" );
return ( 0 );
}

return ( 1 );
}



static void react ( float * d, float * u, float * v )
{
int size = (N+2)*(N+2);
float max_velocity2 = 0.0f, shared_max_velocity2 = max_velocity2;
float max_density = 0.0f, shared_max_density = max_density;

#pragma omp parallel default(shared) firstprivate(max_velocity2, max_density)
{
#pragma omp for schedule(static) nowait 
for ( unsigned int i = 0 ; i<size ; i++ ) {
if (max_velocity2 < u[i]*u[i] + v[i]*v[i]) {
max_velocity2 = u[i]*u[i] + v[i]*v[i];
}
if (max_density < d[i]) {
max_density = d[i];
}
u[i] = v[i] = d[i] = 0.0f;
}

#pragma omp critical 
{
if (shared_max_density < max_density) {
shared_max_density = max_density;
}
if (shared_max_velocity2 < max_velocity2) {
shared_max_velocity2 = max_velocity2;
}
}
}

if (shared_max_velocity2<0.0000005f) {
u[IX(N/2,N/2)] = force * 10.0f;
v[IX(N/2,N/2)] = force * 10.0f;
}
if (shared_max_density<1.0f) {
d[IX(N/2,N/2)] = source * 10.0f;
}

return;
}

static void one_step ( void )
{
static int times = 1;
static double start_t = 0.0;
static double one_second = 0.0;
static double react_ns_p_cell = 0.0;
static double vel_ns_p_cell = 0.0;
static double dens_ns_p_cell = 0.0;

start_t = wtime();
react ( dens_prev, u_prev, v_prev );
react_ns_p_cell += 1.0e9 * (wtime()-start_t)/(N*N);

start_t = wtime();
vel_step ( N, u, v, u_prev, v_prev, visc, dt );
vel_ns_p_cell += 1.0e9 * (wtime()-start_t)/(N*N);

start_t = wtime();
dens_step ( N, dens, dens_prev, u, v, diff, dt );
dens_ns_p_cell += 1.0e9 * (wtime()-start_t)/(N*N);

if (1.0<wtime()-one_second) { 
printf("%lf, %lf, %lf, %lf: ns per cell total, react, vel_step, dens_step\n",
(react_ns_p_cell+vel_ns_p_cell+dens_ns_p_cell)/times,
react_ns_p_cell/times, vel_ns_p_cell/times, dens_ns_p_cell/times);
one_second = wtime();
react_ns_p_cell = 0.0;
vel_ns_p_cell = 0.0;
dens_ns_p_cell = 0.0;
times = 1;
} else {
times++;
}
}




int main ( int argc, char ** argv )
{
int i = 0;

if ( argc != 1 && argc != 2 && argc != 6 ) {
fprintf ( stderr, "usage : %s N dt diff visc force source\n", argv[0] );
fprintf ( stderr, "where:\n" );\
fprintf ( stderr, "\t N      : grid resolution\n" );
fprintf ( stderr, "\t dt     : time step\n" );
fprintf ( stderr, "\t diff   : diffusion rate of the density\n" );
fprintf ( stderr, "\t visc   : viscosity of the fluid\n" );
fprintf ( stderr, "\t force  : scales the mouse movement that generate a force\n" );
fprintf ( stderr, "\t source : amount of density that will be deposited\n" );
exit ( 1 );
}

if ( argc == 1 ) {
N = 128;
dt = 0.1f;
diff = 0.0f;
visc = 0.0f;
force = 5.0f;
source = 100.0f;
fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
N, dt, diff, visc, force, source );
} else if (argc == 2) {
N = atoi( argv[1] );
dt = 0.1f;
diff = 0.0f;
visc = 0.0f;
force = 5.0f;
source = 100.0f;
} else {
N = atoi(argv[1]);
dt = atof(argv[2]);
diff = atof(argv[3]);
visc = atof(argv[4]);
force = atof(argv[5]);
source = atof(argv[6]);
}

if ( !allocate_data () ) exit ( 1 );
clear_data ();
for (i=0; i<2048; i++)
one_step ();
free_data ();

exit ( 0 );
}
