#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
const double G                    = 6.67300e-11;   
const double EARTH_MASS           = 5.9742e24;     
const double EARTH_DIAMETER       = 12756000.32;   
const double TIMESTEP             =   1.0; 
#define NUMBODIES          100
#define NUMSTEPS           200
struct body{
float mass;
float x, y, z;            
float vx, vy, vz;         
float fx, fy, fz;         
float xnew, ynew, znew;
float vxnew, vynew, vznew;
};
typedef struct body Body;
Body     Bodies[NUMBODIES];
float            GetDistanceSquared( Body *, Body * );
float            GetUnitVector( Body *, Body *, float *, float *, float * );
float            Ranf( float, float );
int              Ranf( int, int );
int main( int argc, char *argv[ ] ){
#ifndef _OPENMP
fprintf( stderr, "OpenMP is not available\n" );
return 1;
#endif
omp_set_num_threads( NUMTHREADS );
int numProcessors = omp_get_num_procs( );
fprintf( stderr, "Have %d processors.\n", numProcessors );
for( int i = 0; i < NUMBODIES; i++ ){
Bodies[i].mass = EARTH_MASS  * Ranf( 0.5f, 10.f );
Bodies[i].x = EARTH_DIAMETER * Ranf( -100.f, 100.f );
Bodies[i].y = EARTH_DIAMETER * Ranf( -100.f, 100.f );
Bodies[i].z = EARTH_DIAMETER * Ranf( -100.f, 100.f );
Bodies[i].vx = Ranf( -100.f, 100.f );;
Bodies[i].vy = Ranf( -100.f, 100.f );;
Bodies[i].vz = Ranf( -100.f, 100.f );;
};
double time0 = omp_get_wtime( );
for( int t = 0; t < NUMSTEPS; t++ ){
#pragma omp parallel for schedule(static) 
for( int i = 0; i < NUMBODIES; i++ ){
float fx = 0.;
float fy = 0.;
float fz = 0.;
Body *bi = &Bodies[i];
for( int j = 0; j < NUMBODIES; j++ ){
if( j == i )     continue;
Body *bj = &Bodies[j];
float rsqd = GetDistanceSquared( bi, bj );
if( rsqd > 0. ){
float f = G * bi->mass * bj->mass / rsqd;
float ux, uy, uz;
GetUnitVector( bi, bj,   &ux, &uy, &uz );
fx += f * ux;
fy += f * uy;
fz += f * uz;
}
}
float ax = fx / Bodies[i].mass;
float ay = fy / Bodies[i].mass;
float az = fz / Bodies[i].mass;
Bodies[i].xnew = Bodies[i].x + Bodies[i].vx*TIMESTEP + 0.5*ax*TIMESTEP*TIMESTEP;
Bodies[i].ynew = Bodies[i].y + Bodies[i].vy*TIMESTEP + 0.5*ay*TIMESTEP*TIMESTEP;
Bodies[i].znew = Bodies[i].z + Bodies[i].vz*TIMESTEP + 0.5*az*TIMESTEP*TIMESTEP;
Bodies[i].vxnew = Bodies[i].vx + ax*TIMESTEP;
Bodies[i].vynew = Bodies[i].vy + ay*TIMESTEP;
Bodies[i].vznew = Bodies[i].vz + az*TIMESTEP;
}
for( int i = 0; i < NUMBODIES; i++){
Bodies[i].x = Bodies[i].xnew;
Bodies[i].y = Bodies[i].ynew;
Bodies[i].z = Bodies[i].znew;
Bodies[i].vx = Bodies[i].vxnew;
Bodies[i].vy = Bodies[i].vynew;
Bodies[i].vz = Bodies[i].vznew;
}
}  
double time1 = omp_get_wtime( );
double megaBodies = ((double)(NUMBODIES*NUMBODIES*NUMSTEPS)/(time1-time0)/1000000.);
printf("Static\n");
printf("Coarse \n");
printf("Threads: %d\n", NUMTHREADS);
printf("Performance = %8.2lf megaBodiesPerSec/Sec\n", megaBodies);
printf("***** \n");
FILE *fp;
fp = fopen("output.txt", "a");
fprintf(fp, "Static\t");
fprintf(fp,"Coarse \t");
fprintf(fp, "%d\t", NUMTHREADS);
fprintf(fp, "%f\t",megaBodies);
fprintf(fp, "\n");
return 0;
}
float GetDistanceSquared( Body *bi, Body *bj ){
float dx = bi->x - bj->x;
float dy = bi->y - bj->y;
float dz = bi->z - bj->z;
return dx*dx + dy*dy + dz*dz;
}
float GetUnitVector( Body *from, Body *to, float *ux, float *uy, float *uz ){
float dx = to->x - from->x;
float dy = to->y - from->y;
float dz = to->z - from->z;
float d = sqrt( dx*dx + dy*dy + dz*dz );
if( d > 0. ){
dx /= d;
dy /= d;
dz /= d;
}
*ux = dx;
*uy = dy;
*uz = dz;
return d;
}
float Ranf( float low, float high ){
float r = (float) rand();         
return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}
int Ranf( int ilow, int ihigh ){
float low = (float)ilow;
float high = (float)ihigh + 0.9999f;
return (int)(  Ranf(low,high) );
}