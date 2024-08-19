#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
#include "omp.h"


#define density 0.0005
#define cutoff  0.01
#define PARICLE_BIN(p) (int)(floor(p.x / cutoff) * bin_size + floor(p.y / cutoff))

int particle_num;
int bin_size;
int num_bins;
int * bin_Ids;

class bin{
public:
int num_par, num_nei;   
int * nei_id;           
int * par_id;           

bin(){
num_nei = num_par = 0;
nei_id = new int[9];
par_id = new int[particle_num];
}
};                          


void init_bins(bin * bins){
int x, y, i, k, next_x, next_y, new_id;
int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

#pragma omp parallel for
for(i = 0; i < num_bins; ++i){
x = i % bin_size;
y = (i - x) / bin_size;
for(k = 0; k < 9; ++k){
next_x = x + dx[k];
next_y = y + dy[k];
if (next_x >= 0 && next_y >= 0 && next_x < bin_size && next_y < bin_size) {
new_id = next_x + next_y * bin_size;
bins[i].nei_id[bins[i].num_nei] = new_id;
bins[i].num_nei++;
}
}
}
return;
}


void binning(bin * bins){
int i, id, idx;
for(i = 0; i < num_bins; ++i){
bins[i].num_par = 0;
}

for(i = 0; i < particle_num; ++i){
id = bin_Ids[i];
idx = bins[id].num_par;
bins[id].par_id[idx] = i;
bins[id].num_par++;
}
return;
}


void apply_force_bin(particle_t * _particles, bin * bins, int i, double * dmin, double * davg, int * navg){
bin * cur_bin = bins + i;
bin * new_bin;
int k, j, par_cur, par_nei;

for(i = 0; i < cur_bin->num_par; ++i){
for(k = 0; k < cur_bin->num_nei; ++k){
new_bin = bins + cur_bin->nei_id[k];
for(j = 0; j < new_bin->num_par; ++j){
par_cur = cur_bin->par_id[i];
par_nei = new_bin->par_id[j];
apply_force(_particles[par_cur],
_particles[par_nei],
dmin, davg, navg);
}
}
}
return;
}

int main( int argc, char **argv )
{
int navg, nabsavg=0, numthreads;
double dmin, absmin=1.0, davg, absavg=0.0;

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
particle_num = n;
set_size( n );

bin_size = (int) ceil(sqrt(density * particle_num) / cutoff);
num_bins = bin_size * bin_size;
bin_Ids =  new int[particle_num];
bin * bins = new bin[num_bins];

init_bins(bins);
init_particles(n, particles);

for(int i = 0; i < particle_num; ++i){
move(particles[i]);
particles[i].ax = particles[i].ay = 0;
bin_Ids[i] = PARICLE_BIN(particles[i]);
}


omp_lock_t * locks = new omp_lock_t[num_bins];
for(int i = 0; i < num_bins; ++i)
omp_init_lock(&locks[i]);

#pragma omp parallel for
for(int i = 0; i < num_bins; ++i){
bins[i].num_par = 0;
}

#pragma omp parallel for
for(int i = 0; i < particle_num; ++i){
int id = bin_Ids[i];
omp_set_lock(&locks[id]); 
int idx = bins[id].num_par;
bins[id].par_id[idx] = i;
bins[id].num_par++;
omp_unset_lock(&locks[id]);
}

double simulation_time = read_timer( );

#pragma omp parallel private(dmin)
{
numthreads = omp_get_num_threads();
for( int step = 0; step < NSTEPS; step++ )
{
navg = 0;
davg = 0.0;
dmin = 1.0;
#pragma omp for
for(int i = 0; i < particle_num; ++i){
particles[i].ax = particles[i].ay = 0;
}

#pragma omp for reduction (+:navg) reduction(+:davg)
for(int i = 0; i < num_bins; ++i){
apply_force_bin(particles, bins, i, &dmin, &davg, &navg);
}

#pragma omp for
for(int i = 0; i < particle_num; ++i){
move(particles[i]);
particles[i].ax = particles[i].ay = 0;
bin_Ids[i] = PARICLE_BIN(particles[i]);
}

#pragma omp for
for(int i = 0; i < num_bins; ++i){
bins[i].num_par = 0;
}

#pragma omp for
for(int i = 0; i < particle_num; ++i){
int id = bin_Ids[i];
omp_set_lock(&locks[id]);
int idx = bins[id].num_par;
bins[id].par_id[idx] = i;
bins[id].num_par++;
omp_unset_lock(&locks[id]);
}


if(find_option( argc, argv, "-no" ) == -1 )
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
simulation_time = read_timer( ) - simulation_time;

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
