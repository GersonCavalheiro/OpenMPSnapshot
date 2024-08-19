#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef _WIN32
#include <sys/time.h>
#endif
#ifdef _WIN32
#define NULL    ((void *)0)
#endif
#define NSPEEDS         9
#define PARAMFILE       "input.params"
#define OBSTACLEFILE    "obstacles_300x200.dat"
#define FINALSTATEFILE  "final_state%s%s.dat"
#define AVVELSFILE      "av_vels%s%s.dat"
char finalStateFile[128];
char avVelocityFile[128];
typedef struct {
int nx;            
int ny;            
int maxIters;      
int reynolds_dim;  
double density;    
double accel;      
double omega;      
} t_param;
typedef struct {
double speeds[NSPEEDS];
} t_speed;
int initialise(t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr, int **obstacles_ptr, double **av_vels_ptr);
int timestep(t_param params, t_speed *src_cells, t_speed *dst_cells, const int *obstacles);
int accelerate_flow(t_param params, t_speed *cells, const int *obstacles);
int collision(t_param params, t_speed *dst_cells, t_speed *src_cells, const int *obstacles);
int write_values(t_param params, t_speed *cells, int *obstacles, double *av_vels);
int finalise(t_speed **cells_ptr, t_speed **tmp_cells_ptr, int **obstacles_ptr, double **av_vels_ptr);
double total_density(t_param params, t_speed *cells);
double av_velocity(t_param params, t_speed *cells, const int *obstacles);
double calc_reynolds(t_param params, t_speed *cells, int *obstacles);
void die(const char *message, const int line, const char *file);
int main(int argc, char *argv[]) {
t_param params;            
t_speed *src_cells = NULL; 
t_speed *dst_cells = NULL; 
t_speed *temp_swap = NULL; 
int *obstacles = NULL;     
double *av_vels = NULL;    
int ii;                    
clock_t tic, toc;          
printf("Running Boltzman OpenMP simulation...\n");
#ifndef _WIN32
struct timeval tv1, tv2, tv3;
#endif
if (argc > 1) {
sprintf(finalStateFile, FINALSTATEFILE, ".", argv[1]);
sprintf(avVelocityFile, AVVELSFILE, ".", argv[1]);
} else {
sprintf(finalStateFile, FINALSTATEFILE, "", "");
sprintf(avVelocityFile, AVVELSFILE, "", "");
}
initialise(&params, &src_cells, &dst_cells, &obstacles, &av_vels);
tic = clock();
#ifndef _WIN32
gettimeofday(&tv1, NULL);
#endif
for (ii = 0; ii < params.maxIters; ii++) {
timestep(params, src_cells, dst_cells, obstacles);
temp_swap = src_cells;
src_cells = dst_cells;
dst_cells = temp_swap;
av_vels[ii] = av_velocity(params, src_cells, obstacles);
#ifdef DEBUG
printf("==timestep: %d==\n",ii);
printf("av velocity: %.12E\n", av_vels[ii]);
printf("tot density: %.12E\n",total_density(params,src_cells));
#endif
}
#ifndef _WIN32
gettimeofday(&tv2, NULL);
timersub(&tv2, &tv1, &tv3);
#endif
toc = clock();
printf("==done==\n");
printf("Reynolds number:\t%.12E\n", calc_reynolds(params, src_cells, obstacles));
printf("Elapsed CPU time:\t%ld (ms)\n", (toc - tic) / (CLOCKS_PER_SEC / 1000));
#ifndef _WIN32
printf("Elapsed wall time:\t%ld (ms)\n", (tv3.tv_sec * 1000) + (tv3.tv_usec / 1000));
#endif
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
printf("Threads used:\t\t%d\n", omp_get_num_threads());
#endif
write_values(params, src_cells, obstacles, av_vels);
finalise(&src_cells, &dst_cells, &obstacles, &av_vels);
return EXIT_SUCCESS;
}
int timestep(const t_param params, t_speed *src_cells, t_speed *dst_cells, const int *obstacles) {
accelerate_flow(params, src_cells, obstacles);
collision(params, src_cells, dst_cells, obstacles);
return EXIT_SUCCESS;
}
int accelerate_flow(const t_param params, t_speed *cells, const int *obstacles) {
int ii, offset; 
double *speeds;
const double w1 = params.density * params.accel / 9.0;
const double w2 = params.density * params.accel / 36.0;
#pragma omp parallel for private(ii, offset, speeds) shared(cells)
for (ii = 0; ii < params.ny; ii++) {
offset = ii * params.nx ;
speeds = cells[offset].speeds;
if (!obstacles[offset] && (speeds[3] - w1) > 0.0 && (speeds[6] - w2) > 0.0 && (speeds[7] - w2) > 0.0) {
speeds[1] += w1;
speeds[5] += w2;
speeds[8] += w2;
speeds[3] -= w1;
speeds[6] -= w2;
speeds[7] -= w2;
}
}
return EXIT_SUCCESS;
}
int collision(const t_param params, t_speed *src_cells, t_speed *dst_cells, const int *obstacles) {
int ii, jj, kk, offset;            
#ifdef BOLTZMANN_ACCURATE
const double c_sq = 1.0 / 3.0;     
#endif
const double w0 = 4.0 / 9.0;       
const double w1 = 1.0 / 9.0;       
const double w2 = 1.0 / 36.0;      
double u_x, u_y;                   
double u[NSPEEDS];                 
double d_equ[NSPEEDS];             
double u_sq;                       
double local_density;              
int x_e, x_w, y_n, y_s;            
double *dst_speeds;
double speeds[NSPEEDS];
#pragma omp parallel for private(ii, jj, kk, offset, speeds, dst_speeds, local_density, u_x, u_y, u_sq, u, d_equ, x_e, x_w, y_n, y_s) shared(src_cells, dst_cells)
for (ii = 0; ii < params.ny; ii++) {
for (jj = 0; jj < params.nx; jj++) {
offset = ii * params.nx + jj;
dst_speeds = dst_cells[offset].speeds;
y_s = (ii + 1) % params.ny;
x_w = (jj + 1) % params.nx;
y_n = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
x_e = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
if (obstacles[ii * params.nx + jj]) {
dst_speeds[0] = src_cells[ii * params.nx + jj].speeds[0];  
dst_speeds[1] = src_cells[ii * params.nx + x_w].speeds[3];  
dst_speeds[2] = src_cells[y_s * params.nx + jj].speeds[4];  
dst_speeds[3] = src_cells[ii * params.nx + x_e].speeds[1];  
dst_speeds[4] = src_cells[y_n * params.nx + jj].speeds[2];  
dst_speeds[5] = src_cells[y_s * params.nx + x_w].speeds[7];  
dst_speeds[6] = src_cells[y_s * params.nx + x_e].speeds[8];  
dst_speeds[7] = src_cells[y_n * params.nx + x_e].speeds[5];  
dst_speeds[8] = src_cells[y_n * params.nx + x_w].speeds[6];  
} else {
speeds[0] = src_cells[ii * params.nx + jj].speeds[0];  
speeds[1] = src_cells[ii * params.nx + x_e].speeds[1];  
speeds[2] = src_cells[y_n * params.nx + jj].speeds[2];  
speeds[3] = src_cells[ii * params.nx + x_w].speeds[3];  
speeds[4] = src_cells[y_s * params.nx + jj].speeds[4];  
speeds[5] = src_cells[y_n * params.nx + x_e].speeds[5];  
speeds[6] = src_cells[y_n * params.nx + x_w].speeds[6];  
speeds[7] = src_cells[y_s * params.nx + x_w].speeds[7];  
speeds[8] = src_cells[y_s * params.nx + x_e].speeds[8];  
local_density =
speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] +
speeds[8];
u_x = (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
u_y = (speeds[2] + speeds[5] + speeds[6] - (speeds[4] + speeds[7] + speeds[8])) / local_density;
u[1] = u_x;       
u[2] = u_y;       
u[5] = u_x + u_y; 
u[6] = -u_x + u_y; 
#ifdef BOLTZMANN_ACCURATE
u_sq = (u_x * u_x + u_y * u_y) / (2.0 * c_sq);
d_equ[0] = w0 * local_density * (1.0 - u_sq);
d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq + (u[1] * u[1]) / (2.0 * c_sq * c_sq) - u_sq);
d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq + (u[2] * u[2]) / (2.0 * c_sq * c_sq) - u_sq);
d_equ[3] = w1 * local_density * (1.0 - u[1] / c_sq + (u[1] * u[1]) / (2.0 * c_sq * c_sq) - u_sq);
d_equ[4] = w1 * local_density * (1.0 - u[2] / c_sq + (u[2] * u[2]) / (2.0 * c_sq * c_sq) - u_sq);
d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq + (u[5] * u[5]) / (2.0 * c_sq * c_sq) - u_sq);
d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq + (u[6] * u[6]) / (2.0 * c_sq * c_sq) - u_sq);
d_equ[7] = w2 * local_density * (1.0 - u[5] / c_sq + (u[5] * u[5]) / (2.0 * c_sq * c_sq) - u_sq);
d_equ[8] = w2 * local_density * (1.0 - u[6] / c_sq + (u[6] * u[6]) / (2.0 * c_sq * c_sq) - u_sq);
#else
u_sq = (u_x * u_x + u_y * u_y) * 1.5;
d_equ[0] = w0 * local_density * (1.0 - u_sq);
#ifdef BOLTZMANN_NO_DIVISION
d_equ[1] = w1 * local_density * (1.0 + u[1] * 3.0 + (u[1] * u[1] * 4.5) - u_sq);
d_equ[2] = w1 * local_density * (1.0 + u[2] * 3.0 + (u[2] * u[2] * 4.5) - u_sq);
d_equ[3] = w1 * local_density * (1.0 - u[1] * 3.0 + (u[1] * u[1] * 4.5) - u_sq);
d_equ[4] = w1 * local_density * (1.0 - u[2] * 3.0 + (u[2] * u[2] * 4.5) - u_sq);
d_equ[5] = w2 * local_density * (1.0 + u[5] * 3.0 + (u[5] * u[5] * 4.5) - u_sq);
d_equ[6] = w2 * local_density * (1.0 + u[6] * 3.0 + (u[6] * u[6] * 4.5) - u_sq);
d_equ[7] = w2 * local_density * (1.0 - u[5] * 3.0 + (u[5] * u[5] * 4.5) - u_sq);
d_equ[8] = w2 * local_density * (1.0 - u[6] * 3.0 + (u[6] * u[6] * 4.5) - u_sq);
#else 
u[0] = 1.0 + (u[1] * u[1] * 4.5) - u_sq;
d_equ[1] = w1 * local_density * (u[0] + u[1] * 3.0);
d_equ[3] = w1 * local_density * (u[0] - u[1] * 3.0);
u[0] = 1.0 + (u[2] * u[2] * 4.5) - u_sq;
d_equ[2] = w1 * local_density * (u[0] + u[2] * 3.0);
d_equ[4] = w1 * local_density * (u[0] - u[2] * 3.0);
u[0] = 1.0 + (u[5] * u[5] * 4.5) - u_sq;
d_equ[5] = w2 * local_density * (u[0] + u[5] * 3.0);
d_equ[7] = w2 * local_density * (u[0] - u[5] * 3.0);
u[0] = 1.0 + (u[6] * u[6] * 4.5) - u_sq;
d_equ[6] = w2 * local_density * (u[0] + u[6] * 3.0);
d_equ[8] = w2 * local_density * (u[0] - u[6] * 3.0);
#endif
#endif
for (kk = 0; kk < NSPEEDS; kk++) {
speeds[kk] += params.omega * (d_equ[kk] - speeds[kk]);
}
*((t_speed *) dst_speeds) = *((t_speed *) speeds);
}
}
}
return EXIT_SUCCESS;
}
int
initialise(t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr, int **obstacles_ptr, double **av_vels_ptr) {
FILE *fp;          
int ii, jj;        
int xx, yy;        
int blocked;       
int retval;        
double w0, w1, w2; 
fp = fopen(PARAMFILE, "r");
if (fp == NULL) {
die("could not open file input.params", __LINE__, __FILE__);
}
retval = fscanf(fp, "%d\n", &(params->nx));
if (retval != 1)
die("could not read param file: nx", __LINE__, __FILE__);
retval = fscanf(fp, "%d\n", &(params->ny));
if (retval != 1)
die("could not read param file: ny", __LINE__, __FILE__);
retval = fscanf(fp, "%d\n", &(params->maxIters));
if (retval != 1)
die("could not read param file: maxIters", __LINE__, __FILE__);
retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
if (retval != 1)
die("could not read param file: reynolds_dim", __LINE__, __FILE__);
retval = fscanf(fp, "%lf\n", &(params->density));
if (retval != 1)
die("could not read param file: density", __LINE__, __FILE__);
retval = fscanf(fp, "%lf\n", &(params->accel));
if (retval != 1)
die("could not read param file: accel", __LINE__, __FILE__);
retval = fscanf(fp, "%lf\n", &(params->omega));
if (retval != 1)
die("could not read param file: omega", __LINE__, __FILE__);
fclose(fp);
*cells_ptr = (t_speed *) malloc(sizeof(t_speed) * (params->ny * params->nx));
if (*cells_ptr == NULL)
die("cannot allocate memory for cells", __LINE__, __FILE__);
*tmp_cells_ptr = (t_speed *) malloc(sizeof(t_speed) * (params->ny * params->nx));
if (*tmp_cells_ptr == NULL)
die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
*obstacles_ptr = malloc(sizeof(int *) * (params->ny * params->nx));
if (*obstacles_ptr == NULL)
die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
w0 = params->density * 4.0 / 9.0;
w1 = params->density / 9.0;
w2 = params->density / 36.0;
for (ii = 0; ii < params->ny; ii++) {
for (jj = 0; jj < params->nx; jj++) {
(*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
(*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
(*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
(*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
(*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
(*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
(*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
(*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
(*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
}
}
for (ii = 0; ii < params->ny; ii++) {
for (jj = 0; jj < params->nx; jj++) {
(*obstacles_ptr)[ii * params->nx + jj] = 0;
}
}
fp = fopen(OBSTACLEFILE, "r");
if (fp == NULL) {
die("could not open file obstacles", __LINE__, __FILE__);
}
while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
if (retval != 3)
die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
if (xx < 0 || xx > params->nx - 1)
die("obstacle x-coord out of range", __LINE__, __FILE__);
if (yy < 0 || yy > params->ny - 1)
die("obstacle y-coord out of range", __LINE__, __FILE__);
if (blocked != 1)
die("obstacle blocked value should be 1", __LINE__, __FILE__);
(*obstacles_ptr)[yy * params->nx + xx] = blocked;
}
fclose(fp);
*av_vels_ptr = (double *) malloc(sizeof(double) * params->maxIters);
return EXIT_SUCCESS;
}
int finalise(t_speed **cells_ptr, t_speed **tmp_cells_ptr, int **obstacles_ptr, double **av_vels_ptr) {
free(*cells_ptr);
*cells_ptr = NULL;
free(*tmp_cells_ptr);
*tmp_cells_ptr = NULL;
free(*obstacles_ptr);
*obstacles_ptr = NULL;
free(*av_vels_ptr);
*av_vels_ptr = NULL;
return EXIT_SUCCESS;
}
double av_velocity(const t_param params, t_speed *cells, const int *obstacles) {
int ii, jj, offset;     
int tot_cells = 0;      
double local_density;   
double tot_u_x;         
double *speeds;         
tot_u_x = 0.0;
#pragma omp parallel for reduction(+ : tot_cells, tot_u_x) private(ii, jj, offset, speeds, local_density) shared(obstacles, cells)
for (ii = 0; ii < params.ny; ii++) {
for (jj = 0; jj < params.nx; jj++) {
offset = ii * params.nx + jj;
if (!obstacles[offset]) {
speeds = cells[offset].speeds;
local_density =
speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] +
speeds[8];
tot_u_x += (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
++tot_cells;
}
}
}
return tot_u_x / (double) tot_cells;
}
double calc_reynolds(const t_param params, t_speed *cells, int *obstacles) {
const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}
double total_density(const t_param params, t_speed *cells) {
int ii, jj, kk;     
double total = 0.0; 
#pragma omp parallel for reduction(+:total) private(ii, jj, kk)
for (ii = 0; ii < params.ny; ii++) {
for (jj = 0; jj < params.nx; jj++) {
for (kk = 0; kk < NSPEEDS; kk++) {
total += cells[ii * params.nx + jj].speeds[kk];
}
}
}
return total;
}
int write_values(const t_param params, t_speed *cells, int *obstacles, double *av_vels) {
FILE *fp; 
int ii, jj, offset; 
const double c_sq = 1.0 / 3.0; 
double local_density; 
double pressure; 
double u_x; 
double u_y; 
double *speeds;
fp = fopen(finalStateFile, "w");
if (fp == NULL) {
die("could not open file output file", __LINE__, __FILE__);
}
for (ii = 0; ii < params.ny; ii++) {
for (jj = 0; jj < params.nx; jj++) {
offset = ii * params.nx + jj;
if (obstacles[offset]) {
u_x = u_y = 0.0;
pressure = params.density * c_sq;
} else {
speeds = cells[offset].speeds;
local_density =
speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] +
speeds[8];
u_x = (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
u_y = (speeds[2] + speeds[5] + speeds[6] - (speeds[4] + speeds[7] + speeds[8])) / local_density;
pressure = local_density * c_sq;
}
fprintf(fp, "%d %d %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, pressure, obstacles[offset]);
}
}
fclose(fp);
fp = fopen(avVelocityFile, "w");
if (fp == NULL) {
die("could not open file output file", __LINE__, __FILE__);
}
for (ii = 0; ii < params.maxIters; ii++) {
fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
}
fclose(fp);
return EXIT_SUCCESS;
}
void die(const char *message, const int line, const char *file) {
fprintf(stderr, "Error at line %d of file %s:\n", line, file);
fprintf(stderr, "%s\n", message);
fflush(stderr);
exit(EXIT_FAILURE);
}
