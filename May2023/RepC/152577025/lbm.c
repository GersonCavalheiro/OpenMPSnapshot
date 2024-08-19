#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
typedef struct {
int    nx;            
int    ny;            
int    maxIters;      
int    reynolds_dim;  
float density;       
float accel;         
float omega;         
} t_param;
typedef struct {
float* speed_0;
float* speed_1;
float* speed_2;
float* speed_3;
float* speed_4;
float* speed_5;
float* speed_6;
float* speed_7;
float* speed_8;
} t_soa;
int initialise(const char* paramfile, const char* obstaclefile,
t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
int** obstacles_ptr, float** av_vels_ptr);
int timestep(const t_param params, t_soa* restrict cells, t_soa* restrict tmp_cells, int* restrict obstacles, float* restrict av_vels, const int tt);
int accelerate_flow(const t_param params, t_soa* restrict cells, int* restrict obstacles);
float prop_rebound_collision_avels(const t_param params, t_soa* restrict cells, t_soa* restrict tmp_cells, int* restrict obstacles);
int write_values(const t_param params, t_soa* cells, int* obstacles, float* av_vels);
int finalise(const t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
int** obstacles_ptr, float** av_vels_ptr);
float total_density(const t_param params, t_soa* cells);
float av_velocity(const t_param params, t_soa* cells, int* obstacles);
float calc_reynolds(const t_param params, t_soa* cells, int* obstacles);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
int main(int argc, char* argv[]) {
char*    paramfile = NULL;    
char*    obstaclefile = NULL; 
t_param  params;              
t_soa* cells     = NULL;    
t_soa* tmp_cells = NULL;    
int*     obstacles = NULL;    
float* av_vels   = NULL;      
struct timeval timstr;        
struct rusage ru;             
double tic, toc;              
double usrtim;                
double systim;                
if (argc != 3) {
usage(argv[0]);
} else {
paramfile = argv[1];
obstaclefile = argv[2];
}
initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
gettimeofday(&timstr, NULL);
tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
for (int tt = 0; tt < params.maxIters; tt++) {
timestep(params,     cells, tmp_cells, obstacles, av_vels, tt); tt++;
timestep(params, tmp_cells,     cells, obstacles, av_vels, tt);
}
gettimeofday(&timstr, NULL);
toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
getrusage(RUSAGE_SELF, &ru);
timstr = ru.ru_utime;
usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
timstr = ru.ru_stime;
systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
printf("==done==\n");
printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
printf(" memory bandwidth: %lf GB/s\n", (4 * 18 * (double)(params.nx / 1024) * (double)(params.ny / 1024) * params.maxIters) / ((toc - tic) * 1024) );
write_values(params, cells, obstacles, av_vels);
finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
return EXIT_SUCCESS;
}
int timestep(const t_param params, t_soa* restrict cells, t_soa* restrict tmp_cells, int* restrict obstacles, float* restrict av_vels, const int tt) {
accelerate_flow(params, cells, obstacles);
av_vels[tt] = prop_rebound_collision_avels(params, cells, tmp_cells, obstacles);
#ifdef DEBUG
printf("==timestep: %d==\n", tt);
printf("av velocity: %.12E\n", av_vels[tt]);
printf("tot density: %.12E\n", total_density(params, tmp_cells));
#endif
return EXIT_SUCCESS;
}
int accelerate_flow(const t_param params, t_soa* restrict cells, int* restrict obstacles) {
const float w1 = params.density * params.accel / 9.f;
const float w2 = params.density * params.accel / 36.f;
const int jj = params.ny - 2;
__assume_aligned(cells->speed_0, 64);
__assume_aligned(cells->speed_1, 64);
__assume_aligned(cells->speed_2, 64);
__assume_aligned(cells->speed_3, 64);
__assume_aligned(cells->speed_4, 64);
__assume_aligned(cells->speed_5, 64);
__assume_aligned(cells->speed_6, 64);
__assume_aligned(cells->speed_7, 64);
__assume_aligned(cells->speed_8, 64);
#pragma omp simd
for (int ii = 0; ii < params.nx; ii++) {
if (!obstacles[ii + jj*params.nx]
&& (cells->speed_3[ii + jj*params.nx] - w1) > 0.f
&& (cells->speed_6[ii + jj*params.nx] - w2) > 0.f
&& (cells->speed_7[ii + jj*params.nx] - w2) > 0.f) {
cells->speed_1[ii + jj*params.nx] += w1;
cells->speed_5[ii + jj*params.nx] += w2;
cells->speed_8[ii + jj*params.nx] += w2;
cells->speed_3[ii + jj*params.nx] -= w1;
cells->speed_6[ii + jj*params.nx] -= w2;
cells->speed_7[ii + jj*params.nx] -= w2;
}
}
return EXIT_SUCCESS;
}
float prop_rebound_collision_avels(const t_param params, t_soa* restrict cells, t_soa* restrict tmp_cells, int* restrict obstacles) {
const float c_sq = 1.f / 3.f;  
const float w0   = 4.f / 9.f;  
const float w1   = 1.f / 9.f;  
const float w2   = 1.f / 36.f; 
int   tot_cells  = 0;   
float tot_u      = 0.f; 
#pragma omp parallel for reduction(+:tot_u), reduction(+:tot_cells)
for (int jj = 0; jj < params.ny; jj++) {
__assume_aligned(cells->speed_0, 64);
__assume_aligned(cells->speed_1, 64);
__assume_aligned(cells->speed_2, 64);
__assume_aligned(cells->speed_3, 64);
__assume_aligned(cells->speed_4, 64);
__assume_aligned(cells->speed_5, 64);
__assume_aligned(cells->speed_6, 64);
__assume_aligned(cells->speed_7, 64);
__assume_aligned(cells->speed_8, 64);
__assume_aligned(tmp_cells->speed_0, 64);
__assume_aligned(tmp_cells->speed_1, 64);
__assume_aligned(tmp_cells->speed_2, 64);
__assume_aligned(tmp_cells->speed_3, 64);
__assume_aligned(tmp_cells->speed_4, 64);
__assume_aligned(tmp_cells->speed_5, 64);
__assume_aligned(tmp_cells->speed_6, 64);
__assume_aligned(tmp_cells->speed_7, 64);
__assume_aligned(tmp_cells->speed_8, 64);
#pragma omp simd
for (int ii = 0; ii < params.nx; ii++) {
const int y_n = (jj + 1) % params.ny;
const int x_e = (ii + 1) % params.nx;
const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
const float s0 = cells->speed_0[ii + jj*params.nx]; 
const float s1 = cells->speed_1[x_w + jj*params.nx]; 
const float s2 = cells->speed_2[ii + y_s*params.nx]; 
const float s3 = cells->speed_3[x_e + jj*params.nx]; 
const float s4 = cells->speed_4[ii + y_n*params.nx]; 
const float s5 = cells->speed_5[x_w + y_s*params.nx]; 
const float s6 = cells->speed_6[x_e + y_s*params.nx]; 
const float s7 = cells->speed_7[x_e + y_n*params.nx]; 
const float s8 = cells->speed_8[x_w + y_n*params.nx]; 
const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
const float u_x = (s1 + s5 + s8 - (s3 + s6 + s7)) / local_density;
const float u_y = (s2 + s5 + s6 - (s4 + s7 + s8)) / local_density;
const float u_sq = u_x * u_x + u_y * u_y;
float u[NSPEEDS];
u[1] =   u_x;        
u[2] =         u_y;  
u[3] = - u_x;        
u[4] =       - u_y;  
u[5] =   u_x + u_y;  
u[6] = - u_x + u_y;  
u[7] = - u_x - u_y;  
u[8] =   u_x - u_y;  
float d_equ[NSPEEDS];
d_equ[0] = w0 * local_density
* (1.f - u_sq / (2.f * c_sq));
d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
+ (u[1] * u[1]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
+ (u[2] * u[2]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
+ (u[3] * u[3]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
+ (u[4] * u[4]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
+ (u[5] * u[5]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
+ (u[6] * u[6]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
+ (u[7] * u[7]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
+ (u[8] * u[8]) / (2.f * c_sq * c_sq)
- u_sq / (2.f * c_sq));
const float t0 = (obstacles[jj*params.nx + ii] != 0) ? s0 : (s0 + params.omega * (d_equ[0] - s0));
const float t1 = (obstacles[jj*params.nx + ii] != 0) ? s3 : (s1 + params.omega * (d_equ[1] - s1));
const float t2 = (obstacles[jj*params.nx + ii] != 0) ? s4 : (s2 + params.omega * (d_equ[2] - s2));
const float t3 = (obstacles[jj*params.nx + ii] != 0) ? s1 : (s3 + params.omega * (d_equ[3] - s3));
const float t4 = (obstacles[jj*params.nx + ii] != 0) ? s2 : (s4 + params.omega * (d_equ[4] - s4));
const float t5 = (obstacles[jj*params.nx + ii] != 0) ? s7 : (s5 + params.omega * (d_equ[5] - s5));
const float t6 = (obstacles[jj*params.nx + ii] != 0) ? s8 : (s6 + params.omega * (d_equ[6] - s6));
const float t7 = (obstacles[jj*params.nx + ii] != 0) ? s5 : (s7 + params.omega * (d_equ[7] - s7));
const float t8 = (obstacles[jj*params.nx + ii] != 0) ? s6 : (s8 + params.omega * (d_equ[8] - s8));
const float local_density_v = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;
const float u_x_v = (t1 + t5 + t8 - (t3 + t6 + t7)) / local_density_v;
const float u_y_v = (t2 + t5 + t6 - (t4 + t7 + t8)) / local_density_v;
tot_u += (obstacles[jj*params.nx + ii] != 0) ? 0 : sqrtf((u_x_v * u_x_v) + (u_y_v * u_y_v));
tot_cells += (obstacles[jj*params.nx + ii] != 0) ? 0 : 1;
tmp_cells->speed_0[ii + jj*params.nx] = t0;
tmp_cells->speed_1[ii + jj*params.nx] = t1;
tmp_cells->speed_2[ii + jj*params.nx] = t2;
tmp_cells->speed_3[ii + jj*params.nx] = t3;
tmp_cells->speed_4[ii + jj*params.nx] = t4;
tmp_cells->speed_5[ii + jj*params.nx] = t5;
tmp_cells->speed_6[ii + jj*params.nx] = t6;
tmp_cells->speed_7[ii + jj*params.nx] = t7;
tmp_cells->speed_8[ii + jj*params.nx] = t8;
}
}
return tot_u / (float)tot_cells;
}
float av_velocity(const t_param params, t_soa* cells, int* obstacles) {
int    tot_cells = 0;  
float tot_u;          
tot_u = 0.f;
for (int jj = 0; jj < params.ny; jj++) {
for (int ii = 0; ii < params.nx; ii++) {
if (!obstacles[ii + jj*params.nx]) {
float local_density =  cells->speed_0[ii + jj*params.nx] +
cells->speed_1[ii + jj*params.nx] +
cells->speed_2[ii + jj*params.nx] +
cells->speed_3[ii + jj*params.nx] +
cells->speed_4[ii + jj*params.nx] +
cells->speed_5[ii + jj*params.nx] +
cells->speed_6[ii + jj*params.nx] +
cells->speed_7[ii + jj*params.nx] +
cells->speed_8[ii + jj*params.nx];
float u_x = (cells->speed_1[ii + jj*params.nx]
+  cells->speed_5[ii + jj*params.nx]
+  cells->speed_8[ii + jj*params.nx]
- (cells->speed_3[ii + jj*params.nx]
+  cells->speed_6[ii + jj*params.nx]
+  cells->speed_7[ii + jj*params.nx]))
/ local_density;
float u_y = (cells->speed_2[ii + jj*params.nx]
+  cells->speed_5[ii + jj*params.nx]
+  cells->speed_6[ii + jj*params.nx]
- (cells->speed_4[ii + jj*params.nx]
+  cells->speed_7[ii + jj*params.nx]
+  cells->speed_8[ii + jj*params.nx]))
/ local_density;
tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
tot_cells++;
}
}
}
return tot_u / (float)tot_cells;
}
int initialise(const char* paramfile, const char* obstaclefile,
t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
int** obstacles_ptr, float** av_vels_ptr) {
char   message[1024];  
FILE*   fp;            
int    xx, yy;         
int    blocked;        
int    retval;         
fp = fopen(paramfile, "r");
if (fp == NULL) {
sprintf(message, "could not open input parameter file: %s", paramfile);
die(message, __LINE__, __FILE__);
}
retval = fscanf(fp, "%d\n", &(params->nx));
if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
retval = fscanf(fp, "%d\n", &(params->ny));
if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
retval = fscanf(fp, "%d\n", &(params->maxIters));
if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
retval = fscanf(fp, "%f\n", &(params->density));
if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
retval = fscanf(fp, "%f\n", &(params->accel));
if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
retval = fscanf(fp, "%f\n", &(params->omega));
if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);
fclose(fp);
*cells_ptr = (t_soa*) malloc(sizeof(t_soa));
if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
(*cells_ptr)->speed_0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*cells_ptr)->speed_8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
*tmp_cells_ptr = (t_soa*) malloc(sizeof(t_soa));
if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
(*tmp_cells_ptr)->speed_0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
(*tmp_cells_ptr)->speed_8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
*obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));
if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
float w0 = params->density * 4.f / 9.f;
float w1 = params->density      / 9.f;
float w2 = params->density      / 36.f;
#pragma omp parallel for
for (int jj = 0; jj < params->ny; jj++) {
for (int ii = 0; ii < params->nx; ii++) {
(*cells_ptr)->speed_0[ii + jj*params->nx] = w0;
(*cells_ptr)->speed_1[ii + jj*params->nx] = w1;
(*cells_ptr)->speed_2[ii + jj*params->nx] = w1;
(*cells_ptr)->speed_3[ii + jj*params->nx] = w1;
(*cells_ptr)->speed_4[ii + jj*params->nx] = w1;
(*cells_ptr)->speed_5[ii + jj*params->nx] = w2;
(*cells_ptr)->speed_6[ii + jj*params->nx] = w2;
(*cells_ptr)->speed_7[ii + jj*params->nx] = w2;
(*cells_ptr)->speed_8[ii + jj*params->nx] = w2;
}
}
#pragma omp parallel for
for (int jj = 0; jj < params->ny; jj++) {
for (int ii = 0; ii < params->nx; ii++) {
(*obstacles_ptr)[ii + jj*params->nx] = 0;
}
}
fp = fopen(obstaclefile, "r");
if (fp == NULL) {
sprintf(message, "could not open input obstacles file: %s", obstaclefile);
die(message, __LINE__, __FILE__);
}
while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
(*obstacles_ptr)[xx + yy*params->nx] = blocked;
}
fclose(fp);
*av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
return EXIT_SUCCESS;
}
int finalise(const t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
int** obstacles_ptr, float** av_vels_ptr) {
free(*obstacles_ptr);
*obstacles_ptr = NULL;
free(*av_vels_ptr);
*av_vels_ptr = NULL;
return EXIT_SUCCESS;
}
float calc_reynolds(const t_param params, t_soa* cells, int* obstacles) {
const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}
float total_density(const t_param params, t_soa* cells) {
float total = 0.f;  
for (int jj = 0; jj < params.ny; jj++) {
for (int ii = 0; ii < params.nx; ii++) {
total += cells->speed_0[ii + jj*params.nx] +
cells->speed_1[ii + jj*params.nx] +
cells->speed_2[ii + jj*params.nx] +
cells->speed_3[ii + jj*params.nx] +
cells->speed_4[ii + jj*params.nx] +
cells->speed_5[ii + jj*params.nx] +
cells->speed_6[ii + jj*params.nx] +
cells->speed_7[ii + jj*params.nx] +
cells->speed_8[ii + jj*params.nx];
}
}
return total;
}
int write_values(const t_param params, t_soa* cells, int* obstacles, float* av_vels) {
FILE* fp;                     
const float c_sq = 1.f / 3.f; 
float local_density;         
float pressure;              
float u_x;                   
float u_y;                   
float u;                     
fp = fopen(FINALSTATEFILE, "w");
if (fp == NULL) {
die("could not open file output file", __LINE__, __FILE__);
}
for (int jj = 0; jj < params.ny; jj++)  {
for (int ii = 0; ii < params.nx; ii++) {
if (obstacles[ii + jj*params.nx]) {
u_x = u_y = u = 0.f;
pressure = params.density * c_sq;
}
else {
local_density = cells->speed_0[ii + jj*params.nx] +
cells->speed_1[ii + jj*params.nx] +
cells->speed_2[ii + jj*params.nx] +
cells->speed_3[ii + jj*params.nx] +
cells->speed_4[ii + jj*params.nx] +
cells->speed_5[ii + jj*params.nx] +
cells->speed_6[ii + jj*params.nx] +
cells->speed_7[ii + jj*params.nx] +
cells->speed_8[ii + jj*params.nx];
u_x = (cells->speed_1[ii + jj*params.nx]
+  cells->speed_5[ii + jj*params.nx]
+  cells->speed_8[ii + jj*params.nx]
- (cells->speed_3[ii + jj*params.nx]
+  cells->speed_6[ii + jj*params.nx]
+  cells->speed_7[ii + jj*params.nx]))
/ local_density;
u_y = (cells->speed_2[ii + jj*params.nx]
+  cells->speed_5[ii + jj*params.nx]
+  cells->speed_6[ii + jj*params.nx]
- (cells->speed_4[ii + jj*params.nx]
+  cells->speed_7[ii + jj*params.nx]
+  cells->speed_8[ii + jj*params.nx]))
/ local_density;
u = sqrtf((u_x * u_x) + (u_y * u_y));
pressure = local_density * c_sq;
}
fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
}
}
fclose(fp);
fp = fopen(AVVELSFILE, "w");
if (fp == NULL) {
die("could not open file output file", __LINE__, __FILE__);
}
for (int ii = 0; ii < params.maxIters; ii++) {
fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
}
fclose(fp);
return EXIT_SUCCESS;
}
void die(const char* message, const int line, const char* file) {
fprintf(stderr, "Error at line %d of file %s:\n", line, file);
fprintf(stderr, "%s\n", message);
fflush(stderr);
exit(EXIT_FAILURE);
}
void usage(const char* exe) {
fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
exit(EXIT_FAILURE);
}
