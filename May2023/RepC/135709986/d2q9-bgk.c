#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>
#include "mpi.h"
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
typedef struct
{
int    nx;            
int    ny;            
int    maxIters;      
int    reynolds_dim;  
float density;       
float accel;         
float omega;         
} t_param;
int initialise(const char* paramfile, const char* obstaclefile, t_param* params, float** speed0_ptr, float** speed1_ptr, float** speed2_ptr, float** speed3_ptr, float** speed4_ptr, float** speed5_ptr, float** speed6_ptr, float** speed7_ptr, float** speed8_ptr, float** tmp_speed0_ptr, float** tmp_speed1_ptr, float** tmp_speed2_ptr, float** tmp_speed3_ptr, float** tmp_speed4_ptr, float** tmp_speed5_ptr, float** tmp_speed6_ptr, float** tmp_speed7_ptr, float** tmp_speed8_ptr, int** obstacles_ptr, float** av_vels_ptr);
int timestepInner(const float* parameters, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, float* tmp_speed0, float* tmp_speed1, float* tmp_speed2, float* tmp_speed3, float* tmp_speed4, float* tmp_speed5, float* tmp_speed6, float* tmp_speed7, float* tmp_speed8, int* obstacles, float *reduction_buffer, int tt);
int timestepOuter(const t_param parameters, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, float* tmp_speed0, float* tmp_speed1, float* tmp_speed2, float* tmp_speed3, float* tmp_speed4, float* tmp_speed5, float* tmp_speed6, float* tmp_speed7, float* tmp_speed8, int* obstacles, float *reduction_buffer);
int accelerate_flow(const float* parameters, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles);
int write_values(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles, float* av_vels);
int finalise(const t_param* params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, float* tmp_speed0, float* tmp_speed1, float* tmp_speed2, float* tmp_speed3, float* tmp_speed4, float* tmp_speed5, float* tmp_speed6, float* tmp_speed7, float* tmp_speed8, int** obstacles_ptr, float** av_vels_ptr);
float total_density(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8);
float av_velocity(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles);
float calc_reynolds(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
int main(int argc, char* argv[]) {
char* paramfile = NULL;    
char* obstaclefile = NULL; 
t_param params;              
float* speed0 = NULL;
float* speed1 = NULL;
float* speed2 = NULL;
float* speed3 = NULL;
float* speed4 = NULL;
float* speed5 = NULL;
float* speed6 = NULL;
float* speed7 = NULL;
float* speed8 = NULL;
float* tmp_speed0 = NULL;
float* tmp_speed1 = NULL;
float* tmp_speed2 = NULL;
float* tmp_speed3 = NULL;
float* tmp_speed4 = NULL;
float* tmp_speed5 = NULL;
float* tmp_speed6 = NULL;
float* tmp_speed7 = NULL;
float* tmp_speed8 = NULL;
int* obstacles = NULL;    
float* av_vels = NULL;     
struct timeval timstr;        
struct rusage ru;             
double tic, toc;              
double usrtim;                
double systim;                
int worldSize;
int rank;
MPI_Init(NULL, NULL);
MPI_Datatype MPI_T_PARAM;
MPI_Datatype types2[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
int blocklen2[] = {1, 1, 1, 1, 1, 1, 1};
MPI_Aint disp2[] = {0, 1 * sizeof(int), 2 * sizeof(int), 3 * sizeof(int), 4 * sizeof(int), 4 * sizeof(float) + sizeof(float), 4 * sizeof(int) + 2 * sizeof(float)};
MPI_Type_create_struct(7, blocklen2, disp2, types2, &MPI_T_PARAM);
MPI_Type_commit(&MPI_T_PARAM);
MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int ndims = 1;
int dims[] = {worldSize};
int periods[] = {1};
MPI_Comm cart_world;
MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_world);
if (argc != 3)
{
usage(argv[0]);
}
else
{
paramfile = argv[1];
obstaclefile = argv[2];
}
int tot_cells = 0;
int cols_per_proc;
int *send_cnts = (int *) malloc(sizeof(int) * worldSize);
int *row_cnts = (int *) malloc(sizeof(int) * worldSize);
int *displs = (int *) malloc(sizeof(int) * worldSize);
if (rank == 0) {
initialise(paramfile, obstaclefile, &params, &speed0, &speed1, &speed2, &speed3, &speed4, &speed5, &speed6, &speed7, &speed8, &tmp_speed0, &tmp_speed1, &tmp_speed2, &tmp_speed3, &tmp_speed4, &tmp_speed5, &tmp_speed6, &tmp_speed7, &tmp_speed8, &obstacles, &av_vels);
for (int j = 0; j < params.ny; j++) {
for (int i = 0; i < params.nx; i++) {
tot_cells += !obstacles[i + j * params.nx];
}
}
int rem = params.ny % worldSize;
int sum = 0;
for (int i = 0; i < worldSize; i++) {
row_cnts[i] = params.ny / worldSize;
if (rem > 0) {
row_cnts[i]++;
rem--;
}
send_cnts[i] = params.nx * row_cnts[i];
displs[i] = sum;
sum += send_cnts[i];
}
cols_per_proc = params.nx;
}
MPI_Bcast(send_cnts, worldSize, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(row_cnts, worldSize, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(displs, worldSize, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&cols_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&params, 1, MPI_T_PARAM, 0, MPI_COMM_WORLD);
t_param sub_params;
memcpy(&sub_params, &params, sizeof(t_param));
sub_params.ny = row_cnts[rank];
sub_params.nx = cols_per_proc;
int haloOffset = sub_params.nx * (sub_params.ny + 1);
int N = sub_params.nx * (sub_params.ny + 2);
int rowCnt = sub_params.nx;
int *sub_obstacles = (int *) malloc(sizeof(int) * N);
float* sub_speed0 = (float *) malloc(sizeof(float) * N);
float* sub_speed1 = (float *) malloc(sizeof(float) * N);
float* sub_speed2 = (float *) malloc(sizeof(float) * N);
float* sub_speed3 = (float *) malloc(sizeof(float) * N);
float* sub_speed4 = (float *) malloc(sizeof(float) * N);
float* sub_speed5 = (float *) malloc(sizeof(float) * N);
float* sub_speed6 = (float *) malloc(sizeof(float) * N);
float* sub_speed7 = (float *) malloc(sizeof(float) * N);
float* sub_speed8 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed0 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed1 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed2 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed3 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed4 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed5 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed6 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed7 = (float *) malloc(sizeof(float) * N);
float* sub_tmp_speed8 = (float *) malloc(sizeof(float) * N);
float* reduction_buffer = malloc(sub_params.maxIters * sizeof(float));
float parameters[] = {
sub_params.nx,
sub_params.ny,
sub_params.omega,
sub_params.density,
sub_params.accel,
};
int maxIters = params.maxIters;
#pragma omp target enter data map(alloc: sub_speed0[0:N], sub_speed1[0:N], sub_speed2[0:N], sub_speed3[0:N], sub_speed4[0:N], sub_speed5[0:N], sub_speed6[0:N], sub_speed7[0:N], sub_speed8[0:N], sub_tmp_speed0[0:N], sub_tmp_speed1[0:N], sub_tmp_speed2[0:N], sub_tmp_speed3[0:N], sub_tmp_speed4[0:N], sub_tmp_speed5[0:N], sub_tmp_speed6[0:N], sub_tmp_speed7[0:N], sub_tmp_speed8[0:N], sub_obstacles[0:N], reduction_buffer[0:maxIters], parameters[0:5])
{}
MPI_Scatterv(obstacles, send_cnts, displs, MPI_INT, (sub_obstacles + sub_params.nx), send_cnts[rank], MPI_INT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed0, send_cnts, displs, MPI_FLOAT, (sub_speed0 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed1, send_cnts, displs, MPI_FLOAT, (sub_speed1 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed2, send_cnts, displs, MPI_FLOAT, (sub_speed2 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed3, send_cnts, displs, MPI_FLOAT, (sub_speed3 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed4, send_cnts, displs, MPI_FLOAT, (sub_speed4 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed5, send_cnts, displs, MPI_FLOAT, (sub_speed5 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed6, send_cnts, displs, MPI_FLOAT, (sub_speed6 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed7, send_cnts, displs, MPI_FLOAT, (sub_speed7 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(speed8, send_cnts, displs, MPI_FLOAT, (sub_speed8 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Request haloRequests[6];
MPI_Status haloStatuses[6];
float* sendbuf2 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* sendbuf4 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* sendbuf5 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* sendbuf6 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* sendbuf7 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* sendbuf8 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* recvbuf2 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* recvbuf4 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* recvbuf5 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* recvbuf6 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* recvbuf7 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* recvbuf8 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
float* swp_speed0;
float* swp_speed1;
float* swp_speed2;
float* swp_speed3;
float* swp_speed4;
float* swp_speed5;
float* swp_speed6;
float* swp_speed7;
float* swp_speed8;
float* swp_tmp_speed0;
float* swp_tmp_speed1;
float* swp_tmp_speed2;
float* swp_tmp_speed3;
float* swp_tmp_speed4;
float* swp_tmp_speed5;
float* swp_tmp_speed6;
float* swp_tmp_speed7;
float* swp_tmp_speed8;
memset(reduction_buffer, 0, maxIters * sizeof(float));
#pragma omp target update to(sub_speed0[0:N], sub_speed1[0:N], sub_speed2[0:N], sub_speed3[0:N], sub_speed4[0:N], sub_speed5[0:N], sub_speed6[0:N], sub_speed7[0:N], sub_speed8[0:N], sub_obstacles[0:N], parameters[0:5], reduction_buffer[0:maxIters])
{}
if (rank == 0) {
gettimeofday(&timstr, NULL);
tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
}
for (int tt = 0; tt < maxIters; tt++)
{
swp_speed0 = (tt % 2) ? sub_tmp_speed0 : sub_speed0;
swp_tmp_speed0 = (tt % 2) ? sub_speed0 : sub_tmp_speed0;
swp_speed1 = (tt % 2) ? sub_tmp_speed1 : sub_speed1;
swp_tmp_speed1 = (tt % 2) ? sub_speed1 : sub_tmp_speed1;
swp_speed2 = (tt % 2) ? sub_tmp_speed2 : sub_speed2;
swp_tmp_speed2 = (tt % 2) ? sub_speed2 : sub_tmp_speed2;
swp_speed3 = (tt % 2) ? sub_tmp_speed3 : sub_speed3;
swp_tmp_speed3 = (tt % 2) ? sub_speed3 : sub_tmp_speed3;
swp_speed4 = (tt % 2) ? sub_tmp_speed4 : sub_speed4;
swp_tmp_speed4 = (tt % 2) ? sub_speed4 : sub_tmp_speed4;
swp_speed5 = (tt % 2) ? sub_tmp_speed5 : sub_speed5;
swp_tmp_speed5 = (tt % 2) ? sub_speed5 : sub_tmp_speed5;
swp_speed6 = (tt % 2) ? sub_tmp_speed6 : sub_speed6;
swp_tmp_speed6 = (tt % 2) ? sub_speed6 : sub_tmp_speed6;
swp_speed7 = (tt % 2) ? sub_tmp_speed7 : sub_speed7;
swp_tmp_speed7 = (tt % 2) ? sub_speed7 : sub_tmp_speed7;
swp_speed8 = (tt % 2) ? sub_tmp_speed8 : sub_speed8;
swp_tmp_speed8 = (tt % 2) ? sub_speed8 : sub_tmp_speed8;
if ((sub_params.ny == 1 && (worldSize - 2) == rank) || (sub_params.ny >= 2 && (worldSize - 1) == rank)) {
accelerate_flow(parameters, swp_speed0, swp_speed1, swp_speed2, swp_speed3, swp_speed4, swp_speed5, swp_speed6, swp_speed7, swp_speed8, sub_obstacles);
}
#pragma omp target update from(swp_speed2[haloOffset-rowCnt:rowCnt], swp_speed5[haloOffset-rowCnt:rowCnt], swp_speed6[haloOffset-rowCnt:rowCnt], swp_speed4[rowCnt:rowCnt], swp_speed7[rowCnt:rowCnt], swp_speed8[rowCnt:rowCnt])
memcpy(sendbuf2 + sub_params.nx, swp_speed2 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
memcpy(sendbuf4, swp_speed4 + sub_params.nx, sizeof(float) * sub_params.nx);
memcpy(sendbuf5 + sub_params.nx, swp_speed5 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
memcpy(sendbuf6 + sub_params.nx, swp_speed6 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
memcpy(sendbuf7, swp_speed7 + sub_params.nx, sizeof(float) * sub_params.nx);
memcpy(sendbuf8, swp_speed8 + sub_params.nx, sizeof(float) * sub_params.nx);
MPI_Ineighbor_alltoall(sendbuf2, sub_params.nx, MPI_FLOAT, recvbuf2, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[0]);
MPI_Ineighbor_alltoall(sendbuf4, sub_params.nx, MPI_FLOAT, recvbuf4, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[1]);
MPI_Ineighbor_alltoall(sendbuf5, sub_params.nx, MPI_FLOAT, recvbuf5, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[2]);
MPI_Ineighbor_alltoall(sendbuf6, sub_params.nx, MPI_FLOAT, recvbuf6, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[3]);
MPI_Ineighbor_alltoall(sendbuf7, sub_params.nx, MPI_FLOAT, recvbuf7, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[4]);
MPI_Ineighbor_alltoall(sendbuf8, sub_params.nx, MPI_FLOAT, recvbuf8, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[5]);
MPI_Waitall(6, haloRequests, haloStatuses);
if (worldSize > 2) {
memcpy(swp_speed2, recvbuf2, sizeof(float) * sub_params.nx);
memcpy(swp_speed4 + ((sub_params.ny + 1) * sub_params.nx), recvbuf4 + sub_params.nx, sizeof(float) * sub_params.nx);
memcpy(swp_speed5, recvbuf5, sizeof(float) * sub_params.nx);
memcpy(swp_speed6, recvbuf6, sizeof(float) * sub_params.nx);
memcpy(swp_speed7 + ((sub_params.ny + 1) * sub_params.nx), recvbuf7 + sub_params.nx, sizeof(float) * sub_params.nx);
memcpy(swp_speed8 + ((sub_params.ny + 1) * sub_params.nx), recvbuf8 + sub_params.nx, sizeof(float) * sub_params.nx);
} else {
memcpy(swp_speed2, recvbuf2 + sub_params.nx, sizeof(float) * sub_params.nx);
memcpy(swp_speed4 + ((sub_params.ny + 1) * sub_params.nx), recvbuf4, sizeof(float) * sub_params.nx);
memcpy(swp_speed5, recvbuf5 + sub_params.nx, sizeof(float) * sub_params.nx);
memcpy(swp_speed6, recvbuf6 + sub_params.nx, sizeof(float) * sub_params.nx);
memcpy(swp_speed7 + ((sub_params.ny + 1) * sub_params.nx), recvbuf7, sizeof(float) * sub_params.nx);
memcpy(swp_speed8 + ((sub_params.ny + 1) * sub_params.nx), recvbuf8, sizeof(float) * sub_params.nx);
}
#pragma omp target update to(swp_speed2[0:rowCnt], swp_speed5[0:rowCnt], swp_speed6[0:rowCnt], swp_speed4[haloOffset:rowCnt], swp_speed7[haloOffset:rowCnt], swp_speed8[haloOffset:rowCnt])
timestepInner(parameters, swp_speed0, swp_speed1, swp_speed2, swp_speed3, swp_speed4, swp_speed5, swp_speed6, swp_speed7, swp_speed8, swp_tmp_speed0, swp_tmp_speed1, swp_tmp_speed2, swp_tmp_speed3, swp_tmp_speed4, swp_tmp_speed5, swp_tmp_speed6, swp_tmp_speed7, swp_tmp_speed8, sub_obstacles, reduction_buffer, tt);
#ifdef DEBUG
if (rank == 0) {
printf("==timestep: %d==\n", tt);
printf("av velocity: %.12E\n", av_vels[tt]);
printf("tot density: %.12E\n", total_density(sub_params, sub_speed0, sub_speed1, sub_speed2, sub_speed3, sub_speed4, sub_speed5, sub_speed6, sub_speed7, sub_speed8));
}
#endif
}
if (rank == 0) {
gettimeofday(&timstr, NULL);
toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
getrusage(RUSAGE_SELF, &ru);
timstr = ru.ru_utime;
usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
timstr = ru.ru_stime;
systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
}
#pragma omp target update from(sub_speed0[0:N], sub_speed1[0:N], sub_speed2[0:N], sub_speed3[0:N], sub_speed4[0:N], sub_speed5[0:N], sub_speed6[0:N], sub_speed7[0:N], sub_speed8[0:N], reduction_buffer[0:maxIters])
{}
for (int tt = 0; tt < maxIters; ++tt) {
float global_tot_vel;
MPI_Reduce(&reduction_buffer[tt], &global_tot_vel, 1, MPI_FLOAT, MPI_SUM, 0, cart_world);
if (rank == 0) {
av_vels[tt] = global_tot_vel / (float) tot_cells;
}
}
MPI_Gatherv((sub_obstacles + sub_params.nx), send_cnts[rank], MPI_INT, obstacles, send_cnts, displs, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed0 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed0, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed1 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed1, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed2 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed2, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed3 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed3, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed4 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed4, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed5 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed5, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed6 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed6, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed7 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed7, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv((sub_speed8 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed8, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
#pragma omp target exit data map(release: sub_speed0[:N], sub_speed1[:N], sub_speed2[:N], sub_speed3[:N], sub_speed4[:N], sub_speed5[:N], sub_speed6[:N], sub_speed7[:N], sub_speed8[:N], sub_tmp_speed0[:N], sub_tmp_speed1[:N], sub_tmp_speed2[:N], sub_tmp_speed3[:N], sub_tmp_speed4[:N], sub_tmp_speed5[:N], sub_tmp_speed6[:N], sub_tmp_speed7[:N], sub_tmp_speed8[:N], sub_obstacles[:N], reduction_buffer[0:maxIters], parameters[0:5])
{}
MPI_Finalize();
if (rank == 0) {
printf("==done==\n");
printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, obstacles));
printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
write_values(params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, obstacles, av_vels);
finalise(&params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, tmp_speed0, tmp_speed1, tmp_speed2, tmp_speed3, tmp_speed4, tmp_speed5, tmp_speed6, tmp_speed7, tmp_speed8, &obstacles, &av_vels);
}
return EXIT_SUCCESS;
}
int timestepInner(const float* params, float* __restrict__ speed0, float* __restrict__ speed1, float* __restrict__ speed2, float* __restrict__ speed3, float* __restrict__ speed4, float* __restrict__ speed5, float* __restrict__ speed6, float* __restrict__ speed7, float* __restrict__ speed8, float* __restrict__ tmp_speed0, float* __restrict__ tmp_speed1, float* __restrict__ tmp_speed2, float* __restrict__ tmp_speed3, float* __restrict__ tmp_speed4, float* __restrict__ tmp_speed5, float* __restrict__ tmp_speed6, float* __restrict__ tmp_speed7, float* __restrict__ tmp_speed8, int* __restrict__ obstacles, float* __restrict__ reduction_buffer, int tt) {
const float c_sq = 1.f / 3.f; 
const float w0 = 4.f / 9.f;  
const float w1 = 1.f / 9.f;  
const float w2 = 1.f / 36.f; 
const float c_2_sq_sq = 2.f * c_sq * c_sq;
float tmpSpeed0, tmpSpeed1, tmpSpeed2, tmpSpeed3, tmpSpeed4, tmpSpeed5, tmpSpeed6, tmpSpeed7, tmpSpeed8;
int y_n, x_e, y_s, x_w;
float local_density, u_x, u_y, u_sq;
int ny = (int) params[1], nx = (int) params[0];
float omega = params[2];
#pragma omp target teams distribute parallel for reduction(+:reduction_buffer[tt]) collapse(2) private(y_n, x_e, y_s, x_w, local_density, u_x, u_y, u_sq, tmpSpeed0, tmpSpeed1, tmpSpeed2, tmpSpeed3, tmpSpeed4, tmpSpeed5, tmpSpeed6, tmpSpeed7, tmpSpeed8) shared(ny, nx, omega, c_sq, w0, w1, w2, c_2_sq_sq)
for (int jj = 1; jj < ny + 1; ++jj) {
for (int ii = 0; ii < nx; ++ii) {
y_n = jj + 1;
x_e = (ii + 1) % nx;
y_s = jj - 1;
x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
tmpSpeed0 = speed0[ii + jj*nx]; 
tmpSpeed1 = speed1[x_w + jj*nx]; 
tmpSpeed2 = speed2[ii + y_s*nx]; 
tmpSpeed3 = speed3[x_e + jj*nx]; 
tmpSpeed4 = speed4[ii + y_n*nx]; 
tmpSpeed5 = speed5[x_w + y_s*nx]; 
tmpSpeed6 = speed6[x_e + y_s*nx]; 
tmpSpeed7 = speed7[x_e + y_n*nx]; 
tmpSpeed8 = speed8[x_w + y_n*nx]; 
if (!obstacles[ii + jj*nx]) {
local_density = 0.f;
local_density += tmpSpeed0;
local_density += tmpSpeed1;
local_density += tmpSpeed2;
local_density += tmpSpeed3;
local_density += tmpSpeed4;
local_density += tmpSpeed5;
local_density += tmpSpeed6;
local_density += tmpSpeed7;
local_density += tmpSpeed8;
u_x = (tmpSpeed1
+ tmpSpeed5
+ tmpSpeed8
- (tmpSpeed3
+ tmpSpeed6
+ tmpSpeed7))
/ local_density;
u_y = (tmpSpeed2
+ tmpSpeed5
+ tmpSpeed6
- (tmpSpeed4
+ tmpSpeed7
+ tmpSpeed8))
/ local_density;
u_sq = u_x * u_x + u_y * u_y;
reduction_buffer[tt] += sqrtf(u_sq);
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
+ (u[1] * u[1]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
+ (u[2] * u[2]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
+ (u[3] * u[3]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
+ (u[4] * u[4]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
+ (u[5] * u[5]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
+ (u[6] * u[6]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
+ (u[7] * u[7]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
+ (u[8] * u[8]) / c_2_sq_sq
- u_sq / (2.f * c_sq));
tmp_speed0[ii + jj*nx] = tmpSpeed0
+ omega
* (d_equ[0] - tmpSpeed0);
tmp_speed1[ii + jj*nx] = tmpSpeed1
+ omega
* (d_equ[1] - tmpSpeed1);
tmp_speed2[ii + jj*nx] = tmpSpeed2
+ omega
* (d_equ[2] - tmpSpeed2);
tmp_speed3[ii + jj*nx] = tmpSpeed3
+ omega
* (d_equ[3] - tmpSpeed3);
tmp_speed4[ii + jj*nx] = tmpSpeed4
+ omega
* (d_equ[4] - tmpSpeed4);
tmp_speed5[ii + jj*nx] = tmpSpeed5
+ omega
* (d_equ[5] - tmpSpeed5);
tmp_speed6[ii + jj*nx] = tmpSpeed6
+ omega
* (d_equ[6] - tmpSpeed6);
tmp_speed7[ii + jj*nx] = tmpSpeed7
+ omega
* (d_equ[7] - tmpSpeed7);
tmp_speed8[ii + jj*nx] = tmpSpeed8
+ omega
* (d_equ[8] - tmpSpeed8);
} else {
tmp_speed0[ii + jj*nx] = tmpSpeed0;
tmp_speed1[ii + jj*nx] = tmpSpeed3;
tmp_speed2[ii + jj*nx] = tmpSpeed4;
tmp_speed3[ii + jj*nx] = tmpSpeed1;
tmp_speed4[ii + jj*nx] = tmpSpeed2;
tmp_speed5[ii + jj*nx] = tmpSpeed7;
tmp_speed6[ii + jj*nx] = tmpSpeed8;
tmp_speed7[ii + jj*nx] = tmpSpeed5;
tmp_speed8[ii + jj*nx] = tmpSpeed6;
}
}
}
return EXIT_SUCCESS;
}
int timestepOuter(const t_param params, float* __restrict__ speed0, float* __restrict__ speed1, float* __restrict__ speed2, float* __restrict__ speed3, float* __restrict__ speed4, float* __restrict__ speed5, float* __restrict__ speed6, float* __restrict__ speed7, float* __restrict__ speed8, float* __restrict__ tmp_speed0, float* __restrict__ tmp_speed1, float* __restrict__ tmp_speed2, float* __restrict__ tmp_speed3, float* __restrict__ tmp_speed4, float* __restrict__ tmp_speed5, float* __restrict__ tmp_speed6, float* __restrict__ tmp_speed7, float* __restrict__ tmp_speed8, int* __restrict__ obstacles, float* __restrict__ reduction_buffer) {
const float c_sq = 1.f / 3.f; 
const float w0 = 4.f / 9.f;  
const float w1 = 1.f / 9.f;  
const float w2 = 1.f / 36.f; 
float tmpSpeed0, tmpSpeed1, tmpSpeed2, tmpSpeed3, tmpSpeed4, tmpSpeed5, tmpSpeed6, tmpSpeed7, tmpSpeed8;
int y_n, x_e, y_s, x_w;
float local_density, u_x, u_y, u_sq;
int ny = params.ny, nx = params.nx;
float omega = params.omega;
int jj;
#pragma omp target teams distribute parallel for reduction(+:reduction_buffer[0]) collapse(2) schedule(static, 1) private(y_n, x_e, y_s, x_w, local_density, u_x, u_y, u_sq, tmpSpeed0, tmpSpeed1, tmpSpeed2, tmpSpeed3, tmpSpeed4, tmpSpeed5, tmpSpeed6, tmpSpeed7, tmpSpeed8)
for (int c = 0; c < 2; c++) {
for (int ii = 0; ii < nx; ii++) {
jj = c == 0 ? 1 : ny;
int y_n = jj + 1;
int x_e = (ii + 1) % nx;
int y_s = jj - 1;
int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
float tmpSpeed0 = speed0[ii + jj*nx]; 
float tmpSpeed1 = speed1[x_w + jj*nx]; 
float tmpSpeed2 = speed2[ii + y_s*nx]; 
float tmpSpeed3 = speed3[x_e + jj*nx]; 
float tmpSpeed4 = speed4[ii + y_n*nx]; 
float tmpSpeed5 = speed5[x_w + y_s*nx]; 
float tmpSpeed6 = speed6[x_e + y_s*nx]; 
float tmpSpeed7 = speed7[x_e + y_n*nx]; 
float tmpSpeed8 = speed8[x_w + y_n*nx]; 
float local_density = 0.f;
local_density += tmpSpeed0;
local_density += tmpSpeed1;
local_density += tmpSpeed2;
local_density += tmpSpeed3;
local_density += tmpSpeed4;
local_density += tmpSpeed5;
local_density += tmpSpeed6;
local_density += tmpSpeed7;
local_density += tmpSpeed8;
float u_x = (tmpSpeed1
+ tmpSpeed5
+ tmpSpeed8
- (tmpSpeed3
+ tmpSpeed6
+ tmpSpeed7))
/ local_density;
float u_y = (tmpSpeed2
+ tmpSpeed5
+ tmpSpeed6
- (tmpSpeed4
+ tmpSpeed7
+ tmpSpeed8))
/ local_density;
u_sq = u_x * u_x + u_y * u_y;
reduction_buffer[0] += !obstacles[ii + jj*nx] ? sqrtf(u_sq) : 0;
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
tmp_speed0[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed0
+ omega
* (d_equ[0] - tmpSpeed0) : tmpSpeed0;
tmp_speed1[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed1
+ omega
* (d_equ[1] - tmpSpeed1) : tmpSpeed3;
tmp_speed2[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed2
+ omega
* (d_equ[2] - tmpSpeed2) : tmpSpeed4;
tmp_speed3[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed3
+ omega
* (d_equ[3] - tmpSpeed3) : tmpSpeed1;
tmp_speed4[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed4
+ omega
* (d_equ[4] - tmpSpeed4) : tmpSpeed2;
tmp_speed5[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed5
+ omega
* (d_equ[5] - tmpSpeed5) : tmpSpeed7;
tmp_speed6[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed6
+ omega
* (d_equ[6] - tmpSpeed6) : tmpSpeed8;
tmp_speed7[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed7
+ omega
* (d_equ[7] - tmpSpeed7) : tmpSpeed5;
tmp_speed8[ii + jj*nx] = !obstacles[ii + jj*nx] ? tmpSpeed8
+ omega
* (d_equ[8] - tmpSpeed8) : tmpSpeed6;
}
}
return EXIT_SUCCESS;
}
int accelerate_flow(const float* params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles)
{
float w1 = params[3] * params[4] / 9.f;
float w2 = params[3] * params[4] / 36.f;
int jj = (int) params[1] >= 2 ? (int) params[1] - 1 : (int) params[1];
int nx = (int) params[0];
#pragma omp target teams distribute parallel for shared(w1, w2, nx, jj)
for (int ii = 0; ii < nx; ii++)
{
if (!obstacles[ii + jj*nx]
&& (speed3[ii + jj*nx] - w1) > 0.f
&& (speed6[ii + jj*nx] - w2) > 0.f
&& (speed7[ii + jj*nx] - w2) > 0.f)
{
speed1[ii + jj*nx] += w1;
speed5[ii + jj*nx] += w2;
speed8[ii + jj*nx] += w2;
speed3[ii + jj*nx] -= w1;
speed6[ii + jj*nx] -= w2;
speed7[ii + jj*nx] -= w2;
}
}
return EXIT_SUCCESS;
}
float av_velocity(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles)
{
int    tot_cells = 0;  
float tot_u;          
tot_u = 0.f;
for (int jj = 0; jj < params.ny; jj++)
{
for (int ii = 0; ii < params.nx; ii++)
{
if (!obstacles[ii + jj*params.nx])
{
float local_density = 0.f;
local_density += speed0[ii + jj*params.nx];
local_density += speed1[ii + jj*params.nx];
local_density += speed2[ii + jj*params.nx];
local_density += speed3[ii + jj*params.nx];
local_density += speed4[ii + jj*params.nx];
local_density += speed5[ii + jj*params.nx];
local_density += speed6[ii + jj*params.nx];
local_density += speed7[ii + jj*params.nx];
local_density += speed8[ii + jj*params.nx];
float u_x = (speed1[ii + jj*params.nx]
+ speed5[ii + jj*params.nx]
+ speed8[ii + jj*params.nx]
- (speed3[ii + jj*params.nx]
+ speed6[ii + jj*params.nx]
+ speed7[ii + jj*params.nx]))
/ local_density;
float u_y = (speed2[ii + jj*params.nx]
+ speed5[ii + jj*params.nx]
+ speed6[ii + jj*params.nx]
- (speed4[ii + jj*params.nx]
+ speed7[ii + jj*params.nx]
+ speed8[ii + jj*params.nx]))
/ local_density;
tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
++tot_cells;
}
}
}
return tot_u / (float)tot_cells;
}
int initialise(const char* paramfile, const char* obstaclefile,
t_param* params, float** speed0_ptr, float** speed1_ptr, float** speed2_ptr, float** speed3_ptr, float** speed4_ptr, float** speed5_ptr, float** speed6_ptr, float** speed7_ptr, float** speed8_ptr, float** tmp_speed0_ptr, float** tmp_speed1_ptr, float** tmp_speed2_ptr, float** tmp_speed3_ptr, float** tmp_speed4_ptr, float** tmp_speed5_ptr, float** tmp_speed6_ptr, float** tmp_speed7_ptr, float** tmp_speed8_ptr, int** obstacles_ptr, float** av_vels_ptr)
{
char   message[1024];  
FILE*   fp;            
int    xx, yy;         
int    blocked;        
int    retval;         
fp = fopen(paramfile, "r");
if (fp == NULL)
{
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
*speed0_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed1_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed2_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed3_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed4_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed5_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed6_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed7_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*speed8_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
if (*speed0_ptr == NULL) die("cannot allocate memory for speed0", __LINE__, __FILE__);
if (*speed1_ptr == NULL) die("cannot allocate memory for speed1", __LINE__, __FILE__);
if (*speed2_ptr == NULL) die("cannot allocate memory for speed2", __LINE__, __FILE__);
if (*speed3_ptr == NULL) die("cannot allocate memory for speed3", __LINE__, __FILE__);
if (*speed4_ptr == NULL) die("cannot allocate memory for speed4", __LINE__, __FILE__);
if (*speed5_ptr == NULL) die("cannot allocate memory for speed5", __LINE__, __FILE__);
if (*speed6_ptr == NULL) die("cannot allocate memory for speed6", __LINE__, __FILE__);
if (*speed7_ptr == NULL) die("cannot allocate memory for speed7", __LINE__, __FILE__);
if (*speed8_ptr == NULL) die("cannot allocate memory for speed8", __LINE__, __FILE__);
*tmp_speed0_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed1_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed2_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed3_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed4_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed5_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed6_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed7_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
*tmp_speed8_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
if (*tmp_speed0_ptr == NULL) die("cannot allocate memory for tmp_speed0", __LINE__, __FILE__);
if (*tmp_speed1_ptr == NULL) die("cannot allocate memory for tmp_speed1", __LINE__, __FILE__);
if (*tmp_speed2_ptr == NULL) die("cannot allocate memory for tmp_speed2", __LINE__, __FILE__);
if (*tmp_speed3_ptr == NULL) die("cannot allocate memory for tmp_speed3", __LINE__, __FILE__);
if (*tmp_speed4_ptr == NULL) die("cannot allocate memory for tmp_speed4", __LINE__, __FILE__);
if (*tmp_speed5_ptr == NULL) die("cannot allocate memory for tmp_speed5", __LINE__, __FILE__);
if (*tmp_speed6_ptr == NULL) die("cannot allocate memory for tmp_speed6", __LINE__, __FILE__);
if (*tmp_speed7_ptr == NULL) die("cannot allocate memory for tmp_speed7", __LINE__, __FILE__);
if (*tmp_speed8_ptr == NULL) die("cannot allocate memory for tmp_speed8", __LINE__, __FILE__);
*obstacles_ptr = (int *) malloc(sizeof(int) * (params->ny * params->nx));
if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
float w0 = params->density * 4.f / 9.f;
float w1 = params->density      / 9.f;
float w2 = params->density      / 36.f;
for (int jj = 0; jj < params->ny; jj++)
{
for (int ii = 0; ii < params->nx; ii++)
{
(*speed0_ptr)[ii + jj*params->nx] = w0;
(*speed1_ptr)[ii + jj*params->nx] = w1;
(*speed2_ptr)[ii + jj*params->nx] = w1;
(*speed3_ptr)[ii + jj*params->nx] = w1;
(*speed4_ptr)[ii + jj*params->nx] = w1;
(*speed5_ptr)[ii + jj*params->nx] = w2;
(*speed6_ptr)[ii + jj*params->nx] = w2;
(*speed7_ptr)[ii + jj*params->nx] = w2;
(*speed8_ptr)[ii + jj*params->nx] = w2;
}
}
for (int jj = 0; jj < params->ny; jj++)
{
for (int ii = 0; ii < params->nx; ii++)
{
(*obstacles_ptr)[ii + jj*params->nx] = 0;
}
}
fp = fopen(obstaclefile, "r");
if (fp == NULL)
{
sprintf(message, "could not open input obstacles file: %s", obstaclefile);
die(message, __LINE__, __FILE__);
}
while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
{
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
int finalise(const t_param* params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, float* tmp_speed0, float* tmp_speed1, float* tmp_speed2, float* tmp_speed3, float* tmp_speed4, float* tmp_speed5, float* tmp_speed6, float* tmp_speed7, float* tmp_speed8, int** obstacles_ptr, float** av_vels_ptr)
{
free(speed0);
free(speed1);
free(speed2);
free(speed3);
free(speed4);
free(speed5);
free(speed6);
free(speed7);
free(speed8);
free(tmp_speed0);
free(tmp_speed1);
free(tmp_speed2);
free(tmp_speed3);
free(tmp_speed4);
free(tmp_speed5);
free(tmp_speed6);
free(tmp_speed7);
free(tmp_speed8);
free(*obstacles_ptr);
*obstacles_ptr = NULL;
free(*av_vels_ptr);
*av_vels_ptr = NULL;
return EXIT_SUCCESS;
}
float calc_reynolds(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles)
{
const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
return av_velocity(params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, obstacles) * params.reynolds_dim / viscosity;
}
float total_density(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8)
{
float total = 0.f;  
for (int jj = 0; jj < params.ny; jj++)
{
for (int ii = 0; ii < params.nx; ii++)
{
total += speed0[ii + jj*params.nx];
total += speed1[ii + jj*params.nx];
total += speed2[ii + jj*params.nx];
total += speed3[ii + jj*params.nx];
total += speed4[ii + jj*params.nx];
total += speed5[ii + jj*params.nx];
total += speed6[ii + jj*params.nx];
total += speed7[ii + jj*params.nx];
total += speed8[ii + jj*params.nx];
}
}
return total;
}
int write_values(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles, float* av_vels)
{
FILE* fp;                     
const float c_sq = 1.f / 3.f; 
float local_density;         
float pressure;              
float u_x;                   
float u_y;                   
float u;                     
fp = fopen(FINALSTATEFILE, "w");
if (fp == NULL)
{
die("could not open file output file", __LINE__, __FILE__);
}
for (int jj = 0; jj < params.ny; jj++)
{
for (int ii = 0; ii < params.nx; ii++)
{
if (obstacles[ii + jj*params.nx])
{
u_x = u_y = u = 0.f;
pressure = params.density * c_sq;
}
else
{
local_density = 0.f;
local_density += speed0[ii + jj*params.nx];
local_density += speed1[ii + jj*params.nx];
local_density += speed2[ii + jj*params.nx];
local_density += speed3[ii + jj*params.nx];
local_density += speed4[ii + jj*params.nx];
local_density += speed5[ii + jj*params.nx];
local_density += speed6[ii + jj*params.nx];
local_density += speed7[ii + jj*params.nx];
local_density += speed8[ii + jj*params.nx];
float u_x = (speed1[ii + jj*params.nx]
+ speed5[ii + jj*params.nx]
+ speed8[ii + jj*params.nx]
- (speed3[ii + jj*params.nx]
+ speed6[ii + jj*params.nx]
+ speed7[ii + jj*params.nx]))
/ local_density;
float u_y = (speed2[ii + jj*params.nx]
+ speed5[ii + jj*params.nx]
+ speed6[ii + jj*params.nx]
- (speed4[ii + jj*params.nx]
+ speed7[ii + jj*params.nx]
+ speed8[ii + jj*params.nx]))
/ local_density;
u = sqrtf((u_x * u_x) + (u_y * u_y));
pressure = local_density * c_sq;
}
fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
}
}
fclose(fp);
fp = fopen(AVVELSFILE, "w");
if (fp == NULL)
{
die("could not open file output file", __LINE__, __FILE__);
}
for (int ii = 0; ii < params.maxIters; ii++)
{
fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
}
fclose(fp);
return EXIT_SUCCESS;
}
void die(const char* message, const int line, const char* file)
{
fprintf(stderr, "Error at line %d of file %s:\n", line, file);
fprintf(stderr, "%s\n", message);
fflush(stderr);
exit(EXIT_FAILURE);
}
void usage(const char* exe)
{
fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
exit(EXIT_FAILURE);
}
