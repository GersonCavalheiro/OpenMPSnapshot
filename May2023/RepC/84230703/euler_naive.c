#include <stdio.h>
#include <stdlib.h>
#define _BSD_SOURCE
#include <unistd.h>
#include <string.h>
#include "multMV.h"
#include "createSpMat.h"
#include "parser.h"
#include "sgpu.h"
#define DIM_CART 2
#define SHIFT 1
#define RESERVE SHIFT*2
#define IND(x, y, z) ((x) + (y)*NX + (z)*NX*(NYr + RESERVE))
int main(int argc, char **argv) {
int sizeP = 1;
int rankP = ROOT;
size_t sizeTime;
double t0 = 0.0, t1 = 0.0;
int blockYP = 0, blockZP = 0;
MPI_Status status[4];
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &sizeP);
MPI_Comm_rank(MPI_COMM_WORLD, &rankP);
if (rankP == ROOT) printf("MPI RUN. %d size processes\n", sizeP);
MPI_Comm gridComm;
char nameHost[40];
gethostname(nameHost, 40);
printf("Rank - %d name host - %s\n", rankP, nameHost);
int dim = 0;
TYPE coeffs[4];
TYPE *u = NULL, *u_chunk = NULL, *un_chunk;
int NX, NY, NZ;
#if ENABLE_PARALLEL
omp_set_num_threads(atoi(argv[1]));
if (rankP == ROOT) printf("PARALLEL VERSION! Number of threads - %u\n", omp_get_max_threads());
#endif
if (rankP == ROOT) {
int error;
Setting setting;
error = readSetting(INPUT_EULER_SETTING_PATH, &setting);
if (error != OK) return error;
dim = (setting.NX + RESERVE) * (setting.NY + RESERVE) * (setting.NZ + RESERVE);
u = (TYPE *) calloc((size_t) dim, sizeof(TYPE));
error = readFunction(INPUT_EULER_FUNCTION_PATH, u, setting.NX + RESERVE,
setting.NY + RESERVE, setting.NZ + RESERVE, SHIFT);
if (error != OK) return error;
sizeTime = (size_t) ((setting.TFINISH - setting.TSTART) / setting.dt);
printf("TimeSize -\t%lu\n", sizeTime);
TYPE dx = ABS(setting.XSTART - setting.XEND) / setting.NX;
TYPE dy = ABS(setting.YSTART - setting.YEND) / setting.NY;
TYPE dz = ABS(setting.ZSTART - setting.ZEND) / setting.NZ;
coeffs[0] = 1.f - 2.f * setting.dt * setting.SIGMA * (1.f / (dx * dx) + 1.f / (dy * dy) + 1.f / (dz * dz));
coeffs[1] = setting.dt * setting.SIGMA / (dx * dx);
coeffs[2] = setting.dt * setting.SIGMA / (dy * dy);
coeffs[3] = setting.dt * setting.SIGMA / (dz * dz);
NX = setting.NX + RESERVE;
NY = setting.NY + RESERVE;
NZ = setting.NZ + RESERVE;
}
MPI_Bcast(&sizeTime, 1, MPI_UNSIGNED_LONG, ROOT, MPI_COMM_WORLD);
MPI_Bcast(coeffs, 4, MPI_TYPE, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&NX, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&NY, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&NZ, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
get_blocks(&blockYP, &blockZP, sizeP);
const int NYr = (NY - RESERVE) / blockYP;
const int NZr = (NZ - RESERVE) / blockZP;
const int dimChunk = NX * (NYr + RESERVE) * (NZr + RESERVE);
u_chunk = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
un_chunk = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
int rank_left, rank_right, rank_down, rank_top;
const int dims[] = {blockZP, blockYP};
const int periods[] = {0, 0};
int reorder = 0;
MPI_Cart_create(MPI_COMM_WORLD, DIM_CART, dims, periods, reorder, &gridComm);
scatter_by_block(u, u_chunk, NX, NY, NYr, NZr, gridComm, RESERVE);
MPI_Cart_shift(gridComm, 1, -1, &rank_left, &rank_right);
MPI_Cart_shift(gridComm, 0, 1, &rank_down, &rank_top);
MPI_Datatype planeXY;
MPI_Type_vector(NZr + RESERVE, NX, NX * (NYr + RESERVE), MPI_TYPE, &planeXY);
MPI_Type_commit(&planeXY);
MPI_Datatype planeXZ;
MPI_Type_contiguous(NX * (NYr + RESERVE), MPI_TYPE, &planeXZ);
MPI_Type_commit(&planeXZ);
copyingBorders(u_chunk, NX, NYr + RESERVE, NZr + RESERVE);
if (rankP == ROOT) {
printf("START!\n");
t0 = WTIME();
}
#if FPGA_RUN || CPU_CL_RUN || GPU_CL_RUN
naive_formula(un_chunk, u_chunk, coeffs, NX, NYr + RESERVE, NZr + RESERVE, sizeTime);
TYPE *tmp = u_chunk;
u_chunk = un_chunk;
un_chunk = tmp;
#else
for (int t = 1; t <= sizeTime; t++) {
MPI_Sendrecv(&u_chunk[IND(0, NYr, 0)], 1, planeXY, rank_left, 0,
&u_chunk[IND(0, 0, 0)], 1, planeXY, rank_right, 0, gridComm, &status[0]);
MPI_Sendrecv(&u_chunk[IND(0, 1, 0)], 1, planeXY, rank_right, 1,
&u_chunk[IND(0, NYr + 1, 0)], 1, planeXY, rank_left, 1, gridComm, &status[1]);
MPI_Sendrecv(&u_chunk[IND(0, 0, 1)], 1, planeXZ, rank_down, 2,
&u_chunk[IND(0, 0, NZr + 1)], 1, planeXZ, rank_top, 2, gridComm, &status[2]);
MPI_Sendrecv(&u_chunk[IND(0, 0, NZr)], 1, planeXZ, rank_top, 3,
&u_chunk[IND(0, 0, 0)], 1, planeXZ, rank_down, 3, gridComm, &status[3]);
copyingBorders(u_chunk, NX, NYr + RESERVE, NZr + RESERVE);
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = SHIFT; z < NZr + RESERVE - SHIFT; z++) {
for (int y = SHIFT; y < NYr + RESERVE - SHIFT; y++) {
for (int x = 1; x < NX - 1; x++) {
un_chunk[IND(x,y,z)] =
coeffs[3]*u_chunk[IND(x, y, z - 1)] +
coeffs[2]*u_chunk[IND(x, y - 1, z)] +
coeffs[1]*u_chunk[IND(x - 1, y, z)] +
coeffs[0]*u_chunk[IND(x, y, z)]     +
coeffs[1]*u_chunk[IND(x + 1, y, z)] +
coeffs[2]*u_chunk[IND(x, y + 1, z)] +
coeffs[3]*u_chunk[IND(x, y, z + 1)];
} 
} 
} 
TYPE * tmp = u_chunk;
u_chunk = un_chunk;
un_chunk = tmp;
}
#endif
if (rankP == ROOT) {
printf("FINISH!\n\n");
t1 = WTIME();
}
gather_by_block(u, u_chunk, NX, NY, NYr, NZr, RESERVE, gridComm);
if (rankP == ROOT) {
double diffTime = t1 - t0;
printf("Time -\t%.3lf\n", diffTime);
printf("DONE!!!\n\n");
free(u);
}
free(un_chunk);
free(u_chunk);
MPI_Type_free(&planeXY);
MPI_Type_free(&planeXZ);
MPI_Finalize();
return 0;
}
