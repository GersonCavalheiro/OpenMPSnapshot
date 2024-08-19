#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/utsname.h>
#include "multMV.h"
#include "parser.h"
#include "sgpu.h"
#include "createSpMat.h"
#include "utils/ts.h"
#define DIM_CART 2
#define SHIFT 1
#define RESERVE SHIFT*2
#define IND(x, y, z) ((x) + (y)*NX + (z)*NX*(NYr + RESERVE))
#define IF_BOUND_F ((x > SHIFT - 1) && (x < NX - SHIFT) && (y > SHIFT - 1) && (y < NYr + SHIFT) && (z > SHIFT - 1) && (z < NZr + SHIFT))
#define COPY_BOUND_F(F) \
if (x == SHIFT - 1) {                                       \
F[IND(x, y, z)] = F[IND(x + 1, y, z)]; \
}\
if (x == NX - SHIFT) {\
F[IND(x, y, z)] = F[IND(x - 1, y, z)];\
}\
if (y == SHIFT - 1) {\
F[IND(x, y, z)] = F[IND(x, y + 1, z)];\
}\
if (y == NYr + SHIFT) {\
F[IND(x, y, z)] = F[IND(x, y - 1, z)];\
}\
if (z == SHIFT - 1) {\
F[IND(x, y, z)] = F[IND(x, y, z + 1)];\
}\
if (z == NZr + SHIFT) {\
F[IND(x, y, z)] = F[IND(x, y, z - 1)];\
}
int main(int argc, char **argv) {
int sizeP, rankP;
size_t sizeTime;
double t1 = 0.0, t0 = 0.0;
SpMatrix A, B, C;
Setting setting;
TYPE coeffs[3][4];
TYPE dt;
MPI_Status status[4];
int blockYP = 0, blockZP = 0;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &sizeP);
MPI_Comm_rank(MPI_COMM_WORLD, &rankP);
const size_t len = 80;
char nameHost[len];
gethostname(nameHost, len);
printf("rank - %d name host - %s\n", rankP, nameHost);
TYPE *u = NULL, *u_chunk = NULL, *un_chunk = NULL;
int NX, NY, NZ, NYr, NZr;
#if ENABLE_PARALLEL
omp_set_num_threads(atoi(argv[1]));
if (rankP == ROOT) printf("PARALLEL VERSION! Number of threads - %u\n", omp_get_max_threads());
#endif
if (rankP == ROOT) {
int error = readSetting(INPUT_EULER_SETTING_PATH, &setting);
if (error != OK) return error;
NX = (setting.NX + RESERVE);
NY = (setting.NY + RESERVE);
NZ = (setting.NZ + RESERVE);
int dim = NX * NY * NZ;
u = (TYPE *) calloc(dim, sizeof(TYPE));
error = readFunction(INPUT_EULER_FUNCTION_PATH, u, NX, NY, NZ, SHIFT);
if (error != OK) return error;
sizeTime = (size_t) ((setting.TFINISH - setting.TSTART) / setting.dt);
printf("TimeSize -\t%lu\n", sizeTime);
TYPE dx = ABS(setting.XSTART - setting.XEND) / setting.NX;
TYPE dy = ABS(setting.YSTART - setting.YEND) / setting.NY;
TYPE dz = ABS(setting.ZSTART - setting.ZEND) / setting.NZ;
dt = setting.dt;
coeffs[0][1] = setting.SIGMA / (dx * dx);
coeffs[0][2] = setting.SIGMA / (dy * dy);
coeffs[0][3] = setting.SIGMA / (dz * dz);
coeffs[0][0] = -2.0 * (coeffs[0][1] + coeffs[0][2] + coeffs[0][3]);
}
MPI_Bcast(&sizeTime, 1, MPI_UNSIGNED_LONG, ROOT, MPI_COMM_WORLD);
MPI_Bcast(coeffs, 4*3, MPI_TYPE, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&dt, 1, MPI_TYPE, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&NX, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&NY, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&NZ, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
get_blocks(&blockYP, &blockZP, sizeP);
if (rankP == ROOT) printf("blockY %d blockZ %d\n", blockYP, blockZP);
NYr = (NY - RESERVE) / blockYP;
NZr = (NZ - RESERVE) / blockZP;
MPI_Comm gridComm;
int dims[DIM_CART], periods[DIM_CART];
int gridCoords[DIM_CART];
dims[0] = blockZP;
dims[1] = blockYP;
periods[0] = 0;
periods[1] = 0;
int reorder = 0;
MPI_Cart_create(MPI_COMM_WORLD, DIM_CART, dims, periods, reorder, &gridComm);
MPI_Cart_coords(gridComm, rankP, DIM_CART, gridCoords);
int dimChunk = NX * (NYr + RESERVE) * (NZr + RESERVE);
u_chunk = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
un_chunk = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
scatter_by_block(u, u_chunk, NX, NY, NYr, NZr, gridComm, RESERVE);
int rank_left, rank_right, rank_down, rank_top;
MPI_Cart_shift(gridComm, 1, -1, &rank_left, &rank_right);
MPI_Cart_shift(gridComm, 0, 1, &rank_down, &rank_top);
coeffs[1][1] = dt * coeffs[0][1] * 0.5;
coeffs[1][2] = dt * coeffs[0][2] * 0.5;
coeffs[1][3] = dt * coeffs[0][3] * 0.5;
coeffs[1][0] = 1.0 - 2.0 * (coeffs[1][1] + coeffs[2][1] + coeffs[3][1]);
coeffs[2][1] = coeffs[1][1] * 2.0;
coeffs[2][2] = coeffs[1][2] * 2.0;
coeffs[2][3] = coeffs[1][3] * 2.0;
coeffs[2][0] = 1.0 - 2.0 * (coeffs[2][1] + coeffs[2][2] + coeffs[2][3]);
TYPE *k1 = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
TYPE *k2 = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
TYPE *k3 = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
TYPE *k4 = (TYPE *) aligned_alloc(ALIGNMENT, sizeof(TYPE) * dimChunk);
TYPE h = dt / 6.0;
TYPE *tmp;
for (int z = 0; z < NZr + RESERVE; z++) {
for (int y = 0; y < NYr + RESERVE; y++) {
for (int x = 0; x < NX; x++) {
COPY_BOUND_F(u_chunk);
}
}
}
MPI_Datatype planeXY;
MPI_Type_vector(NZr + RESERVE, NX * SHIFT, NX * (NYr + RESERVE), MPI_TYPE, &planeXY);
MPI_Type_commit(&planeXY);
MPI_Datatype planeXZ;
MPI_Type_contiguous(NX * SHIFT * (NYr + RESERVE), MPI_TYPE, &planeXZ);
MPI_Type_commit(&planeXZ);
if (rankP == ROOT) {
printf("START!\n");
t0 = WTIME();
}
for (int t = 1; t <= sizeTime; t++) {
MPI_Sendrecv(&u_chunk[IND(0, NYr, 0)], 1, planeXY, rank_left, 0,
&u_chunk[IND(0, 0, 0)], 1, planeXY, rank_right, 0,
gridComm, &status[0]);
MPI_Sendrecv(&u_chunk[IND(0, SHIFT, 0)], 1, planeXY, rank_right, 1,
&u_chunk[IND(0, NYr + SHIFT, 0)], 1, planeXY, rank_left, 1,
gridComm, &status[1]);
MPI_Sendrecv(&u_chunk[IND(0, 0, SHIFT)], 1, planeXZ, rank_down, 2,
&u_chunk[IND(0, 0, NZr + SHIFT)], 1, planeXZ, rank_top, 2,
gridComm, &status[2]);
MPI_Sendrecv(&u_chunk[IND(0, 0, NZr)], 1, planeXZ, rank_top, 3,
&u_chunk[IND(0, 0, 0)], 1, planeXZ, rank_down, 3,
gridComm, &status[3]);
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = SHIFT; z < NZr + RESERVE - SHIFT; z++) {
for (int y = SHIFT; y < NYr + RESERVE - SHIFT; y++) {
for (int x = SHIFT; x < NX - SHIFT; x++) {
k1[IND(x, y, z)] = coeffs[0][1]*u_chunk[IND(x - 1, y, z)] + coeffs[0][2]*u_chunk[IND(x, y - 1, z)] +
coeffs[0][3]*u_chunk[IND(x, y, z - 1)] + coeffs[0][1]*u_chunk[IND(x + 1, y, z)] + coeffs[0][2]*u_chunk[IND(x, y + 1, z)] +
coeffs[0][3]*u_chunk[IND(x, y, z + 1)] + coeffs[0][0]*u_chunk[IND(x, y, z)];
}
}
}
copyingBorders(k1, NX, NYr + RESERVE, NZr + RESERVE);
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = SHIFT; z < NZr + RESERVE - SHIFT; z++) {
for (int y = SHIFT; y < NYr + RESERVE - SHIFT; y++) {
for (int x = SHIFT; x < NX - SHIFT; x++) {
if (IF_BOUND_F) {
k2[IND(x, y, z)] = coeffs[1][1]*k1[IND(x - 1, y, z)] + coeffs[1][2]*k1[IND(x, y - 1, z)] +
coeffs[1][3]*k1[IND(x, y, z - 1)] + coeffs[1][1]*k1[IND(x + 1, y, z)] + coeffs[1][2]*k1[IND(x, y + 1, z)] +
coeffs[1][3]*k1[IND(x, y, z + 1)] + coeffs[1][0]*k1[IND(x, y, z)];
}
}
}
}
copyingBorders(k2, NX, NYr + RESERVE, NZr + RESERVE);
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = SHIFT; z < NZr + RESERVE - SHIFT; z++) {
for (int y = SHIFT; y < NYr + RESERVE - SHIFT; y++) {
for (int x = SHIFT; x < NX - SHIFT; x++) {
if (IF_BOUND_F) {
k3[IND(x, y, z)] = coeffs[1][1]*k2[IND(x - 1, y, z)] + coeffs[1][2]*k2[IND(x, y - 1, z)] +
coeffs[1][3]*k2[IND(x, y, z - 1)] + coeffs[1][1]*k2[IND(x + 1, y, z)] + coeffs[1][2]*u_chunk[IND(x, y + 1, z)] +
coeffs[1][3]*k2[IND(x, y, z + 1)] + coeffs[1][0]*k2[IND(x, y, z)];
}
}
}
}
copyingBorders(k3, NX, NYr + RESERVE, NZr + RESERVE);
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = SHIFT; z < NZr + RESERVE - SHIFT; z++) {
for (int y = SHIFT; y < NYr + RESERVE - SHIFT; y++) {
for (int x = SHIFT; x < NX - SHIFT; x++) {
if (IF_BOUND_F) {
k4[IND(x, y, z)] = coeffs[2][1]*k3[IND(x - 1, y, z)] + coeffs[2][2]*k3[IND(x, y - 1, z)] +
coeffs[2][3]*k3[IND(x, y, z - 1)] + coeffs[2][1]*k3[IND(x + 1, y, z)] + coeffs[2][2]*u_chunk[IND(x, y + 1, z)] +
coeffs[2][3]*k3[IND(x, y, z + 1)] + coeffs[2][0]*k3[IND(x, y, z)];
}
}
}
}
copyingBorders(k4, NX, NYr + RESERVE, NZr + RESERVE);
sumV(un_chunk, u_chunk, k1, k2, k3, k4, dimChunk, h);
tmp = u_chunk;
u_chunk = un_chunk;
un_chunk = tmp;
}
if (rankP == ROOT) {
printf("FINISH!\n");
t1 = WTIME();
}
gather_by_block(u, u_chunk, NX, NY, NYr, NZr, RESERVE, gridComm);
if (rankP == ROOT) {
double diffTime = t1 - t0;
printf("Time -\t%.3lf\n", diffTime);
free(u);
}
MPI_Type_free(&planeXY);
MPI_Type_free(&planeXZ);
free(un_chunk);
free(k1);
free(k2);
free(k3);
free(k4);
MPI_Finalize();
return 0;
}
