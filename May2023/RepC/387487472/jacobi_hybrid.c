#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#define CONVERGE_CHECK_TRUE 1
#ifndef SCHEDULE_TYPE
#define SCHEDULE_TYPE static
#endif
#ifdef ENABLE_COLLAPSE
#define COLLAPSE_S collapse(2)
#else
#define COLLAPSE_S
#endif
double checkSolution(double xStart, double yStart,
int maxXCount, int maxYCount,
double *u,
double deltaX, double deltaY,
double alpha)
{
#define U(XX, YY) u[(YY)*maxXCount + (XX)]
double fX;
double localError, error = 0.0;
#pragma omp parallel for private(localError, fX) reduction(+: error)
for (int y = 1; y < (maxYCount - 1); y++)
{
for (int x = 1; x < (maxXCount - 1); x++)
{
fX = xStart + (x - 1) * deltaX;
localError = U(x, y) - (1.0 - fX * fX) * (1.0 - (yStart + (y - 1) * deltaY) * (yStart + (y - 1) * deltaY));
error += localError * localError;
}
}
return error;
}
int main(int argc, char **argv)
{
int n, m, mits;
double alpha, tol, relax;
double maxAcceptableError;
double error, local_error0, local_error1, local_error2, local_error3, local_error4;
double *u, *u_old, *u_all, *tmp;
int allocCount, localAlloc;
int iterationCount, maxIterationCount;
double t1, t2;
int myRank, numProcs;
int prevRank, nextRank;
double error_sum;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
MPI_Comm comm_cart;
int periodic[2] = { 0, 0 };
int dims[2] = { 0, 0 };
MPI_Dims_create(numProcs, 2, dims);
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &comm_cart);
MPI_Comm_rank(comm_cart, &myRank);
if (myRank == 0)
{
scanf("%d,%d", &n, &m);
scanf("%lf", &alpha);
scanf("%lf", &relax);
scanf("%lf", &tol);
scanf("%d", &mits);
printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);
}
MPI_Bcast(&n, 1, MPI_INT, 0, comm_cart);
MPI_Bcast(&m, 1, MPI_INT, 0, comm_cart);
MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, comm_cart);
MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, comm_cart);
MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, comm_cart);
MPI_Bcast(&mits, 1, MPI_INT, 0, comm_cart);
int local_n = n / dims[0];
int local_m = m / dims[1];
MPI_Datatype row_t;
MPI_Type_contiguous(local_m, MPI_DOUBLE, &row_t);
MPI_Type_commit(&row_t);
MPI_Datatype col_t;
MPI_Type_vector(local_n, 1, local_m+2, MPI_DOUBLE, &col_t);
MPI_Type_commit(&col_t);
u =     (double *) calloc(((local_n + 2) * (local_m + 2)), sizeof(double));
u_old = (double *) calloc(((local_n + 2) * (local_m + 2)), sizeof(double));
if (u == NULL || u_old == NULL) {
printf("Not enough memory for two %ix%i matrices\n", local_n + 2, local_m + 2); exit(1);
}
int tag = 666;  
int south, north, east, west;
MPI_Cart_shift(comm_cart, 0, 1, &north, &south);  
MPI_Cart_shift(comm_cart, 1, 1, &west, &east);    
MPI_Request req_send_u_old[4], req_recv_u_old[4], req_send_u[4], req_recv_u[4];
MPI_Request *req_send = req_send_u_old, *req_recv = req_recv_u_old;
MPI_Recv_init(&u_old[1], 1, row_t, north, tag, comm_cart, &req_recv_u_old[0]);
MPI_Recv_init(&u_old[(local_m+2)*(local_n+1)+1], 1, row_t, south, tag, comm_cart, &req_recv_u_old[1]);
MPI_Recv_init(&u_old[local_m + 2], 1, col_t, west, tag, comm_cart, &req_recv_u_old[2]);
MPI_Recv_init(&u_old[local_m + 2 + local_m + 1], 1, col_t, east, tag, comm_cart, &req_recv_u_old[3]);
MPI_Send_init(&u_old[local_m + 2 + 1], 1, row_t, north, tag, comm_cart, &req_send_u_old[0]);
MPI_Send_init(&u_old[(local_m+2) * (local_n) + 1], 1, row_t, south, tag, comm_cart, &req_send_u_old[1]);
MPI_Send_init(&u_old[local_m + 2 + 1], 1, col_t, west, tag, comm_cart, &req_send_u_old[2]);
MPI_Send_init(&u_old[local_m + 2 + local_m], 1, col_t, east, tag, comm_cart, &req_send_u_old[3]);
MPI_Recv_init(&u[1], 1, row_t, north, tag, comm_cart, &req_recv_u[0]);
MPI_Recv_init(&u[(local_m+2)*(local_n+1)+1], 1, row_t, south, tag, comm_cart, &req_recv_u[1]);
MPI_Recv_init(&u[local_m + 2], 1, col_t, west, tag, comm_cart, &req_recv_u[2]);
MPI_Recv_init(&u[local_m + 2 + local_m + 1], 1, col_t, east, tag, comm_cart, &req_recv_u[3]);
MPI_Send_init(&u[local_m + 2 + 1], 1, row_t, north, tag, comm_cart, &req_send_u[0]);
MPI_Send_init(&u[(local_m+2) * (local_n) + 1], 1, row_t, south, tag, comm_cart, &req_send_u[1]);
MPI_Send_init(&u[local_m + 2 + 1], 1, col_t, west, tag, comm_cart, &req_send_u[2]);
MPI_Send_init(&u[local_m + 2 + local_m], 1, col_t, east, tag, comm_cart, &req_send_u[3]);
maxIterationCount = mits;
maxAcceptableError = tol;
double xLeft = -1.0, xRight = 1.0;
double yBottom = -1.0, yUp = 1.0;
double deltaX = (xRight - xLeft) / (n - 1);
double deltaY = (yUp - yBottom) / (m - 1);
iterationCount = 0;
error = HUGE_VAL;
clock_t start, diff;
MPI_Barrier(comm_cart);
start = clock();
t1 = MPI_Wtime();
int maxXCount = local_n + 2;
int maxYCount = local_m + 2;
int indices[maxYCount];
double cx = 1.0 / (deltaX * deltaX);
double cy = 1.0 / (deltaY * deltaY);
double cc = -2.0 * (cx + cy) - alpha;
double div_cc = 1.0 / cc;
double cx_cc = 1.0 / (deltaX * deltaX) * div_cc;
double cy_cc = 1.0 / (deltaY * deltaY) * div_cc;
double c1 = (2.0 + alpha) * div_cc;
double c2 = 2.0 * div_cc;
double fX_sq[local_n], fY_sq[local_m], updateVal, fX_dot_fY_sq;
#pragma omp parallel shared(fX_sq, fY_sq, error, iterationCount, local_error0, local_error1,local_error2, local_error3, local_error4) private(fX_dot_fY_sq, updateVal) 
{
#pragma omp for schedule(static)
for (int x = 0; x < local_n; x++) {
fX_sq[x] = (xLeft + x * deltaX) * (xLeft + x * deltaX);
}
#pragma omp for schedule(static)
for (int y = 0; y < local_m; y++) {
fY_sq[y] = (yBottom + y * deltaY) * (yBottom + y * deltaY);
}
#pragma omp for schedule(static)
for (int i = 0; i < maxYCount; i++) {
indices[i] = i * maxXCount;
}
#ifdef CONVERGE_CHECK_TRUE
while (iterationCount < maxIterationCount && error > maxAcceptableError)
#else
while (iterationCount < maxIterationCount)
#endif
{
#pragma omp barrier
#pragma omp master
{
iterationCount++;
error = local_error0 = local_error1 = local_error2 = local_error3 = local_error4 = 0.0;
MPI_Startall(4, req_recv);
MPI_Startall(4, req_send);
}
#pragma omp barrier
#pragma omp for reduction(+:local_error0) schedule(static) COLLAPSE_S
for (int y = 2; y < (maxYCount - 2); y++)
{
for (int x = 2; x < (maxXCount - 2); x++)
{
fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[y - 1];
updateVal = (u_old[indices[y] + x - 1] + u_old[indices[y] + x + 1]) * cx_cc +
(u_old[indices[y-1] + x] + u_old[indices[y+1] + x]) * cy_cc +
u_old[indices[y] + x] +
c1 * (1.0 - fX_sq[x - 1] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);
u[indices[y] + x] = u_old[indices[y] + x] - relax * updateVal;
local_error0 += updateVal * updateVal;
}
}
#pragma omp master
MPI_Waitall(4, req_recv, MPI_STATUSES_IGNORE);
#pragma omp barrier
#pragma omp for reduction(+:local_error1) schedule(static)
for (int x = 1; x < (maxXCount - 1); x++)
{
fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[0];
updateVal = (u_old[indices[1] + x - 1] + u_old[indices[1] + x + 1]) * cx_cc +
(u_old[indices[0] + x] + u_old[indices[2] + x]) * cy_cc +
u_old[indices[1] + x] +
c1 * (1.0 - fX_sq[x - 1] - fY_sq[0] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);
u[indices[1] + x] = u_old[indices[1] + x] - relax * updateVal;
local_error1 += updateVal * updateVal;
}
#pragma omp for reduction(+:local_error2) schedule(static)
for (int x = 1; x < (maxXCount - 1); x++)
{
fX_dot_fY_sq = fX_sq[x - 1] * fY_sq[maxYCount - 3];
updateVal = (u_old[indices[maxYCount - 2] + x - 1] + u_old[indices[maxYCount - 2] + x + 1]) * cx_cc +
(u_old[indices[maxYCount - 3] + x] + u_old[indices[maxYCount - 1] + x]) * cy_cc +
u_old[indices[maxYCount - 2] + x] +
c1 * (1.0 - fX_sq[x - 1] - fY_sq[maxYCount - 3] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);
u[indices[maxYCount - 2] + x] = u_old[indices[maxYCount - 2] + x] - relax * updateVal;
local_error2 += updateVal * updateVal;
}
#pragma omp for reduction(+:local_error3) schedule(static)
for (int y = 1; y < (maxYCount - 1); y++)
{
fX_dot_fY_sq = fX_sq[0] * fY_sq[y - 1];
updateVal = (u_old[indices[y] + 0] + u_old[indices[y] + 2]) * cx_cc +
(u_old[indices[y-1] + 1] + u_old[indices[y+1] + 1]) * cy_cc +
u_old[indices[y] + 1] +
c1 * (1.0 - fX_sq[0] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);
u[indices[y] + 1] = u_old[indices[y] + 1] - relax * updateVal;
local_error3 += updateVal * updateVal;
}
#pragma omp for reduction(+:local_error4) schedule(static)
for (int y = 1; y < (maxYCount - 1); y++)
{
fX_dot_fY_sq = fX_sq[maxXCount - 3] * fY_sq[y - 1];
updateVal = (u_old[indices[y] + maxXCount - 3] + u_old[indices[y] + maxXCount - 1]) * cx_cc +
(u_old[indices[y-1] + maxXCount-2] + u_old[indices[y+1] + maxXCount-2]) * cy_cc +
u_old[indices[y] + maxXCount-2] +
c1 * (1.0 - fX_sq[maxXCount - 3] - fY_sq[y - 1] + fX_dot_fY_sq) - c2 * (fX_dot_fY_sq - 1.0);
u[indices[y] + maxXCount-2] = u_old[indices[y] + maxXCount-2] - relax * updateVal;
local_error4 += updateVal * updateVal;
}
#pragma omp master
{
#ifdef CONVERGE_CHECK_TRUE
error = local_error0 + local_error1 + local_error2 + local_error3 + local_error4;
MPI_Allreduce(&error, &error_sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
error = sqrt(error_sum) / (n * m);
#endif
MPI_Waitall(4, req_send, MPI_STATUSES_IGNORE);
tmp = u_old;
u_old = u;
u = tmp;
req_recv = (req_recv == req_recv_u_old) ? req_recv_u : req_recv_u_old;
req_send = (req_send == req_send_u_old) ? req_send_u : req_send_u_old;
}
#ifdef CONVERGE_CHECK_TRUE
#pragma omp barrier
#endif
}
}  
#ifndef CONVERGE_CHECK_TRUE
error = local_error0 + local_error1 + local_error2 + local_error3 + local_error4;
#endif
t2 = MPI_Wtime();
diff = clock() - start;
int msec = diff * 1000 / CLOCKS_PER_SEC;
int max_msec;
MPI_Reduce(&msec, &max_msec, 1, MPI_INT, MPI_MAX, 0, comm_cart);
double final_time, local_final_time = t2 - t1;
MPI_Reduce(&local_final_time, &final_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
#ifndef CONVERGE_CHECK_TRUE
MPI_Reduce(&error, &error_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
error = sqrt(error_sum) / (n * m);
#endif
if (myRank == 0) {
printf("Iterations=%3d\nElapsed MPI Wall time is %f\n", iterationCount, final_time);
printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
printf("Residual %g\n", error);
}
free(u);
double absolute_error, local_absolute_error;
local_absolute_error = checkSolution(xLeft, yBottom, local_n + 2, local_m + 2, u_old, deltaX, deltaY, alpha);
MPI_Reduce(&local_absolute_error, &absolute_error, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
absolute_error = sqrt(absolute_error) / (n * m);
if (myRank == 0)
printf("The error of the iterative solution is %g\n", absolute_error);
MPI_Datatype block_t;
MPI_Type_vector(local_m, local_n, local_n+2, MPI_DOUBLE, &block_t);
MPI_Type_commit(&block_t);
if (myRank == 0)
{
u_all = (double *)calloc(numProcs *local_m*(local_n+2), sizeof(double)); 
if (u_all == NULL)
{
printf("Not enough memory for u_all matrix\n"); 
exit(1);
}
}
MPI_Gather(&u_old[local_n+3], 1, block_t, u_all, 1, block_t, 0, comm_cart);
free(u_old);
if (myRank == 0)
{
#define INDEX(y) (y*(n+2))
double *u_final = calloc((n+2)*(m+2), sizeof(double));
int index = 0;
for (int x = 1; x < n+1; x+=local_n) {  
for (int y = 1; y < m+1; y++) {     
memcpy(&u_final[INDEX(y)+x], &u_all[index], local_n*sizeof(double));
index += local_n; 
}
}
free(u_all);
double absoluteError = checkSolution(xLeft, yBottom, n + 2, m + 2, u_final, deltaX, deltaY, alpha);
absoluteError = sqrt(absoluteError) / (n * m);
printf("The error of the gathered solution is %g\n", absoluteError);
free(u_final);
}
MPI_Finalize();
return 0;
}