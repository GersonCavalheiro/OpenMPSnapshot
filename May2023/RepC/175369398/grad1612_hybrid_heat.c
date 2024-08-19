#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>

#define NXPROB 10                      
#define NYPROB 10                      
#define STEPS 100                     
#define MASTER 0                       

#define REORGANISATION 1               
#define GRIDX 1
#define GRIDY 1

#define CONVERGENCE 1                  
#define INTERVAL 20                    
#define SENSITIVITY 0.1                

#define CX 0.1                         
#define CY 0.1

#define DEBUG  0                       

#define NUMTHREADS 4

float readfloat(FILE *);

enum coordinates {SOUTH = 0, EAST, NORTH, WEST};

int main(void) {
float **u[2];
FILE *fs, *fd;
int i, j, k, iz, *xs, *ys, comm_sz, my_rank, neighBor[4];
MPI_File fh;
MPI_Comm comm2d;
MPI_Datatype column, row, filetype, memtype;
MPI_Request recvRequest[2][4], sendRequest[2][4];

double start_time, end_time, local_elapsed_time, elapsed_time;
#if CONVERGENCE
int tobreak = 0;
float locdiff, totdiff;
#endif


MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


int xcell = NXPROB / GRIDX;
int ycell = NYPROB / GRIDY;


int block_x = xcell + 2;
int block_y = ycell + 2;

if (my_rank == MASTER) {
if (comm_sz != GRIDX * GRIDY) {
printf("ERROR: the number of tasks must be equal to %d.\nQuiting...\n", GRIDX*GRIDY);
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}
else if (NXPROB % GRIDX || NYPROB % GRIDY) {
printf("ERROR: (%d/%d) or (%d/%d) is not an integer\nQuiting...\n", NXPROB, GRIDX, NYPROB, GRIDY);
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}
else {
printf("Starting with %d processes and %d threads\nProblem size:%dx%d\nEach process will take: %dx%d\nAmount of iterations: %d\n", comm_sz, NUMTHREADS, NXPROB, NYPROB, xcell, ycell, STEPS);
#if CONVERGENCE
printf("Check for convergence every %d iterations\n", INTERVAL);
#endif
}
}


int periods[2] = {0,0};
int dims[2] = {GRIDY, GRIDX};
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, REORGANISATION, &comm2d);


MPI_Cart_shift(comm2d, 0, 1, &neighBor[WEST], &neighBor[EAST]);

MPI_Cart_shift(comm2d, 1, 1, &neighBor[NORTH], &neighBor[SOUTH]);



if ((u[0] = malloc(block_x * sizeof(*u[0]))) == NULL) {
perror ("u[0] malloc failed");
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}
if ((u[1] = malloc(block_x * sizeof(*u[1]))) == NULL) {
perror ("u[1] malloc failed");
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}

if ((u[0][0] = malloc(block_x * block_y * sizeof(**u[0]))) == NULL) {
perror ("u[0][0] malloc failed");
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}
if ((u[1][0] = malloc(block_x * block_y * sizeof(**u[1]))) == NULL) {
perror ("u[1][0] malloc failed");
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}

for (i = 1; i < block_x; i++) {

u[0][i] = u[0][0] + i * block_y;
u[1][i] = u[1][0] + i * block_y;
}


if ((xs = malloc(comm_sz * sizeof(int))) == NULL) {
perror ("xs malloc failed");
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}
if ((ys = malloc(comm_sz * sizeof(int))) == NULL) {
perror ("ys malloc failed");
MPI_Abort(MPI_COMM_WORLD, 1);
exit(1);
}

if (my_rank == MASTER) {

#pragma omp parallel for num_threads(NUMTHREADS) schedule (static,1) default(none) private(i) shared(ys)
for (i = 0; i < GRIDX; i++) ys[i] = 0;

#pragma omp parallel for num_threads(NUMTHREADS) collapse(2) schedule (static,1) default(none) private(i,j) shared(ys, ycell)
for (i = 1; i < GRIDY; i++)
for (j = 0; j < GRIDX; j++)
ys[i * GRIDX + j] = ys[(i - 1) * GRIDX + j] + ycell;

#pragma omp parallel for num_threads(NUMTHREADS) schedule (static,1) default(none) private(i) shared(xs)
for (i = 0; i < GRIDY; i++) xs[i * GRIDX] = 0;

#pragma omp parallel for num_threads(NUMTHREADS) collapse(2) schedule (static,1) default(none) private(i,j) shared(xs, xcell)       
for (i = 1; i <= GRIDY; i++)
for (j = 1; j < GRIDX; j++)
xs[(i - 1) * GRIDX + j] = xs[(i - 1) * GRIDX + (j - 1)] + xcell;
}



MPI_Type_contiguous(ycell, MPI_FLOAT, &row);
MPI_Type_commit(&row);

MPI_Type_vector(xcell, 1, block_y, MPI_FLOAT, &column);
MPI_Type_commit(&column);

MPI_Bcast(xs, comm_sz, MPI_INT, 0, comm2d);
MPI_Bcast(ys, comm_sz, MPI_INT, 0, comm2d);


#pragma omp parallel for num_threads(NUMTHREADS) schedule (static,1) default(none) private(j) shared(u, block_x, block_y) 
for (j=0; j<block_y; j++) {
u[0][0][j]=0;
u[0][block_x-1][j]=0;
u[1][0][j]=0;
u[1][block_x-1][j]=0;
}
#pragma omp parallel for num_threads(NUMTHREADS) schedule (static,1) default(none) private(i) shared(u, block_x, block_y) 
for (i=0; i<block_x; i++) {
u[0][i][0]=0;
u[0][i][block_y-1]=0;
u[1][i][0]=0;
u[1][i][block_y-1]=0;
}
#pragma omp parallel for num_threads(NUMTHREADS) collapse(2) schedule (static,1) default(none) private(i,j) shared(u, xs, ys, xcell, ycell, my_rank) 
for (i = 1; i < xcell+1; i++) {
for (j = 1; j < ycell+1; j++) {
u[0][i][j] = (i-1+xs[my_rank]) * (NXPROB - i - xs[my_rank]) * (j-1+ys[my_rank]) * (NYPROB - j - ys[my_rank]);
u[1][i][j] = 0;
}
}

#if DEBUG
int len;
char processor[MPI_MAX_PROCESSOR_NAME];
MPI_Get_processor_name(processor, &len);
printf("I am %d and my neighbors are North=%d, South=%d, East=%d, West=%d (Running on %s)\n", my_rank, neighBor[NORTH], neighBor[SOUTH], neighBor[EAST], neighBor[WEST], processor);
#endif


MPI_File_open(comm2d, "initial_binary.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
int gsizes[2] = {NXPROB, NYPROB};
int lsizes[2] = {xcell, ycell};
int start_indices[2] = {xs[my_rank], ys[my_rank]};
MPI_Type_create_subarray(2, gsizes, lsizes, start_indices,MPI_ORDER_C, MPI_FLOAT, &filetype);
MPI_Type_commit(&filetype);
MPI_File_set_view(fh, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
int memsizes[2] = {block_x, block_y};
start_indices[0] = start_indices[1] = 1;
MPI_Type_create_subarray(2, memsizes, lsizes, start_indices,MPI_ORDER_C, MPI_FLOAT, &memtype);
MPI_Type_commit(&memtype);
MPI_File_write_all(fh, &(u[0][0][0]), 1, memtype, MPI_STATUS_IGNORE);
MPI_File_close(&fh);
if (my_rank == MASTER) {
printf ("Writing initial.dat ...\n");
fs = fopen("initial_binary.dat","rb");
fd = fopen("initial.dat","w");
for (i = 0; i < NXPROB; i++) {
for (j=0;j< NYPROB;j++){
fprintf(fd, "%6.1f ", readfloat(fs));        
} 
fprintf(fd, "\n");
}
fclose(fs);
fclose(fd);
}

MPI_Barrier(comm2d);

start_time = MPI_Wtime();


MPI_Send_init(&u[0][xcell][1], 1, row, neighBor[SOUTH], 1, comm2d, &sendRequest[0][SOUTH]);     
MPI_Send_init(&u[1][xcell][1], 1, row, neighBor[SOUTH], 1, comm2d, &sendRequest[1][SOUTH]);
MPI_Send_init(&u[0][1][1], 1, row, neighBor[NORTH], 2, comm2d, &sendRequest[0][NORTH]);         
MPI_Send_init(&u[1][1][1], 1, row, neighBor[NORTH], 2, comm2d, &sendRequest[1][NORTH]);
MPI_Send_init(&u[0][1][ycell], 1, column, neighBor[EAST], 3, comm2d, &sendRequest[0][EAST]);    
MPI_Send_init(&u[1][1][ycell], 1, column, neighBor[EAST], 3, comm2d, &sendRequest[1][EAST]);
MPI_Send_init(&u[0][1][1], 1, column, neighBor[WEST], 4, comm2d, &sendRequest[0][WEST]);        
MPI_Send_init(&u[1][1][1], 1, column, neighBor[WEST], 4, comm2d, &sendRequest[1][WEST]);


MPI_Recv_init(&u[0][0][1], 1, row, neighBor[NORTH], 1, comm2d, &recvRequest[0][NORTH]);         
MPI_Recv_init(&u[1][0][1], 1, row, neighBor[NORTH], 1, comm2d, &recvRequest[1][NORTH]);
MPI_Recv_init(&u[0][xcell+1][1], 1, row, neighBor[SOUTH], 2, comm2d, &recvRequest[0][SOUTH]);   
MPI_Recv_init(&u[1][xcell+1][1], 1, row, neighBor[SOUTH], 2, comm2d, &recvRequest[1][SOUTH]);
MPI_Recv_init(&u[0][1][0], 1, column, neighBor[WEST], 3, comm2d, &recvRequest[0][WEST]);        
MPI_Recv_init(&u[1][1][0], 1, column, neighBor[WEST], 3, comm2d, &recvRequest[1][WEST]);
MPI_Recv_init(&u[0][1][ycell+1], 1, column, neighBor[EAST], 4, comm2d, &recvRequest[0][EAST]);  
MPI_Recv_init(&u[1][1][ycell+1], 1, column, neighBor[EAST], 4, comm2d, &recvRequest[1][EAST]);

iz = 0;
int count = 0;
#pragma omp parallel num_threads(NUMTHREADS) private(k)
for (k = 0; k < STEPS; k++) {
#pragma omp master
{

MPI_Startall(4, sendRequest[iz]);

MPI_Startall(4, recvRequest[iz]);
}

#pragma omp barrier 


#pragma omp parallel for schedule(static,1) collapse(2) default(none) private(i,j) shared (u, iz, xcell, ycell)
for (i = 2; i < xcell; i++)
for (j = 2; j < ycell; j++)
u[1-iz][i][j] = u[iz][i][j] + CX*(u[iz][i+1][j] + u[iz][i-1][j] - 2.0*u[iz][i][j]) + CY*(u[iz][i][j+1] + u[iz][i][j-1] - 2.0*u[iz][i][j]);

#pragma omp master
MPI_Waitall(4, recvRequest[iz], MPI_STATUSES_IGNORE);  

#pragma omp barrier                                





#pragma omp parallel for schedule(static,1) default(none) private(j) shared (u, iz, xcell, ycell)
for (j=1; j<ycell+1; j++) {
u[1-iz][1][j] = u[iz][1][j] + CX*(u[iz][2][j] + u[iz][0][j] - 2.0*u[iz][1][j]) + CY*(u[iz][1][j+1] + u[iz][1][j-1] - 2.0*u[iz][1][j]);
u[1-iz][xcell][j] = u[iz][xcell][j] + CX*(u[iz][xcell+1][j] + u[iz][xcell-1][j] - 2.0*u[iz][xcell][j]) + CY*(u[iz][xcell][j+1] + u[iz][xcell][j-1] - 2.0*u[iz][xcell][j]);
}


#pragma omp parallel for schedule(static,1) default(none) private(j) shared (u, iz, xcell, ycell)
for (j=2; j<xcell; j++) {
u[1-iz][j][1] = u[iz][j][1] + CX*(u[iz][j+1][1] + u[iz][j-1][1] - 2.0*u[iz][j][1]) + CY*(u[iz][j][2] + u[iz][j][0] - 2.0*u[iz][j][1]);
u[1-iz][j][ycell] = u[iz][j][ycell] + CX*(u[iz][j+1][ycell] + u[iz][j-1][ycell] - 2.0*u[iz][j][ycell]) + CY*(u[iz][j][ycell+1] + u[iz][j][ycell-1] - 2.0*u[iz][j][ycell]);
}


#if CONVERGENCE
if (i % INTERVAL == 0) {
locdiff = 0.0;
#pragma omp parallel for schedule(static,1) collapse(2) default(none) private(i,j) shared (u, iz, xcell, ycell) reduction(+:locdiff)
for (i = 1; i < xcell+1; i++)
for (j = 1; j < ycell+1; j++)
locdiff += (u[iz][i][j] - u[1-iz][i][j])*(u[iz][i][j] - u[1-iz][i][j]); 
#pragma omp master
{
MPI_Allreduce(&locdiff, &totdiff, 1, MPI_FLOAT, MPI_SUM, comm2d);
if (totdiff < SENSITIVITY) tobreak=1;
}
if (tobreak) break;
}
#endif

#pragma omp master
{
iz = 1-iz; 
MPI_Waitall(4, sendRequest[iz], MPI_STATUSES_IGNORE); 
count ++;
}
}

end_time = MPI_Wtime();

local_elapsed_time = end_time - start_time;
MPI_Reduce(&local_elapsed_time, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, MASTER, comm2d);


MPI_File_open(comm2d, "final_binary.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
MPI_File_write_all(fh, &(u[iz][0][0]), 1, memtype, MPI_STATUS_IGNORE);
MPI_File_close(&fh);
if (my_rank == MASTER) {
printf("Exiting after %d iterations\nElapsed time: %e sec\nWriting final.dat ...\n", count, elapsed_time);
fs = fopen("final_binary.dat","rb");
fd = fopen("final.dat","w");
for (i = 0; i < NXPROB; i++) {
for (j=0;j< NYPROB;j++){
fprintf(fd, "%6.1f ", readfloat(fs));        
} 
fprintf(fd, "\n");
}
fclose(fs);
fclose(fd);
}

free(xs);
free(ys);
free(u[0][0]);
free(u[1][0]);
free(u[0]);
free(u[1]);


MPI_Type_free(&row);
MPI_Type_free(&column);
MPI_Type_free(&filetype);
MPI_Type_free(&memtype);

MPI_Finalize();
return 0;
}


float readfloat(FILE *f) {
float v;
fread((void*)(&v), sizeof(v), 1, f);
return v;
}