#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "gameOfLife.h"
#include "mpiFunctions.h"
#define NLOOPS 100
#include <omp.h>



int main(int argc, char **argv)
{
MPI_Datatype ROW, COLUMN;
int myid, numprocs;
MPI_Status recv_status, send_status;
MPI_Request sendReq[2][8];
MPI_Request receiveReq[2][8];

MPI_Comm gridComm;
int** block1;
int** block2;
int** currentBlock;
int** prevBlock;
int** tempBlock;


int axisSize;
int neighbors[8];
int nRows, nColumns, TotalRows = -1, TotalColumns = -1;
int i, j, k;
int change, GlobalChange;
int finReqstId;
char* filename = NULL;
double startTime, finishTime, localElapsed, globalElapsed;
int pos;
int mpi_support;

if (argc < 5 || argc > 7)
{
printf("Please give all attributes –r <rows> –c <columns> -f <filename(optimal)>\n");
exit(1);
}

for(i=1; i < argc; i++)
{
if(i+1 != argc) 
{
if(strcmp(argv[i], "-r") == 0)
{
TotalRows = atoi(argv[i+1]);
}
else if(strcmp(argv[i], "-c") == 0)
{
TotalColumns = atoi(argv[i+1]);
}
else if(strcmp(argv[i], "-f") == 0)
{
filename = argv[i+1];
}
}
}

if(TotalColumns == -1 || TotalRows == -1)
{
printf("No Input\n");
exit(1);
}

MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &mpi_support);

if (mpi_support != MPI_THREAD_MULTIPLE)
printf("MPI_THREAD_MULTIPLE thread support required\n");

MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myid);

srand(time(NULL) ^ myid); 

axisSize = sqrt(numprocs); 

nRows = TotalRows/axisSize;
nColumns = TotalColumns/axisSize;


if(TotalRows%axisSize != 0 || TotalColumns%axisSize !=0)
{
printf("rows or columns cannot be divided equality to all process\n");
MPI_Finalize();
exit(1);
}

createCartesianTopology(&gridComm, axisSize); 

MPI_Type_contiguous(nColumns, MPI_INT, &ROW); 
MPI_Type_commit(&ROW);
MPI_Type_vector(nRows, 1, nColumns +2, MPI_INT, &COLUMN); 
MPI_Type_commit(&COLUMN);


nRows += 2;
nColumns +=2;

block1 = allocate2dArray(nRows, nColumns);
block2 = allocate2dArray(nRows, nColumns);

MPI_Comm_rank(gridComm, &myid); 

if(filename == NULL) 
{
printf("Random initialization\n");
initializeBlock(block1, nRows, nColumns);
}
else 
{
printf("Input from file\n");
readFromFile(filename, gridComm, myid, block1, nRows-2, nColumns -2, TotalRows, TotalColumns);
}

neighborProcess(gridComm, myid, neighbors); 

prevBlock = block1;
currentBlock = block2;

for(j=0; j < 8; j++)
{
sendMSG(neighbors[j], j, prevBlock, nRows, nColumns, &sendReq[1][j], gridComm, ROW, COLUMN);
receiveMSG(neighbors[j], j, prevBlock, nRows, nColumns, &receiveReq[1][j], gridComm, ROW, COLUMN);

sendMSG(neighbors[j], j, currentBlock, nRows, nColumns, &sendReq[0][j], gridComm, ROW, COLUMN);
receiveMSG(neighbors[j], j, currentBlock, nRows, nColumns, &receiveReq[0][j], gridComm, ROW, COLUMN);
}

MPI_Barrier(gridComm); 
startTime = MPI_Wtime();

int tid;
#pragma omp parallel num_threads(2) private (tid) private (i,j,k,pos)
{

for(k =1; k<= NLOOPS; k++)
{
pos = k%2; 

#pragma omp for
for(i=0; i < 8; i++) 
{
MPI_Start(&sendReq[pos][i]);
}

#pragma omp for
for(i=0; i < 8; i++) 
{
MPI_Start(&receiveReq[pos][i]);
}

for(i = 2; i < nRows-2; i++) 
{
#pragma omp for
for(j = 2; j <nColumns-2; j++)
{
currentBlock[i][j] = updatedValue(prevBlock[i][j], activeNeighborsNoBound(prevBlock, i, j));
}
}

#pragma omp for
for(i=0; i < 8; i++)
{
MPI_Waitany(8, receiveReq[pos], &finReqstId, &recv_status);
if(finReqstId == UP)
{
calculateBound(prevBlock, currentBlock, nRows, nColumns, UPROW);
}
else if(finReqstId == DOWN)
{
calculateBound(prevBlock, currentBlock, nRows, nColumns, DOWNROW);
}
else if(finReqstId == LEFT)
{
calculateBound(prevBlock, currentBlock, nRows, nColumns, LEFTCOLUMN);
}
else if(finReqstId == RIGHT)
{
calculateBound(prevBlock, currentBlock, nRows, nColumns, RIGHTCOLUMN);
}
}

#pragma omp single
{
currentBlock[1][1] = updatedValue(prevBlock[1][1], activeNeighborsNoBound(prevBlock, 1, 1)); 
currentBlock[1][nColumns-2] = updatedValue(prevBlock[1][nColumns-2], activeNeighborsNoBound(prevBlock, 1, nColumns-2)); 
currentBlock[nRows-2][1] = updatedValue(prevBlock[nRows-2][1], activeNeighborsNoBound(prevBlock, nRows-2, 1)); 
currentBlock[nRows-2][nColumns-2] = updatedValue(prevBlock[nRows-2][nColumns-2], activeNeighborsNoBound(prevBlock, nRows-2, nColumns-2)); 
}

#pragma omp for
for(i=0; i <8; i++) 
{
MPI_Wait(&sendReq[pos][i], &send_status);
}


#pragma omp barrier

#pragma omp single
{
tempBlock = prevBlock;
prevBlock = currentBlock; 
currentBlock = tempBlock; 
}

}

}

finishTime = MPI_Wtime();
localElapsed = finishTime - startTime;

MPI_Reduce(&localElapsed, &globalElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, gridComm);
if(myid == 0)
{
printf("Total time is %f\n", globalElapsed);
}

MPI_Comm_free(&gridComm);
MPI_Type_free(&ROW);
MPI_Type_free(&COLUMN);
MPI_Finalize();

free(block1);
free(block2);
exit(0);
}
