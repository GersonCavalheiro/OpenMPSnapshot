#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#define FALSE           0
#define TRUE            1
#define NORTH           0
#define SOUTH           1
#define WEST            2
#define EAST            3
#define DIMENSIONALITY  2
#define HALO_OFFSET     2
#define ROW             0
#define COLUMN          1
#define SEND            0
#define RECEIVE         1
#define UAT_MODE        0
#define PRINT_MODE      0
#define MASTER          0
struct Parms {
float cx;
float cy;
} parms = {0.1, 0.1};
void printTable(float **grid, int totalRows, int totalColumns);
void cleanUp(MPI_Comm *cartComm, MPI_Datatype *rowType, MPI_Datatype *columnType, MPI_Datatype *subgridType, MPI_Datatype *fileType, float ***grid, int **splitter);
int main(int argc, char **argv) {
int commRank;
int commSize;
int threadNum = 1;
char processorName[MPI_MAX_PROCESSOR_NAME];
int processorNameLen;
int version, subversion;
int steps = 1;
int convFreqSteps;
int createData = 0;
int convergenceCheck = 0;
int fullProblemSize[DIMENSIONALITY];
int subProblemSize[DIMENSIONALITY];
int currentStep;
int currentNeighbor;
int currentConvergenceCheck;
int currentRow, currentColumn;
int currentGrid;
int rc;
MPI_Init(&argc, &argv);
MPI_Get_version(&version, &subversion);
MPI_Get_processor_name(processorName, &processorNameLen);
MPI_Comm_size(MPI_COMM_WORLD, &commSize);
MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
if (argc == 5) {
steps = atoi(argv[1]);
fullProblemSize[ROW] = atoi(argv[2]);
fullProblemSize[COLUMN] = atoi(argv[3]);
convergenceCheck = atoi(argv[4]);
} else if (argc == 3) {
fullProblemSize[ROW] = atoi(argv[1]);
fullProblemSize[COLUMN] = atoi(argv[2]);
createData = 1;
} else {
if (UAT_MODE) {
fullProblemSize[ROW] = 16;
fullProblemSize[COLUMN] = 16;
convergenceCheck = 1;
} else {
printf("Usage: heatconv <ROWS> <COLUMNS> <CONVERGENCE_FLAG>\n");
MPI_Finalize();
exit(EXIT_FAILURE);
}
}
convFreqSteps = (int) sqrt(steps);
MPI_Comm cartComm;
MPI_Request request[2][2][4];
int neighbors[4];
MPI_Datatype rowType;
MPI_Datatype columnType;
MPI_Datatype subgridType;
MPI_Datatype fileType;
int topologyDimension[DIMENSIONALITY] = {0, 0};
int period[DIMENSIONALITY] = {FALSE, FALSE};
int reorder = TRUE;
MPI_Dims_create(commSize, DIMENSIONALITY, topologyDimension);
MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, topologyDimension, period, reorder, &cartComm);
MPI_Cart_shift(cartComm, ROW, 1, &neighbors[NORTH], &neighbors[SOUTH]);
MPI_Cart_shift(cartComm, COLUMN, 1, &neighbors[WEST], &neighbors[EAST]);
int cartRank;
MPI_Comm_rank(cartComm, &cartRank);
int currentCoords[DIMENSIONALITY];
MPI_Cart_coords(cartComm, commRank, DIMENSIONALITY, currentCoords);
if ((fullProblemSize[ROW] % topologyDimension[ROW] || fullProblemSize[COLUMN] % topologyDimension[COLUMN])
&& (!(fullProblemSize[ROW] % topologyDimension[COLUMN]) && !(fullProblemSize[COLUMN] % topologyDimension[ROW]))) {
int tempSize = fullProblemSize[ROW];
fullProblemSize[ROW] = fullProblemSize[COLUMN];
fullProblemSize[COLUMN] = tempSize;
}
if (!(fullProblemSize[ROW] % topologyDimension[ROW]) && !(fullProblemSize[COLUMN] % topologyDimension[COLUMN])) {
subProblemSize[ROW] = fullProblemSize[ROW] / topologyDimension[ROW];
subProblemSize[COLUMN] = fullProblemSize[COLUMN] / topologyDimension[COLUMN];
} else {
printf("subProblem creation error:\n\tfullProblemSize: %dx%d\n\ttopologyDimension: %dx%d\n",
fullProblemSize[ROW], fullProblemSize[COLUMN], topologyDimension[ROW], topologyDimension[COLUMN]);
MPI_Finalize();
exit(EXIT_FAILURE);
}
if (cartRank == MASTER) {
printf("- Execution Info:\n");
printf("-- Steps: %d\n", steps);
printf("-- Full Problem Size: %dx%d\n", fullProblemSize[ROW], fullProblemSize[COLUMN]);
printf("-- Topology Dimension: %dx%d\n", topologyDimension[ROW], topologyDimension[COLUMN]);
printf("-- Sub Problem Size: %dx%d\n\n", subProblemSize[ROW], subProblemSize[COLUMN]);
}
int totalRows = subProblemSize[ROW] + HALO_OFFSET;
int totalColumns = subProblemSize[COLUMN] + HALO_OFFSET;
int workingRows = subProblemSize[ROW];
int workingColumns = subProblemSize[COLUMN];
MPI_Type_vector(subProblemSize[COLUMN], 1, 1, MPI_FLOAT, &rowType);
MPI_Type_vector(subProblemSize[ROW], 1, totalColumns, MPI_FLOAT, &columnType);
MPI_Type_commit(&rowType);
MPI_Type_commit(&columnType);
int sizeArray[DIMENSIONALITY];
int subSizeArray[DIMENSIONALITY];
int startArray[DIMENSIONALITY];
sizeArray[ROW] = totalRows;
sizeArray[COLUMN] = totalColumns;
subSizeArray[ROW] = subProblemSize[ROW];
subSizeArray[COLUMN] = subProblemSize[COLUMN];
startArray[ROW] = 1;
startArray[COLUMN] = 1;
MPI_Type_create_subarray(DIMENSIONALITY, sizeArray, subSizeArray, startArray, MPI_ORDER_C, MPI_FLOAT, &subgridType);
MPI_Type_commit(&subgridType);
sizeArray[ROW] = fullProblemSize[ROW];
sizeArray[COLUMN] = fullProblemSize[COLUMN];
subSizeArray[ROW] = subProblemSize[ROW];
subSizeArray[COLUMN] = subProblemSize[COLUMN];
startArray[ROW] = currentCoords[ROW] * subSizeArray[ROW];
startArray[COLUMN] = currentCoords[COLUMN] * subSizeArray[COLUMN];
MPI_Type_create_subarray(DIMENSIONALITY, sizeArray, subSizeArray, startArray, MPI_ORDER_C, MPI_FLOAT, &fileType);
MPI_Type_commit(&fileType);
char inputFileName[256];
char outputFileName[256];
snprintf(inputFileName, 256, "../io/initial_%d_%d.dat", fullProblemSize[ROW], fullProblemSize[COLUMN]);
snprintf(outputFileName, 256, "../io/final_%d_%d__STEPS_%d__PROCS_%d__THREADS_%d.dat", fullProblemSize[ROW], fullProblemSize[COLUMN], steps, commSize, threadNum);
float *oldGrid, *nextGrid;
float **grid[2];
for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
grid[currentGrid] = (float **) malloc(sizeof(float *) * (totalRows));
grid[currentGrid][0] = (float *) malloc(sizeof(float) * (totalRows * totalColumns));
for (currentRow = 1; currentRow < totalRows; ++currentRow)
grid[currentGrid][currentRow] = &grid[currentGrid][0][currentRow * totalColumns];
}
if (createData) {
int tempRow, tempColumn;
for (currentRow = 0; currentRow < totalRows; currentRow++)
for (currentColumn = 0; currentColumn < totalColumns; currentColumn++) {
tempRow = startArray[ROW] + currentRow;
tempColumn = startArray[COLUMN] + currentColumn;
if (currentRow == 0 || currentColumn == 0 || currentRow == totalRows - 1 || currentColumn == totalColumns - 1)
grid[0][currentRow][currentColumn] = 0;
else
grid[0][currentRow][currentColumn] = (float) ((tempRow - 1) * (fullProblemSize[ROW] - tempRow) * (tempColumn - 1) * (fullProblemSize[COLUMN] - tempColumn));
}
if (PRINT_MODE && cartRank == MASTER) printTable(grid[0], totalRows, totalColumns);
MPI_File fpWrite;
MPI_File_open(cartComm, inputFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fpWrite);
MPI_File_set_view(fpWrite, 0, subgridType, fileType, "native", MPI_INFO_NULL);
MPI_File_write_all(fpWrite, grid[0][0], 1, subgridType, MPI_STATUS_IGNORE);
MPI_File_close(&fpWrite);
cleanUp(&cartComm, &rowType, &columnType, &subgridType, &fileType, grid, 0);
MPI_Finalize();
return EXIT_SUCCESS;
}
MPI_File fpRead;
rc = MPI_File_open(cartComm, inputFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fpRead);
MPI_File_set_view(fpRead, 0, subgridType, fileType, "native", MPI_INFO_NULL);
MPI_File_read_all(fpRead, grid[0][0], 1, subgridType, MPI_STATUS_IGNORE);
MPI_File_close(&fpRead);
if (rc) {
printf("Unable to open file \"%s\"\n", inputFileName);
cleanUp(&cartComm, &rowType, &columnType, &subgridType, &fileType, grid, 0);
MPI_Finalize();
return EXIT_FAILURE;
}
for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
if (neighbors[NORTH] == MPI_PROC_NULL)
for (currentColumn = 0; currentColumn < totalColumns; ++currentColumn)
grid[currentGrid][0][currentColumn] = 0;
if (neighbors[SOUTH] == MPI_PROC_NULL)
for (currentColumn = 0; currentColumn < totalColumns; ++currentColumn)
grid[currentGrid][totalRows - 1][currentColumn] = 0;
if (neighbors[WEST] == MPI_PROC_NULL)
for (currentRow = 0; currentRow < totalRows; ++currentRow)
grid[currentGrid][currentRow][0] = 0;
if (neighbors[EAST] == MPI_PROC_NULL)
for (currentRow = 0; currentRow < totalRows; ++currentRow)
grid[currentGrid][currentRow][totalColumns - 1] = 0;
}
int splitterCount = 2 * subProblemSize[ROW] + 2 * subProblemSize[COLUMN] - 4;
int *splitter[2];
splitter[0] = (int *) malloc(sizeof(int) * 2 * splitterCount);
splitter[1] = (splitter[0] + splitterCount);
int *rowSplitter = splitter[0];
int *columnSplitter = splitter[1];
int tempCounter = 0;
for (currentColumn = 1; currentColumn < subProblemSize[COLUMN] + 1; ++currentColumn) {
splitter[ROW][tempCounter] = 1;
splitter[COLUMN][tempCounter++] = currentColumn;
}
for (currentColumn = 1; currentColumn < subProblemSize[COLUMN] + 1; ++currentColumn) {
splitter[ROW][tempCounter] = subProblemSize[ROW];
splitter[COLUMN][tempCounter++] = currentColumn;
}
for (currentRow = 2; currentRow < subProblemSize[ROW]; ++currentRow) {
splitter[ROW][tempCounter] = currentRow;
splitter[COLUMN][tempCounter++] = 1;
}
for (currentRow = 2; currentRow < subProblemSize[ROW]; ++currentRow) {
splitter[ROW][tempCounter] = currentRow;
splitter[COLUMN][tempCounter++] = subProblemSize[COLUMN];
}
for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
MPI_Send_init(&grid[currentGrid][1][1], 1, rowType, neighbors[NORTH], cartRank, cartComm, &request[SEND][currentGrid][NORTH]);
MPI_Recv_init(&grid[currentGrid][0][1], 1, rowType, neighbors[NORTH], neighbors[NORTH] == MPI_PROC_NULL ? cartRank : neighbors[NORTH], cartComm, &request[RECEIVE][currentGrid][NORTH]);
MPI_Send_init(&grid[currentGrid][totalRows - 2][1], 1, rowType, neighbors[SOUTH], cartRank, cartComm, &request[SEND][currentGrid][SOUTH]);
MPI_Recv_init(&grid[currentGrid][totalRows - 1][1], 1, rowType, neighbors[SOUTH], neighbors[SOUTH] == MPI_PROC_NULL ? cartRank : neighbors[SOUTH], cartComm, &request[RECEIVE][currentGrid][SOUTH]);
MPI_Send_init(&grid[currentGrid][1][1], 1, columnType, neighbors[WEST], cartRank, cartComm, &request[SEND][currentGrid][WEST]);
MPI_Recv_init(&grid[currentGrid][1][0], 1, columnType, neighbors[WEST], neighbors[WEST] == MPI_PROC_NULL ? cartRank : neighbors[WEST], cartComm, &request[RECEIVE][currentGrid][WEST]);
MPI_Send_init(&grid[currentGrid][1][totalColumns - 2], 1, columnType, neighbors[EAST], cartRank, cartComm, &request[SEND][currentGrid][EAST]);
MPI_Recv_init(&grid[currentGrid][1][totalColumns - 1], 1, columnType, neighbors[EAST], neighbors[EAST] == MPI_PROC_NULL ? cartRank : neighbors[EAST], cartComm, &request[RECEIVE][currentGrid][EAST]);
}
double startTime, endTime;
currentGrid = 0;
int localConvergence = TRUE;
int globalConvergence = FALSE;
int convergenceStep = -1;
MPI_Barrier(cartComm);
startTime = MPI_Wtime();
#pragma omp parallel private(currentStep, currentRow, currentColumn, tempCounter)
{
#pragma omp single
{
threadNum = omp_get_num_threads();
if (cartRank == MASTER) {
printf("- MPI/OMP Info\n");
printf("-- version: %d.%d\n", version, subversion);
printf("-- processor name: %s\n", processorName);
printf("-- MPI_COMM_WORLD Size: %d\n", commSize);
printf("-- Thread Pool Size: %d\n\n", threadNum);
}
}
for (currentStep = 0; currentStep < steps; ++currentStep) {
#pragma omp single
{
currentConvergenceCheck = convergenceCheck && currentStep % convFreqSteps == 0;
for (currentNeighbor = 0; currentNeighbor < 4; ++currentNeighbor) {
MPI_Start(&request[RECEIVE][currentGrid][currentNeighbor]);
MPI_Start(&request[SEND][currentGrid][currentNeighbor]);
}
}
oldGrid = grid[currentGrid][0];
nextGrid = grid[1 - currentGrid][0];
if (currentConvergenceCheck) {
#pragma omp for schedule(static) collapse(DIMENSIONALITY) reduction(&&:localConvergence)
for (currentRow = 2; currentRow < workingRows; ++currentRow)
for (currentColumn = 2; currentColumn < workingColumns; ++currentColumn) {
*(nextGrid + currentRow * totalColumns + currentColumn) = *(oldGrid + currentRow * totalColumns + currentColumn) +
parms.cx * (*(oldGrid + (currentRow + 1) * totalColumns + currentColumn) +
*(oldGrid + (currentRow - 1) * totalColumns + currentColumn) -
2.0 * *(oldGrid + currentRow * totalColumns + currentColumn)) +
parms.cy * (*(oldGrid + currentRow * totalColumns + currentColumn + 1) +
*(oldGrid + currentRow * totalColumns + currentColumn - 1) -
2.0 * *(oldGrid + currentRow * totalColumns + currentColumn));
localConvergence = localConvergence && (fabs(*(nextGrid + currentRow * totalColumns + currentColumn) - *(oldGrid + currentRow * totalColumns + currentColumn)) < 1e-2);
}
} else {
#pragma omp for schedule(static) collapse(DIMENSIONALITY)
for (currentRow = 2; currentRow < workingRows; ++currentRow)
for (currentColumn = 2; currentColumn < workingColumns; ++currentColumn) {
*(nextGrid + currentRow * totalColumns + currentColumn) = *(oldGrid + currentRow * totalColumns + currentColumn) +
parms.cx * (*(oldGrid + (currentRow + 1) * totalColumns + currentColumn) +
*(oldGrid + (currentRow - 1) * totalColumns + currentColumn) -
2.0 * *(oldGrid + currentRow * totalColumns + currentColumn)) +
parms.cy * (*(oldGrid + currentRow * totalColumns + currentColumn + 1) +
*(oldGrid + currentRow * totalColumns + currentColumn - 1) -
2.0 * *(oldGrid + currentRow * totalColumns + currentColumn));
}
}
#pragma omp single
{
MPI_Waitall(4, request[RECEIVE][currentGrid], MPI_STATUS_IGNORE);
}
if (currentConvergenceCheck) {
#pragma omp for schedule(static) reduction(&&:localConvergence)
for (tempCounter = 0; tempCounter < splitterCount; ++tempCounter) {
if (*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))) > 1e-4)
*(nextGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))) =
*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))) +
parms.cx * (*(oldGrid + ((*(rowSplitter + tempCounter)) + 1) * totalColumns + (*(columnSplitter + tempCounter))) +
*(oldGrid + ((*(rowSplitter + tempCounter)) - 1) * totalColumns + (*(columnSplitter + tempCounter))) -
2.0 * *(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter)))) +
parms.cy * (*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter)) + 1) +
*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter)) - 1) -
2.0 * *(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))));
localConvergence = localConvergence && (fabs(*(nextGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))) -
*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter)))) < 1e-2);
}
} else {
#pragma omp for schedule(static)
for (tempCounter = 0; tempCounter < splitterCount; ++tempCounter) {
if (*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))) > 1e-4)
*(nextGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))) =
*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))) +
parms.cx * (*(oldGrid + ((*(rowSplitter + tempCounter)) + 1) * totalColumns + (*(columnSplitter + tempCounter))) +
*(oldGrid + ((*(rowSplitter + tempCounter)) - 1) * totalColumns + (*(columnSplitter + tempCounter))) -
2.0 * *(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter)))) +
parms.cy * (*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter)) + 1) +
*(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter)) - 1) -
2.0 * *(oldGrid + (*(rowSplitter + tempCounter)) * totalColumns + (*(columnSplitter + tempCounter))));
}
}
#pragma omp single
{
if (currentConvergenceCheck) {
MPI_Allreduce(&localConvergence, &globalConvergence, 1, MPI_INT, MPI_LAND, cartComm);
localConvergence = TRUE;
}
MPI_Waitall(4, request[SEND][currentGrid], MPI_STATUS_IGNORE);
currentGrid = 1 - currentGrid;
}
if (globalConvergence == TRUE) {
#pragma omp single
{
convergenceStep = currentStep;
}
break;
}
}
}
MPI_Barrier(cartComm);
endTime = MPI_Wtime();
if (cartRank == MASTER) {
printf("Results:\n");
printf("- Runtime: %f sec\n", endTime - startTime);
printf("- Convergence:\n");
printf("-- checking: %s\n", convergenceCheck ? "YES" : "NO");
printf("-- achieved: %s\n", globalConvergence ? "YES" : "NO");
printf("-- at step: %d\n", convergenceStep);
}
MPI_File fpWrite;
MPI_File_open(cartComm, outputFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fpWrite);
MPI_File_set_view(fpWrite, 0, subgridType, fileType, "native", MPI_INFO_NULL);
MPI_File_write_all(fpWrite, grid[currentGrid][0], 1, subgridType, MPI_STATUS_IGNORE);
MPI_File_close(&fpWrite);
cleanUp(&cartComm, &rowType, &columnType, &subgridType, &fileType, grid, splitter);
MPI_Finalize();
return EXIT_SUCCESS;
}
void printTable(float **grid, int totalRows, int totalColumns) {
printf("\n");
for (int currentRow = 0; currentRow < totalRows; ++currentRow) {
for (int currentColumn = 0; currentColumn < totalColumns; ++currentColumn) {
printf("%.1f\t", grid[currentRow][currentColumn]);
}
printf("\n");
}
printf("\n");
}
void cleanUp(MPI_Comm *cartComm, MPI_Datatype *rowType, MPI_Datatype *columnType, MPI_Datatype *subgridType, MPI_Datatype *fileType, float ***grid, int **splitter) {
MPI_Type_free(rowType);
MPI_Type_free(columnType);
MPI_Type_free(subgridType);
MPI_Type_free(fileType);
MPI_Comm_free(cartComm);
if (splitter != 0)
free(splitter[ROW]);
for (int currentGrid = 0; currentGrid < 2; ++currentGrid) {
free(grid[currentGrid][0]);
free(grid[currentGrid]);
}
}