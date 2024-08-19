#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#define MAX_ITERATIONS 500000     
#define EPSILON .01               
#define NORTH 100                 
#define SOUTH 100                 
#define EAST 0                    
#define WEST 0                    
#define WIDTH 1000                
#define HEIGHT 1000               
#define MASTER 0                  
void runIterations();           
void printImage(float* plate);
void printHeader(int iterCount);
int main(int argc, char** argv) {
MPI_Init(&argc, &argv);             
int worldRank, worldSize;           
double start, stop;                 
int numProcs, numThreads;           
int iterCount;                      
MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
if (worldRank == MASTER) {
printf("Shaun Jorstad ~ Thermal Plate conduction\n");
start = MPI_Wtime();            
}
runIterations();
if (worldRank == MASTER) {
stop = MPI_Wtime();             
printf("Simulation took: %f seconds\n", (stop-start));
printf("-----Normal Termination-----\n");
}
MPI_Finalize();
return 0;
}
void runIterations() {
int     row, col;               
int     iterCount;              
int     rowStart, rowEnd;       
int     averageColor;           
int     worldRank, worldSize;   
int     procIndex;              
int     processHeight;          
double  diffNorm, gDiffNorm;    
MPI_Status status;              
MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
averageColor = (NORTH + SOUTH + EAST + WEST) / 4;
iterCount = 0;
processHeight = (HEIGHT/ worldSize);
rowStart = 1;
rowEnd = rowStart + processHeight;
if (worldRank == MASTER) {
rowEnd -= 1;
}
else if (worldRank == worldSize-1) {
rowEnd -= 1;
}
float *current = (float*) malloc(WIDTH * (processHeight + 2) * sizeof(float)); 
float *next = (float*) malloc(WIDTH * (processHeight +2) * sizeof(float));
for (row = rowStart; row < rowEnd; row++) {
for (col = 0; col < WIDTH; col++) {
*(current + (row * WIDTH) + col) = averageColor;
*(next + (row * WIDTH) + col) = averageColor;
}
}
if (worldRank == MASTER) {
for (col = 0; col < WIDTH; col++) {
*(current + col) = 100;
*(next + col) = 100;
}
}
if (worldRank == worldSize -1) {
for (col = 0; col < WIDTH; col++) {
*(current + (250 * WIDTH) + col) = 100;
*(next + (250 * WIDTH) + col) = 100;
}
}
for (row = rowStart; row < rowEnd; row++) {
*(current + (row * WIDTH)) = 0;
*(current + ((row + 1) * WIDTH) -1) = 0;
*(next + (row * WIDTH)) = 0;
*(next + ((row + 1) * WIDTH) -1) = 0;
}
do {
if (worldRank < worldSize - 1) {
MPI_Send( current + ((rowEnd -1) * WIDTH), WIDTH, MPI_FLOAT, worldRank + 1, 0, MPI_COMM_WORLD); 
}
if (worldRank > 0) {
MPI_Recv( current, WIDTH, MPI_FLOAT, worldRank - 1, 0, MPI_COMM_WORLD, &status );
}
if (worldRank > 0) {
MPI_Send( current + WIDTH, WIDTH, MPI_FLOAT, worldRank - 1, 1, MPI_COMM_WORLD );
}
if (worldRank < worldSize - 1) {
MPI_Recv( current + (rowEnd * WIDTH), WIDTH, MPI_FLOAT, worldRank + 1, 1, MPI_COMM_WORLD, &status );
}
iterCount ++;
diffNorm = 0.0;
#pragma omp parallel for reduction(+:diffNorm) num_threads(4) private(row, col) shared(current, next)
for (row = rowStart; row < rowEnd; row++) {
for (col = 1; col < WIDTH -1; col++) {
*(next + (row * WIDTH) + col) = (*(current + (row * WIDTH) + (col -1)) + *(current + (row * WIDTH) + (col + 1)) + *(current + ((row - 1) * WIDTH) + col) + *(current + ((row+1) * WIDTH) +col)) / 4;
diffNorm += (*(next + (row * WIDTH) + col) - *(current + (row * WIDTH) + col)) * (*(next + (row * WIDTH) + col) - *(current + (row * WIDTH) + col));
}
}
float* tempPointer = current;
current = next;
next = tempPointer;
MPI_Allreduce( &diffNorm, &gDiffNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
gDiffNorm = sqrt( gDiffNorm );
if (worldRank == 0 && iterCount % 1000 == 0) {
printf( "At iteration %d, diff is %e\n", iterCount, gDiffNorm );
}
} while (gDiffNorm > EPSILON && iterCount < MAX_ITERATIONS);
if (worldRank == MASTER) {
printHeader(iterCount);
printImage(current, iterCount);
for (procIndex = 1; procIndex < 4; procIndex++) {
MPI_Recv(current, WIDTH*(processHeight), MPI_FLOAT, procIndex, 1, MPI_COMM_WORLD, &status);
printImage(current, 0);
}
}
else {
MPI_Send((current + (WIDTH)), WIDTH*(processHeight), MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
}
free(current);
free(next);
}
void printHeader(int iterCount) {
char* path = "./heatmap.ppm";   
FILE *out;                      
out = fopen(path, "w+");        
int r = 0;                      
int c = 0;
int pixel;
float red, blue;                
fprintf(out, "P3\n");
fprintf(out, "# Shaun Jorstad ~ Generated Thermal Plate\n");
fprintf(out, "# Forked source code from Argonne National Laboratory\n");
fprintf(out, "# Executed in: %d iterations\n", iterCount);
fprintf(out, "%d %d 255\n", WIDTH, HEIGHT);
fclose(out);    
}
void printImage(float* plate) {
char* path = "./heatmap.ppm";   
FILE *out;                      
out = fopen(path, "a+");        
int r = 0;                      
int c = 0;
int pixel;
float red, blue;                
while (r < 250) {
while (c < WIDTH) {
for (pixel = 0; pixel < 5; pixel++) {
red = (*(plate + (r*WIDTH) + c)/ 100.0) * 255.0;
blue = 255 - red;
fprintf(out, "%.0f 0 %.0f\t", red, blue);
c += 1;
}
fprintf(out, "\n");
}
r++;
c = 0;
}
fclose(out);    
}
