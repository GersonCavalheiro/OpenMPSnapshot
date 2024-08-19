#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifdef USE_MPI
#include <mpi.h>
#endif 
#define MAX_ITERATIONS 1000
#define WIDTH 1536
#define HEIGHT 1024
#define BLOCK_HEIGHT 4
#define BLOCK_WIDTH 1536
#define NUM_JOBS 256
double When()
{
struct timeval tp;
gettimeofday(&tp, NULL);
return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}
void fileWrite(int* pixels) {
int i, j;
FILE *fp;
fp = fopen("test.pgm", "w");
if (fp == NULL) {
perror ( "Unable to open file" );
exit (EXIT_FAILURE);
}
fprintf(fp,"P2\n");
fprintf(fp,"%d %d\n",WIDTH,HEIGHT);
fprintf(fp,"%d\n",MAX_ITERATIONS);
for (j = 0; j < HEIGHT; j++) {
for (i = 0; i < WIDTH; i++) {
fprintf(fp,"%d ",pixels[i + j * WIDTH]);
}
fprintf(fp,"\n");
}
fclose(fp);
}
int computePoint(int _x, int _y) {
int iteration, color;
double xtemp;
double x0, y0, x, y;
x0 = (((double)_x - 1024) / ((double)1024 / (double)2));
y0 = (((double)_y - 512) / ((double)HEIGHT / (double)2));
iteration = 0;
x = 0;
y = 0;
while (((x*x + y*y) < 4) && (iteration < MAX_ITERATIONS)) 
{
xtemp = x*x - y*y + x0;
y = 2*x*y + y0;
x = xtemp;
iteration++;
}
color = MAX_ITERATIONS - iteration;
return color;
}
int main(int argc, char** argv) {
int* pixels;
int i, j;
int color;
int nproc, iproc;
int* mySendArr;
int* myRecvArr;
#ifdef USE_MPI
MPI_Status status;
#endif
int loop; 
int loopCount;
#ifdef USE_MPI
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nproc);
MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
#endif 
mySendArr = (int*)malloc((BLOCK_WIDTH * BLOCK_HEIGHT + 1) * sizeof(int));
myRecvArr = (int*)malloc((BLOCK_WIDTH * BLOCK_HEIGHT + 1) * sizeof(int));
int iter;
if (iproc == 0) {	
int numJobs;
int jobCount;
int jobStart;
double timestart, timefinish, timetaken;
numJobs = NUM_JOBS;
jobCount = 0;
pixels = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
timestart = When();
for (loopCount = 0; loopCount < (NUM_JOBS * 2 + nproc - 1); loopCount++) {
#ifdef USE_MPI
MPI_Recv(myRecvArr,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
if (myRecvArr[0] == -1) {	
if (numJobs > 0) {
mySendArr[0] = jobCount * BLOCK_WIDTH;
jobCount++;
MPI_Send(mySendArr,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD);
numJobs--;
}
else {
mySendArr[0] = -1;
MPI_Send(mySendArr,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD);
}
}
else if (myRecvArr[0] == -2) {	
MPI_Recv(myRecvArr,BLOCK_WIDTH * BLOCK_HEIGHT + 1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD,&status);
jobStart = myRecvArr[0];
for (i = 1; i < BLOCK_WIDTH * BLOCK_HEIGHT + 1; i++) {
pixels[i-1 + jobStart] = myRecvArr[i];
}
}
#endif
}
timefinish = When();
timetaken = timefinish - timestart;
fprintf(stdout,"(%d) Time taken: %lf\n",iproc,timetaken);
fileWrite(pixels);
free(pixels);
}
else {	
int myJobStart;
pixels = (int*)malloc(BLOCK_WIDTH * BLOCK_HEIGHT * sizeof(int));
loop = 1;
while ((loop) && (iter < 1000) ) {
mySendArr[0] = -1;
#ifdef USE_MPI
MPI_Send(mySendArr,1,MPI_INT,0,0,MPI_COMM_WORLD);
#endif
#ifdef USE_MPI
MPI_Recv(myRecvArr,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
#endif
if (myRecvArr[0] == -1) {	
#ifdef USE_MPI
loop = 0;
#else
iter++;
#endif 
}
else {
myJobStart = myRecvArr[0];
#pragma omp parallel for private(i, j)
for (j = 0; j < BLOCK_HEIGHT; j++) {
for (i = 0; i < BLOCK_WIDTH; i++) {
pixels[i + j * BLOCK_WIDTH] = computePoint(i, j + myJobStart / 1536);
}
}				
mySendArr[0] = -2;
#ifdef USE_MPI
MPI_Send(mySendArr,1,MPI_INT,0,0,MPI_COMM_WORLD);
#endif 
mySendArr[0] = myJobStart;
for (i = 1; i < BLOCK_WIDTH * BLOCK_HEIGHT + 1; i++) {
mySendArr[i] = pixels[i-1];
}
#ifdef USE_MPI
MPI_Send(mySendArr,BLOCK_WIDTH * BLOCK_HEIGHT + 1,MPI_INT,0,0,MPI_COMM_WORLD);
#endif 
}	
}
}
#ifdef USE_MPI
MPI_Finalize();
#endif
return 0;
}
