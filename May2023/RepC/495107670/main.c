#include <stdio.h>
#include <stdlib.h>
#include "mpi/mpi.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define FILENAME "/share/dataset.txt"
void readFile(const char* name, int points[], int numData){
char str[10];
char* split;
int i =0;
FILE* file = fopen(name, "r");
while(fgets(str,10,file)&&i<numData){
split = strtok(str, " ");
points[i] = atoi(split);
i++;
}
fclose (file);
}
int getBarIndex(int point, int min1, int barRange, int barsNum, int *low, int *high){
int j;
for (j=0;j<barsNum;j++){
if (point > low[j] && point<= high[j]){
return j;
}
}
}
int main(int argc , char * argv [])
{
int myid, numprocs;
FILE *fptr;
MPI_Status status;
int barsNum, pointNum, threadsNum, PNum;
int num, min1 = 1e9, max1 = 0, barRange;
int *points;
int *myPoints;
int remainder;
int myPointNum;
int *lowRange, *highRange;
int *countBars, *globalBars;
int threadID;
int i, j;
MPI_Init(& argc ,& argv );
MPI_Comm_size( MPI_COMM_WORLD ,& numprocs );
MPI_Comm_rank( MPI_COMM_WORLD ,& myid );
if (myid == 0){
printf("Enter number of bars: ");
scanf("%d", &barsNum);
printf("Enter number of points: ");
scanf("%d", &pointNum);
printf("Enter number of threads: ");
scanf("%d", &threadsNum);
points = malloc(pointNum * sizeof(int));
lowRange = malloc(barsNum * sizeof(int));
highRange = malloc(barsNum * sizeof(int));
readFile(FILENAME,points,pointNum);
int i;
for(i = 0; i < pointNum; i++){
if(points[i] > max1){
max1 = points[i];
}
else if(points[i] < min1){
min1 = points[i];
}
}
barRange = (max1 - (min1 - 1)) / barsNum;
remainder = (max1 - (min1 - 1)) % barsNum;
lowRange[0]=min1-1;
highRange[0] = barRange;
for (j=1;j<barsNum;j++){
lowRange[j] = highRange[j-1];
highRange[j] = lowRange[j]+barRange;
if(remainder >0){
highRange[j]++;
remainder--;
}
}
myPointNum = pointNum / numprocs;
remainder = pointNum % numprocs;
}
MPI_Bcast(&remainder, 1, MPI_INT , 0 , MPI_COMM_WORLD );
MPI_Bcast(&barsNum, 1, MPI_INT , 0 , MPI_COMM_WORLD );
MPI_Bcast(&myPointNum, 1, MPI_INT , 0 , MPI_COMM_WORLD );
MPI_Bcast(&threadsNum, 1, MPI_INT , 0 , MPI_COMM_WORLD );
MPI_Bcast(&barRange, 1, MPI_INT , 0 , MPI_COMM_WORLD );
MPI_Bcast(&min1, 1, MPI_INT , 0 , MPI_COMM_WORLD );
if(myid!=0){
lowRange = malloc(barsNum * sizeof(int));
highRange = malloc(barsNum * sizeof(int));
}
MPI_Bcast(lowRange, barsNum, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(highRange, barsNum, MPI_INT, 0, MPI_COMM_WORLD);
if(myid < remainder){
myPointNum++;
}
if(myid==1){
for(i=0;i<barsNum;i++)
printf("%d %d", lowRange[i], highRange[i]);
}
myPoints = malloc(myPointNum * sizeof(int));
MPI_Scatter(points, myPointNum, MPI_INT, myPoints, myPointNum, MPI_INT, 0, MPI_COMM_WORLD);
countBars = malloc(barsNum * sizeof(int));
globalBars = malloc(barsNum * sizeof(int));
for (i=0;i<barsNum;i++){
countBars[i] = 0;
}
#pragma omp parallel for private(i, j, threadID) num_threads(threadsNum)
for (i=0;i<myPointNum;i++){
threadID = omp_get_thread_num();
j = getBarIndex(myPoints[i], min1, barRange, barsNum, lowRange, highRange);
#pragma omp critical
countBars[j]++;
}
MPI_Reduce(countBars, globalBars, barsNum , MPI_INT,MPI_SUM, 0 , MPI_COMM_WORLD);
if(myid==0){
for (j=0;j<barsNum;j++){
printf("The range start with %d, end with %d with count %d\n", lowRange[j], highRange[j],globalBars[j]);
}
}
MPI_Finalize();
return 0;
}
