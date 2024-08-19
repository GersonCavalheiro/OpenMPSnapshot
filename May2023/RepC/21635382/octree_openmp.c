#include "octree_openmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
int N;
int S;
Box *box;
Box *leaf; 
double **A, **B;
int box_counter = 1; 
int leaf_counter = 0; 
int num_points = 0;
int running_threads = 0;
int num_levels = 0;
int *level_boxes;
void* cubeCheck(void *arg);
void* cubeDivision(void *arg);
void searchNeighbours();
void checkB();
void checkBoundaries();
void AFile();
void BFile();
void neighboursFile();
int main(int argc, char** argv)
{
if (argc < 3)
{
printf("Error in arguments! Two arguments required: N and S\n");
return 0;
}
int i, j, position, total_time,B_counter = 0;
N = atoi(argv[1]);
S = atoi(argv[2]);
A = (double**)malloc(N * sizeof(double*));
if (A == NULL)
{
exit(1);
}
srand(time(NULL));
double y_max = 0;
for (i = 0; i < N; i++)
{
A[i] = (double*)malloc(3 * sizeof(double));
A[i][0] = ((double)rand()/(double)RAND_MAX);
y_max = sqrt(1 - pow(A[i][0], 2));
A[i][1] = ((double)rand()/(double)RAND_MAX)*y_max;
A[i][2] = sqrt(1 - pow(A[i][0], 2) - pow(A[i][1], 2));  
}
printf("Generation of points completed!\n");
gettimeofday(&startwtime, NULL);
B = (double**) malloc(sizeof(double*) * N);
if (B == NULL)
{
exit(1);
} 
for (i = 0; i < N; i++)
{
B[i]= (double*)malloc(sizeof(double) * 3);
}
box = (Box*)malloc(1 * sizeof(Box)); 
if (box == NULL)
{
exit(1);
}
box[0].level = 0;
box[0].boxid = 1;
box[0].parent = 0;
box[0].length = 1;
box[0].center[0] = 0.5;
box[0].center[1] = 0.5;
box[0].center[2] = 0.5;
box[0].start = 0;
box[0].n = N; 
box[0].points = (int*)malloc(N * sizeof(int));
for(i = 0; i < 26; i++)
{
box[0].colleague[i] = 0;
}
for(i = 0; i < N; i++)
{   
box[0].points[i] = i;
}
position = 0;
cubeCheck(&position);
printf("Creation of octree completed!\n");
level_boxes = (int*)malloc((num_levels + 1) * sizeof(int));
if (level_boxes == NULL){
exit(1);
}
printf("Maximum number of levels = %d\n", num_levels);
printf("Total number of cubes = %d\n", box_counter);
searchNeighbours();
printf("All colleagues have been found!\n");
for (i = 0; i < leaf_counter; i++)
{
leaf[i].start = B_counter;
for (j = 0; j < leaf[i].n; j++)
{
B[B_counter][0] = A[leaf[i].points[j]][0];
B[B_counter][1] = A[leaf[i].points[j]][1];
B[B_counter][2] = A[leaf[i].points[j]][2];
B_counter++;
}
}
printf("Array B updated!\n");
gettimeofday(&endwtime, NULL);
total_time = ((endwtime.tv_sec * 1000000 + endwtime.tv_usec) -(startwtime.tv_sec * 1000000 + startwtime.tv_usec));
printf("Total calculation time is: %d us\n", total_time);
printf("\nTask completed!\n");
return (EXIT_SUCCESS);
}
void* cubeCheck(void *arg)
{
int i;
int boxIndex = *(int *)arg; 
Box temp_box, temp_parent;
#pragma omp critical(boxMutex)
{
temp_box = box[boxIndex];
temp_parent = box[temp_box.parent - 1];
}
if (temp_box.boxid != 1)
{
temp_box.points = (int*)malloc(temp_parent.n * sizeof(int));
for (i = 0; i < temp_parent.n; i++)
{   
if (fabs(temp_box.center[0] - A[temp_parent.points[i]][0]) < temp_box.length / 2)
{
if (fabs(temp_box.center[1] - A[temp_parent.points[i]][1]) < temp_box.length / 2)
{
if (fabs(temp_box.center[2] - A[temp_parent.points[i]][2]) < temp_box.length / 2)
{      
temp_box.n++;
temp_box.points[temp_box.n - 1] = temp_parent.points[i];
}
}
}
}
} 
if (temp_box.n == 0)
{ 
#pragma omp critical(boxMutex)
{
temp_box.boxid = 0;
box[boxIndex] = temp_box;
temp_parent.child[temp_box.child_index] = 0;
box[temp_parent.boxid - 1] = temp_parent;
}
}
else if (temp_box.n <= S)
{         
#pragma omp critical(leafMutex)
{
leaf_counter++;
leaf = (Box*)realloc(leaf, leaf_counter * sizeof(Box));
leaf[leaf_counter - 1] = temp_box;
num_points += temp_box.n;
}
#pragma omp critical(boxMutex)
{
box[boxIndex] = temp_box;
}
} 
else
{ 
#pragma omp critical(boxMutex)
{
box[boxIndex] = temp_box;
}
cubeDivision(&temp_box);
}
return NULL;
}
void* cubeDivision(void *arg)
{
Box cube = *(Box *)arg;
int i, j, pos[8];
#pragma omp critical(boxMutex)
{
box = (Box*)realloc(box, (8 + box_counter) * sizeof(Box));
for (i = 0; i < 8; i++)
{
box_counter++;
box[box_counter - 1].level = cube.level + 1;
box[box_counter - 1].boxid = box_counter;
box[box_counter - 1].parent = cube.boxid;
box[box_counter - 1].length = cube.length / 2;
box[box_counter - 1].n = 0;
box[box_counter - 1].child_index = i;
box[cube.boxid - 1].child[i] = box_counter;
box[box_counter - 1].colleague_counter = 0;
for(j = 0; j < 26; j++)
{
box[box_counter - 1].colleague[j] = 0;
}
}   
cube.temp_counter = box_counter; 
box[box_counter - 8].center[0] = cube.center[0] - cube.length / 4;
box[box_counter - 8].center[1] = cube.center[1] - cube.length / 4;
box[box_counter - 8].center[2] = cube.center[2] - cube.length / 4;
box[box_counter - 7].center[0] = cube.center[0] - cube.length / 4;
box[box_counter - 7].center[1] = cube.center[1] - cube.length / 4;
box[box_counter - 7].center[2] = cube.center[2] + cube.length / 4;
box[box_counter - 6].center[0] = cube.center[0] - cube.length / 4;
box[box_counter - 6].center[1] = cube.center[1] + cube.length / 4;
box[box_counter - 6].center[2] = cube.center[2] - cube.length / 4;
box[box_counter - 5].center[0] = cube.center[0] - cube.length / 4;
box[box_counter - 5].center[1] = cube.center[1] + cube.length / 4;
box[box_counter - 5].center[2] = cube.center[2] + cube.length / 4;
box[box_counter - 4].center[0] = cube.center[0] + cube.length / 4;
box[box_counter - 4].center[1] = cube.center[1] - cube.length / 4;
box[box_counter - 4].center[2] = cube.center[2] - cube.length / 4;
box[box_counter - 3].center[0] = cube.center[0] + cube.length / 4;
box[box_counter - 3].center[1] = cube.center[1] - cube.length / 4;
box[box_counter - 3].center[2] = cube.center[2] + cube.length / 4;
box[box_counter - 2].center[0] = cube.center[0] + cube.length / 4;
box[box_counter - 2].center[1] = cube.center[1] + cube.length / 4;
box[box_counter - 2].center[2] = cube.center[2] - cube.length / 4;
box[box_counter - 1].center[0] = cube.center[0] + cube.length / 4;
box[box_counter - 1].center[1] = cube.center[1] + cube.length / 4;
box[box_counter - 1].center[2] = cube.center[2] + cube.length / 4;
if (cube.level + 1 > num_levels)
{
num_levels = cube.level + 1;
} 
}
for (i = 0; i < 8; i++)
{
pos[i] = cube.temp_counter - i - 1;
}
#pragma omp parallel shared(pos) private(i)
{
#pragma omp for schedule(dynamic,1) nowait
for (i = 7; i >= 0; i--)
{
cubeCheck(&pos[i]);
}
}
return NULL;
}
void searchNeighbours()
{
int level, i, j, m, parent_id, child_id, colleague_id, colleague_index;
double dist0, dist1, dist2;
for (level = 0; level < num_levels + 1; level++)
{
for (i = 1; i < box_counter; i++)
{
if (box[i].level == level)
{
parent_id = box[i].parent;
if (parent_id != 0)
{
for (j = 0; j < 8; j++)
{
child_id = box[parent_id - 1].child[j];
if (child_id != 0)
{
if (box[i].boxid != box[child_id - 1].boxid)
{
box[i].colleague[box[i].colleague_counter++] = box[child_id - 1].boxid;
}
}
}
for (j = 0; j < 26; j++)
{
colleague_id = box[parent_id - 1].colleague[j]; 
if (colleague_id != 0)
{
if (box[colleague_id - 1].n > S)
{    
for (m = 0; m < 8; m++)
{
child_id = box[colleague_id - 1].child[m];
if (child_id != 0)
{
if (box[i].boxid != box[child_id - 1].boxid)
{
dist0 = box[child_id - 1].center[0] - box[i].center[0];
dist1 = box[child_id - 1].center[1] - box[i].center[1];
dist2 = box[child_id - 1].center[2] - box[i].center[2]; 
if (sqrt(dist0 * dist0 + dist1 * dist1 + dist2 * dist2) <= sqrt(3) * box[i].length)
{    
colleague_index = box[i].colleague_counter;
box[i].colleague[colleague_index] = box[child_id - 1].boxid;
box[i].colleague_counter++;
}
}
}
}
}
}
}
}
}
}
}
}  
void checkB()
{
int i, j, same_counter = 0;
for (i = 0; i < N; i++)
{
for (j = 0; j < N; j++)
{
if ((B[i][0] == A[j][0]) && (B[i][1] == A[j][1]) && (B[i][2] == A[j][2]))
{
same_counter++;
}
}
}
if (same_counter == N)
{
printf("All points of B are also points of A\n");
}
else
{
printf("Error with points of B\n");
}
}
void checkBoundaries()
{
int i, j, points_counter = 0;
double x, y, z;
for (i = 0; i < leaf_counter; i++)
{
for (j = 0; j < leaf[i].n; j++)
{
x = fabs(leaf[i].center[0] - A[leaf[i].points[j]][0]);
y = fabs(leaf[i].center[1] - A[leaf[i].points[j]][1]);
z = fabs(leaf[i].center[2] - A[leaf[i].points[j]][2]);
if (x < leaf[i].length / 2 && y < leaf[i].length / 2 && z < leaf[i].length / 2)
{
points_counter++;
}
}
}
if (points_counter == N)
{
printf("All points of leafs meet boundaries of subcubes\n");
}
else
{
printf("Error with points of leafs\n");
}
}
void AFile()
{
remove("alpha.txt");
FILE *A_file;
int i;
A_file = fopen("A.txt", "wt");
for(i = 0; i < N; i++)
{
fprintf(A_file,"%f,%f,%f\n",A[i][0],A[i][1],A[i][2]); fflush(A_file);
}
fclose(A_file);
}
void BFile()
{
remove("B.txt");
FILE *B_file;
int i;
B_file = fopen("B.txt", "wt");
for(i = 0; i < N; i++)
{
fprintf(B_file,"%f,%f,%f\n",B[i][0],B[i][1],B[i][2]); fflush(B_file);  
}
fclose(B_file);
}
void neighboursFile()
{
remove("colleagues.txt");
FILE *neighbours_file;
int i, j;
neighbours_file = fopen("neighbours.txt", "wt");
for( i = 0; i < box_counter; i++)
{               
if (box[i].boxid != 0)
{
fprintf(neighbours_file,"id: %8d   neighbours:",box[i].boxid); fflush(neighbours_file);
for(j = 0; j < 26; j++)
{
if(box[i].colleague[j] != 0)
{
fprintf(neighbours_file,"%8d",box[i].colleague[j]); fflush(neighbours_file);
}
}
fprintf(neighbours_file,"\n"); fflush(neighbours_file);
}   
}
fclose(neighbours_file);   
}