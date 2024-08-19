#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
int create(int size,int **matrix);
int display(int size, int **matrix);
int countNeighbor(int size,int **matrix, int **temp,int q ,int steps, int num_of_threads);
int main( int argc, char* argv[])
{
int size,steps, **matrix, **temp;
size=atoi (argv[1]);
steps = atoi (argv[2]);
int num_of_threads= atoi (argv[3]);
matrix = (int**)malloc((size + 2) * sizeof(int*));
temp= (int**)malloc((size + 2) * sizeof(int*));
int q; 
for (q = 0;q<size + 2;q++){
matrix[q] = (int*)malloc((size + 2) * sizeof(int));
temp[q] = (int*)malloc((size + 2) * sizeof(int));
}
clock_t start = clock();
int z;
for (z = 0;z<size + 2;z++){
matrix[z] = (int*)malloc((size + 2) * sizeof(int));
temp[z] = (int*)malloc((size + 2) * sizeof(int));
}
**matrix = create(size,matrix);
for (z = 0; z<steps; z++)
{
**matrix = countNeighbor(size,matrix,temp,q,steps,num_of_threads);
}
display(size, matrix);
clock_t stop = clock();
double time_taken = (double)(stop - start) / CLOCKS_PER_SEC;
printf("\n Time elapsed in ms: %f", time_taken);
return 0;
}
int create(int size, int **matrix)
{
int a,b;
for( a=1;a<=size;a++){
for( b=1;b<=size;b++){
matrix[a][b]=rand()%2;
int x,y;
for ( x = 0; x<=size+1;x++)
{
for( y = 0; y <= size+1; y++)
{
if( x == 0 || y == 0)
{
matrix[x][y] = 0;
}
if(x == size+1 || y == size +1)
{
matrix[x][y] = 0;
}
}
}
}
}
return **matrix;
}
int display(int size,int **matrix)
{
int c,d;
for ( c =0; c<size+2;c++)
{
for( d = 0;d<size+2;d++)
{
printf("%d",matrix[c][d]);
}
printf("\n");
}
return 0;
}
int countNeighbor(int size, int **matrix, int **temp,int q,int steps,int num_of_threads)
{
int threadid;
#pragma omp parallel shared (size,matrix,temp,num_of_threads) num_threads(num_of_threads)
threadid = omp_get_thread_num();
printf("thread id:%d",threadid);
int low,high;
int neo = size/num_of_threads;
low = threadid*neo;
high = (threadid*neo) +neo;
if (threadid == 0)
low++;
if (threadid == num_of_threads -1)
high = size -1;
for( q =0;q< steps;q++)
{
int i,j;
for( i=1;i<size+1;i++)
{
for( j=low;j<high;j++)
{
int count = 0;
if(matrix[i-1][j]==1){
count+=1;
}
if(matrix[i-1][j-1]==1){
count+=1;}
if(matrix[i-1][j+1]==1){
count+=1;
}
if(matrix[i+1][j]==1){
count+=1;
}
if(matrix[i+1][j+1]==1){
count+=1;
}
if(matrix[i+1][j-1]==1){
count+=1;
}
if(matrix[i][j-1]==1){
count+=1;
}
if(matrix[i][j+1]==1){
count+=1;
}
if(count < 2||count >3)
{
temp[i][j]=0;
}
if(count ==2)
{
temp[i][j]=matrix[i][j];
}
if(count ==3)
{
temp[i][j]=1;
}
}
}
int m,n;
for( m=0;m<size+2;m++)
{
for( n=0;n<size+2;n++)
{
matrix[m][n]=temp[m][n];
}
}
}
return **matrix;
}
