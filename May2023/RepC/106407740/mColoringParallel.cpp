#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<stdbool.h>
#include<sys/time.h>
int V;
#define LEVEL 22
bool found=false;
void printSolution(int color[] , int graph[][100]);
bool isSafe (int v, int graph[][100], int color[], int c)
{
for (int i = 0; i < V; i++)
if (graph[v][i]==1 && c == color[i])
return false;
return true;
}
void graphColoringUtil(int graph[][100], int m, int color[], int v)
{
if(found==false)
{
if (v == V)
{
printSolution(color,graph);
return ;
}
for (int c = 1; c <= m; c++)
{
color[v] = c;
if (isSafe(v, graph, color, c))
{
graphColoringUtil (graph, m, color, v+1);
}
}
return ;
}
}
void graphColoringUtilParallel(int graph[][100], int m, int color[], int v)
{
if(found==false)
{
for (int i = 1; i <= m; ++i) {
color[v] = i;
if (isSafe(v, graph, color, i)) {
if (v < LEVEL)
{
int *tempColors = new int[V];
for (int j = 0; j <= v; ++j) {
tempColors[j] = color[j];
}
#pragma omp task firstprivate(v)
{
int id = omp_get_thread_num();
graphColoringUtilParallel(graph, m, tempColors, v+1);   
}
}
else
{
#pragma omp taskwait
graphColoringUtil(graph, m, color, v+1);
}
}
}
return;
}
}
void graphColoring(int graph[][100], int m)
{
int *color = new int[V];
for (int i = 0; i < V; i++)
color[i] = 0;
#pragma omp parallel shared(found)
{
#pragma omp single
{
graphColoringUtilParallel(graph, m, color, 0 );
}
}
}
void printSolution(int color[] , int graph[][100])
{     found=true;
printf("Solution Exists:"
" Following are the assigned colors \n");
for (int i = 0; i < V; i++)
printf(" %d ", color[i]);
printf("\n");
}
int main()
{
found=false;
struct timeval  TimeValue_Start;
struct timezone TimeZone_Start;
struct timeval  TimeValue_Final;
struct timezone TimeZone_Final;
long   time_start, time_end;
double  time_overhead;
printf("Enter Number of vertices\n");
scanf("%d",&V);
int graph[100][100];
for(int i=0;i<V;i++)
{
for (int j=0;j<V;j++)
{
if(i==j)
graph[i][j]=0;
else{
graph[i][j] = rand()%2;
graph[j][i] = graph[i][j];
}
}
}
printf("Entered Adjacency Matrix\n");
for(int i=0;i<V;i++)
{
for (int j=0;j<V;j++)
{
printf("%d ", graph[i][j]);
}
printf("\n");
}
printf("Enter Number of colours\n");
int m;
scanf("%d", &m);
gettimeofday(&TimeValue_Start, &TimeZone_Start);
graphColoring (graph, m);
if(found==false)
printf("No solution exists\n");
gettimeofday(&TimeValue_Final, &TimeZone_Final);
time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
time_overhead = (time_end - time_start)/1000000.0;
printf("\n Time in Seconds (T)  : %lf",time_overhead);
return 0;
}
