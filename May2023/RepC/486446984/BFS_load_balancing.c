#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "omp.h"
#include <time.h>
double interval(struct timespec start, struct timespec end)
{
struct timespec temp;
temp.tv_sec = end.tv_sec - start.tv_sec;
temp.tv_nsec = end.tv_nsec - start.tv_nsec;
if (temp.tv_nsec < 0) {
temp.tv_sec = temp.tv_sec - 1;
temp.tv_nsec = temp.tv_nsec + 1000000000;
}
return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
void GenerateRandGraphs(int NOE, int NOV, int *MARIX)
{
int i, j, edge[NOE][2], count;
i = 0;
while(i < NOE)
{
edge[i][0] = rand()%NOV;
edge[i][1] = rand()%NOV;
if(edge[i][0] == edge[i][1])
continue;
else
{
for(j = 0; j < i; j++)
{
if((edge[i][0] == edge[j][0] && edge[i][1] == edge[j][1]) || (edge[i][0] == edge[j][1] && edge[i][1] == edge[j][0]))
{	
i--;
}
}
}
i++;
}
for(i = 0; i < NOV; i++)
{
count = 0;
for(j = 0; j < NOE; j++)
{
if(edge[j][0] == i)
{
MARIX[edge[j][1]+i*NOV] = 1;
count++;
}
else if(edge[j][1] == i)
{
MARIX[edge[j][0]+i*NOV] = 1;
count++;
}
else if(j == NOE-1 && count == 0)
continue;
}
}
}
int queue[10000000];
void OMP_BFS_static(int* given_graph, int NUM_NODES, int* visited){
double start[8], end[8];
double thread_time[8]; 
for(int i=0; i<8; i++) thread_time[i]=0;
int first = -1;
int last = 0;
int visited_nodes = 0;
int start_node = 3;
queue[last++] = start_node;
first++;
visited[start_node] = 1;
int cur_node, k=0;
while(first != last){
if(visited_nodes == NUM_NODES)
break;	
cur_node = queue[first++];
int i;
omp_set_num_threads(8);
#pragma omp parallel for schedule(static) shared(visited)
for(i = 0; i < NUM_NODES; i++){
start[omp_get_thread_num()] = omp_get_wtime(); 
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
end[omp_get_thread_num()] = omp_get_wtime(); 
thread_time[omp_get_thread_num()] += end[omp_get_thread_num()] - start[omp_get_thread_num()];
}
}
printf("Thread_0 Thread_1 Thread_2 Thread_3 Thread_4 Thread_5 Thread_6 Thread_7\n");
printf("%f %f %f %f %f %f %f %f\n", thread_time[0], thread_time[1], thread_time[2], thread_time[3], thread_time[4], thread_time[5], thread_time[6], thread_time[7]);
}
void OMP_BFS_dynamic(int* given_graph, int NUM_NODES, int* visited){
double start[8], end[8];
double thread_time[8]; 
for(int i=0; i<8; i++) thread_time[i]=0;
int first = -1;
int last = 0;
int visited_nodes = 0;
int start_node = 3;
queue[last++] = start_node;
first++;
visited[start_node] = 1;
int cur_node, k=0;
while(first != last){
if(visited_nodes == NUM_NODES)
break;	
cur_node = queue[first++];
int i;
omp_set_num_threads(8);
#pragma omp parallel for schedule(dynamic) shared(visited)
for(i = 0; i < NUM_NODES; i++){
start[omp_get_thread_num()] = omp_get_wtime(); 
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
end[omp_get_thread_num()] = omp_get_wtime(); 
thread_time[omp_get_thread_num()] += end[omp_get_thread_num()] - start[omp_get_thread_num()];
}
}
printf("Thread_0 Thread_1 Thread_2 Thread_3 Thread_4 Thread_5 Thread_6 Thread_7\n");
printf("%f %f %f %f %f %f %f %f\n", thread_time[0], thread_time[1], thread_time[2], thread_time[3], thread_time[4], thread_time[5], thread_time[6], thread_time[7]);
}
void OMP_BFS_guided(int* given_graph, int NUM_NODES, int* visited){
double start[8], end[8];
double thread_time[8]; 
for(int i=0; i<8; i++) thread_time[i]=0;
int first = -1;
int last = 0;
int visited_nodes = 0;
int start_node = 3;
queue[last++] = start_node;
first++;
visited[start_node] = 1;
int cur_node, k=0;
while(first != last){
if(visited_nodes == NUM_NODES)
break;	
cur_node = queue[first++];
int i;
omp_set_num_threads(8);
#pragma omp parallel for schedule(guided) shared(visited)
for(i = 0; i < NUM_NODES; i++){
start[omp_get_thread_num()] = omp_get_wtime(); 
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
end[omp_get_thread_num()] = omp_get_wtime(); 
thread_time[omp_get_thread_num()] += end[omp_get_thread_num()] - start[omp_get_thread_num()];
}
}
printf("Thread_0 Thread_1 Thread_2 Thread_3 Thread_4 Thread_5 Thread_6 Thread_7\n");
printf("%f %f %f %f %f %f %f %f\n", thread_time[0], thread_time[1], thread_time[2], thread_time[3], thread_time[4], thread_time[5], thread_time[6], thread_time[7]);
}
int main()
{
int n, i, e, v;
double start, end;
v = 5000;
e = v*12;
for(int j=1; j<50; j++){
int* visited = (int*) malloc(v*sizeof(int));
int* adjugate_matrix  = (int*) malloc(v*v*sizeof(int));
for(int k=0; k<v*v; k++) adjugate_matrix[k] = 0;
printf("\n\nV = %d, E = %d\n", v, e);
printf("[Creating Graph]\n...\n");
GenerateRandGraphs(e, v, adjugate_matrix);
for(i = 0; i < v; i++) visited[i] = 0;
printf("\n[Staic Scheduling]\n");
OMP_BFS_static(adjugate_matrix, v, visited);
for(i = 0; i < v; i++) visited[i] = 0;
printf("\n[Dynamic Scheduling]\n");
OMP_BFS_dynamic(adjugate_matrix, v, visited);
for(i = 0; i < v; i++) visited[i] = 0;
printf("\n[Guided Scheduling]\n");
OMP_BFS_guided(adjugate_matrix, v, visited);
printf("\n");
printf("DONE!\n");
printf("---------------------------------------------------\n");
free(visited);
free(adjugate_matrix);
v = v+5000;
e = e*12;
}
}
