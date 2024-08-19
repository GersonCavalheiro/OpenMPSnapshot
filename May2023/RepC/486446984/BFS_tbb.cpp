#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "tbb/parallel_for_each.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>
#include "tbb/tick_count.h"
#include "tbb/tbb_thread.h"
#include "omp.h"
using namespace tbb;
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
void basic_BFS(int* given_graph, int NUM_NODES, int* visited){
int max_loop = 0;
int first = -1;
int visited_nodes = 0;
int last = 0;
int start_node = 3;
queue[last++] = start_node;
first++;
visited[start_node] = 1;
int cur_node;
int my_counter = 0;
while(first != last){
if(visited_nodes == NUM_NODES)
break;	
cur_node = queue[first++];
int i;
int curr_loop = 0;
for(i = 0; i < NUM_NODES; i++){
curr_loop++;
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
my_counter++;
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
}
if(curr_loop > max_loop)
max_loop = curr_loop;
}
printf("Max iterations is: %d\n", max_loop);
printf("My counter in bfs normal is: %d\n", my_counter);
}
void OMP_BFS(int* given_graph, int NUM_NODES, int* visited, int thread_num){
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
for(i = 0; i < NUM_NODES; i++){
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
}
}
}
void OMP_BFS_static(int* given_graph, int NUM_NODES, int* visited, int thread_num){
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
for(i = 0; i < NUM_NODES; i++){
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
}
}
}
void OMP_BFS_dynamic(int* given_graph, int NUM_NODES, int* visited, int thread_num){
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
for(i = 0; i < NUM_NODES; i++){
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
}
}
}
void OMP_BFS_guided(int* given_graph, int NUM_NODES, int* visited, int thread_num){
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
for(i = 0; i < NUM_NODES; i++){
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
}
}
}
void TBB_BFS(int* given_graph, int NUM_NODES, int* visited, int thread_num){
double start[8], end[8];
double thread_time[8];
task_scheduler_init tbb_init(8);
int max_edges = 0;
int first = -1;
int last = 0;
int visited_nodes = 0;
int start_node = 3;
queue[last++] = start_node;
first++;
visited[start_node] = 1;
int cur_node, k=0;
int my_counter = 0;
while(first != last){
if(visited_nodes == NUM_NODES)
break;	
cur_node = queue[first++];
int i;
parallel_for(blocked_range<int>(0,NUM_NODES),
[&](blocked_range<int> r)
{	
int bgn = r.begin();
int nd = r.end();		
for (int i=bgn; i<nd; ++i)
{
start[tbb::task_arena::current_thread_index()]=omp_get_wtime();
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
my_counter++;	
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
end[tbb::task_arena::current_thread_index()]=omp_get_wtime();
thread_time[tbb::task_arena::current_thread_index()] += end[tbb::task_arena::current_thread_index()] - start[tbb::task_arena::current_thread_index()];
}
});
}
printf("%f %f %f %f %f %f %f %f\n", thread_time[0], thread_time[1], thread_time[2], thread_time[3], thread_time[4], thread_time[5], thread_time[6], thread_time[7]);
printf("MY TBB COunter is %d\n", my_counter);
}
int main()
{
int n, i, e, v;
double start, end;
tick_count t0, t1;
v = 5000;
e = v*12;
for(int j=0; j<50; j++){
int* visited = (int*) malloc(v*sizeof(int));
int* adjugate_matrix  = (int*) malloc(v*v*sizeof(int));
for(int k=0; k<v*v; k++) adjugate_matrix[k] = 0;
printf("\n\nV = %d, E = %d\n", v, e);
printf("[Creating Graph]\n...\n");
GenerateRandGraphs(e, v, adjugate_matrix);
for(i = 0; i < v; i++) visited[i] = 0;
basic_BFS(adjugate_matrix, v, visited);
printf("BASIC BFS took %g seconds\n", (t1-t0).seconds());
for(i = 0; i < v; i++) visited[i] = 0;
OMP_BFS(adjugate_matrix, v, visited, 8);
printf("OMP BFS took %g seconds\n", (t1-t0).seconds());
for(i = 0; i < v; i++) visited[i] = 0;
TBB_BFS(adjugate_matrix, v, visited, 8);
printf("TBB BFS took %g seconds\n", (t1-t0).seconds());
printf("\n");
printf("DONE!\n");
printf("---------------------------------------------------\n");
free(visited);
free(adjugate_matrix);
v = v+5000;
e = v*12;
}
}
