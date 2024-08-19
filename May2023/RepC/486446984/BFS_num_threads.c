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
void basic_BFS(int* given_graph, int NUM_NODES, int* visited){
int first = -1;
int visited_nodes = 0;
int last = 0;
int start_node = 3;
queue[last++] = start_node;
first++;
visited[start_node] = 1;
int cur_node;
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
omp_set_num_threads(thread_num);
#pragma omp parallel for shared(visited)
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
omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(static) shared(visited)
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
omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(dynamic) shared(visited)
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
omp_set_num_threads(thread_num);
#pragma omp parallel for schedule(guided) shared(visited)
for(i = 0; i < NUM_NODES; i++){
if(given_graph[cur_node*NUM_NODES + i]==1 && visited[i]==0){
queue[last++] = i;
visited[i]=1;
visited_nodes++;
}
}
}
}
int main()
{
int n, i, e, v;
double start, end;
v = 50000;
e = v*12;
for(int j=0; j<50; j++){
int* visited = (int*) malloc(v*sizeof(int));
int* adjugate_matrix  = (int*) malloc(v*v*sizeof(int));
for(int k=0; k<v*v; k++) adjugate_matrix[k] = 0;
printf("\n\nV = %d, E = %d\n", v, e);
printf("[Creating Graph]\n...\n");
GenerateRandGraphs(e, v, adjugate_matrix);
for(i = 0; i < v; i++) visited[i] = 0;
printf("\n[original] ");
start = omp_get_wtime(); 
basic_BFS(adjugate_matrix, v, visited);
end = omp_get_wtime(); 
printf("Work took %f seconds\n", end - start);
for(int k=1; k<8; k++){
printf("\n[OPENMP- %d threads]\n", k);
for(i = 0; i < v; i++) visited[i] = 0;
start = omp_get_wtime(); 
OMP_BFS(adjugate_matrix, v, visited, k);
end = omp_get_wtime(); 
printf("Work took %f seconds\n", end - start);
}
printf("\n[OPENMP- 8 threads]\n");
for(i = 0; i < v; i++) visited[i] = 0;
start = omp_get_wtime(); 
OMP_BFS(adjugate_matrix, v, visited, 8);
end = omp_get_wtime(); 
printf("[original] Work took %f seconds\n", end - start);
for(i = 0; i < v; i++) visited[i] = 0;
start = omp_get_wtime(); 
OMP_BFS_static(adjugate_matrix, v, visited, 8);
end = omp_get_wtime(); 
printf("[static] Work took %f seconds\n", end - start);
for(i = 0; i < v; i++) visited[i] = 0;
start = omp_get_wtime(); 
OMP_BFS_dynamic(adjugate_matrix, v, visited, 8);
end = omp_get_wtime(); 
printf("[dynamic] Work took %f seconds\n", end - start);
for(i = 0; i < v; i++) visited[i] = 0;
start = omp_get_wtime(); 
OMP_BFS_guided(adjugate_matrix, v, visited, 8);
end = omp_get_wtime(); 
printf("[guided] Work took %f seconds\n", end - start);
printf("\n");
printf("DONE!\n");
printf("---------------------------------------------------\n");
free(visited);
free(adjugate_matrix);
v = v+5000;
e = v*12;
}
}
