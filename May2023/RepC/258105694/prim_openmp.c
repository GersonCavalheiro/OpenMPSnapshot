#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
void prim()
{
int numVertices; 
int numEdges;   
int root = 0;   
int i,j,k;    
int* key,* parent;
int** adj,**edges;
printf("Ingrese numero de vertices: ");
scanf("%d", &numVertices);
printf("Ingrese numero de aristas: ");
scanf("%d", &numEdges);
key = (int *) malloc(sizeof(int)*numVertices);
parent = (int *) malloc(sizeof(int)*numVertices);
edges = (int **)malloc(sizeof(int*)*numEdges);
for(i = 0; i < numEdges; i++)
edges[i] = (int *) malloc(sizeof(int)*3);
adj = (int **) malloc(sizeof(int *)*numVertices);
for(i = 0; i < numVertices; i++)
adj[i] = (int *) malloc(sizeof(int)*numVertices);
for(i = 0; i < numVertices; i++){
for(j = 0; j < numVertices; j++){
if(i == j)
adj[i][j] = 0;
else
adj[i][j] = INT_MAX;
}
}
printf("Ingrese las aristas y su peso (v1, v2, weight): \n");
for(i = 0; i < numEdges; i++)
{
scanf("%d %d %d", &edges[i][0], &edges[i][1], &edges[i][2]);
adj[edges[i][0]][edges[i][1]] = edges[i][2];
adj[edges[i][1]][edges[i][0]] = edges[i][2];
}  
for(i = 0; i < numVertices; i++){
for(j = 0; j < numVertices; j++){
printf("%d ",adj[i][j]);
}
printf("\n");
}
#pragma parallel for
for(i = 0; i < numVertices; i++){
key[i] = INT_MAX;
parent[i] = -1;
}
key[root] = 0;
int * used = (int *) malloc(sizeof(int)*numVertices);
int numUnused = numVertices;
for(i = 0; i < numVertices; i++)
used[i] = 0;
int u = root;
int closestIdx = root;    
int closestVal = INT_MAX; 
while(numUnused > 0){
closestVal = INT_MAX;
#pragma parallel for
for(i = 0; i < numVertices; i++){
if(used[i] == 0 && key[i] < closestVal){
closestIdx = i;
closestVal = key[i];
}
}
printf("vertice visitado actual es %d\n", u);
u = closestIdx;
used[u] = 1;
numUnused--;
#pragma parallel for
for(i = 0; i < numVertices; i++){
if(used[i] == 0 && adj[u][i] > 0 && adj[u][i] < key[i]){
parent[i] = u;
key[i] = adj[u][i];
printf("key[%d] is %d\n", i, adj[u][i]);
}
}
}
#pragma parallel for
for(i = 0; i < numVertices; i++)
printf("vertice: %d parent: %d\n", i, parent[i]);
}
int main(int argc, char* argv){
prim();
return 0;
}
