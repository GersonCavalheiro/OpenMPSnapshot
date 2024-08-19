#include <stdio.h>
#include <omp.h>
#define NEDGES 6
#define NVERTEX 5
int main(int argc, char **argv)
{  int i,j;
int nedges = NEDGES, degree[NVERTEX]={0,0,0,0,0};
typedef struct {
int vertex1, vertex2;
}tipo_edge;
tipo_edge edge[NEDGES];
edge[0].vertex1 = 0;
edge[0].vertex2 = 1;
edge[1].vertex1 = 1;
edge[1].vertex2 = 2;
edge[2].vertex1 = 1;
edge[2].vertex2 = 4;
edge[3].vertex1 = 2;
edge[3].vertex2 = 4;
edge[4].vertex1 = 2;
edge[4].vertex2 = 3;
edge[5].vertex1 = 3;
edge[5].vertex2 = 4;
int tid = omp_get_thread_num();  
int n_threads = omp_get_num_threads(); 
omp_set_num_threads(4);
#pragma omp parallel for
for (j=0; j<nedges; j++){
#pragma omp atomic
degree[edge[j].vertex1]++;  
#pragma omp atomic
degree[edge[j].vertex2]++; 
}
for (i=0; i < NVERTEX; i++)
printf("Grau do vertice %d = %d \n",i, degree[i]);
}
