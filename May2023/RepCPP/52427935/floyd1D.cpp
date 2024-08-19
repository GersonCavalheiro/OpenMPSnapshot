#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string.h>
#include <omp.h>
#include "Graph.h"


using namespace std;

int main(int argc, char *argv[]) {
int procs, chunk, nverts, i, j, k, ik, kj, ij, *M;
double t;

switch(argc) {
case 3: 
procs = atoi(argv[2]);
break;
case 2: 
procs = omp_get_num_procs();
break;
default: 
cerr << "Sintaxis: " << argv[0] << "<archivo de grafo> <num procs>" << endl;
return(-1);
}
omp_set_num_threads(procs);

Graph G;
G.lee(argv[1]);	
#ifdef PRINT_ALL
cout << "El grafo de entrada es:" << endl;
G.imprime();
#endif

nverts = G.vertices;                                  
procs > nverts ? chunk = 1 : chunk = nverts / procs;  

#ifdef PRINT_ALL
cout << endl;
cout << "El tamaño del problema es: " << nverts << endl;
cout << "El número de procesos es: " << procs << endl;
cout << "El tamaño de bloque es: " << chunk << endl;
#endif

M = (int *) malloc(nverts * nverts * sizeof(int));    
G.copia_matriz(M);                                    
int filK[nverts];

t = omp_get_wtime();
#pragma omp parallel private(k, i, j, ik, ij, kj, filK)
{
for (k = 0; k < nverts; k++) {
#pragma omp single copyprivate(filK)
{
for (i = 0; i < nverts; i++) {
filK[i] = M[k * nverts + i];
}
}
#pragma omp for schedule(static, chunk)  
for (i = 0; i < nverts; i++) {
ik = i * nverts + k;
for (j = 0; j < nverts; j++) {
if (i != j && i != k && j != k) {
kj = k * nverts + j;
ij = i * nverts + j;
M[ij] = min(M[ik] + filK[j], M[ij]);
}
}
}
}
}
t = omp_get_wtime() - t;

G.lee_matriz(M);  

#ifdef PRINT_ALL
cout << endl << "El grafo con las distancias de los caminos más cortos es:" << endl;
G.imprime();
cout << "Tiempo gastado = " << t << endl << endl;
#else
cout << t << endl;
#endif

delete[] M;

return(0);
}
