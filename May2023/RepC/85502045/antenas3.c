#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "cputils.h"
#include <omp.h>
#define NUM_HILOS 11
typedef struct {
int y;
int x;
} Antena;
typedef struct {
int valor;
int x;
int y;
} Registro;
#define m(y,x) mapa[ (y * cols) + x ]
void print_mapa(int * mapa, int rows, int cols, Antena * a)
{
if(rows > 50 || cols > 30){
printf("Mapa muy grande para imprimir\n");
return;
}
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
printf("Mapa [%d,%d]\n",rows,cols);
for(int i=0; i<rows; i++){
for(int j=0; j<cols; j++)
{
int val = m(i,j);
if(val == 0){
if(a != NULL && a->x == j && a->y == i){
printf( ANSI_COLOR_RED "   A"  ANSI_COLOR_RESET);
} else { 
printf( ANSI_COLOR_GREEN "   A"  ANSI_COLOR_RESET);
}
} else {
printf("%4d", val);
}
}
printf("\n");
}
printf("\n");
}
int manhattan(Antena a, int y, int x)
{
int dist = abs(a.x -x) + abs(a.y - y);
return dist * dist;
}
void actualizar(int * mapa, int rows, int cols, Antena antena)
{
#pragma omp parallel
{
#pragma omp single
m(antena.y,antena.x) = 0;
#pragma omp for
for(int i=0; i<rows; i++){
for(int j=0; j<cols; j++){
int nuevadist = manhattan(antena,i,j);
if(nuevadist < m(i,j)){
m(i,j) = nuevadist;
}
} 
} 
}
}
void actualizar_primera_antena(int * mapa, int rows, int cols, Antena antena){
#pragma omp parallel
{	
#pragma omp single
m(antena.y,antena.x) = 0;
#pragma omp for
for(int i=0; i<rows; i++){
for(int j=0; j<cols; j++){
m(i,j) = manhattan(antena,i,j);
}
}
}
}
int calcular_max(int * mapa, int rows, int cols, Registro * registros)
{
int maximo=0, ii=0, jj=0, i, j;
#pragma omp parallel for shared(mapa,registros) firstprivate(ii,jj,maximo) private(i,j)
for(i=0; i < rows; i++)
{
for(j=0; j < cols; j++)
{
if(m(i,j) > maximo)
{
maximo = m(i,j);
ii = i;
jj = j;
}
}
registros[omp_get_thread_num()].x = ii;
registros[omp_get_thread_num()].y = jj;
registros[omp_get_thread_num()].valor = maximo;
}
maximo=0;
for(i=0; i < NUM_HILOS; i++)
{
if(registros[i].valor > maximo)
{
maximo = registros[i].valor;
registros[NUM_HILOS].valor = maximo;
registros[NUM_HILOS].x = registros[i].x;
registros[NUM_HILOS].y = registros[i].y;
}
}
return registros[NUM_HILOS].valor;
}
Antena nueva_antena(Registro * registros, int rows, int cols, int min)
{	
int i=registros[NUM_HILOS].x;
int j=registros[NUM_HILOS].y;
Antena antena = {i,j};
return antena;
}
int main(int nargs, char ** vargs)
{
#if defined(_OPENMP)
#pragma omp parallel
printf("Soy el hilo %d\n",omp_get_thread_num());
#else
printf("Versión secuencial\n");
#endif
if(nargs < 7)
{
fprintf(stderr,"Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",vargs[0]);
return -1;
}
int rows = atoi(vargs[1]);
int cols = atoi(vargs[2]);
int distMax = atoi(vargs[3]);
int nAntenas = atoi(vargs[4]);
if( nAntenas<1 || nargs != (nAntenas*2+5) ){
fprintf(stderr, "Error en la lista de antenas\n");
return -1;
}
printf("Calculando el número de antenas necesarias para cubrir un mapa de"
" (%d x %d)\ncon una distancia máxima no superior a %d "
"y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);
Antena * antenas = malloc(sizeof(Antena) * (size_t) nAntenas);
if(!antenas){
fprintf(stderr, "Error al reservar memoria para las antenas inicales\n");
return -1;
}
for(int i=0; i < nAntenas; i++){
antenas[i].x = atoi(vargs[5+i*2]);
antenas[i].y = atoi(vargs[6+i*2]);
if( antenas[i].y<0 || antenas[i].y>=rows || antenas[i].x<0 || antenas[i].x>=cols )
{
fprintf(stderr, "Antena #%d está fuera del mapa\n", i);
return -1;
}
}
Registro * registros = malloc(sizeof(Registro) * (size_t) (NUM_HILOS+1));	
if(!registros){
fprintf(stderr, "Error al reservar memoria para los registros iniciales\n");
return -1;
}
double tiempo = cp_Wtime();
int * mapa = malloc((size_t) (rows*cols) * sizeof(int) );
actualizar_primera_antena(mapa, rows, cols, antenas[0]);
for(int i=1; i<nAntenas; i++)
{
actualizar(mapa, rows, cols, antenas[i]);
}
#ifdef DEBUG
print_mapa(mapa, rows, cols, NULL);
#endif
int nuevas = 0;
while(1)
{
int max = calcular_max(mapa, rows, cols, registros);
if (max <= distMax) break;	
nuevas++;
Antena antena = nueva_antena(registros, rows, cols, max);
actualizar(mapa, rows, cols, antena);
}
#ifdef DEBUG
print_mapa(mapa, rows, cols, NULL);
#endif
tiempo = cp_Wtime() - tiempo;	
printf("Result: %d\n",nuevas);
printf("Time: %f\n",tiempo);
print_mapa(mapa,rows,cols,NULL);
return 0;
}
