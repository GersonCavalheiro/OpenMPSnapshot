#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "cputils.h"
#include <omp.h>
typedef struct {
int y;
int x;
} Antena;
typedef struct {
int valor;
int x;
int y;
} Registro;
int num_hilos;		
void print_mapa(int ** mapa, int rows, int cols, Antena * a)
{
if(rows > 50 || cols > 30){
printf("Mapa muy grande para imprimir\n");
return;
}
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
printf("Mapa [%d,%d]\n",rows,cols);
for(int i=0; i < rows; i++){
for(int j=0; j < cols; j++)
{
int val = mapa[i][j];
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
int dist = abs(a.x - x) + abs(a.y - y);
return dist * dist;
}
void actualizar(int ** mapa, int rows, int cols, Antena * antena)
{
Antena miAntena = (* antena);
mapa[miAntena.y][miAntena.x] = 0;
for(int r = 1; r < (rows + cols); r++)
{
Antena arriba = { miAntena.y - r, miAntena.x };
Antena abajo = { miAntena.y + r, miAntena.x };
int distancia = manhattan(miAntena, arriba.y, arriba.x);
int flag = 0;
for(int delta = 0; delta <= r; delta++)
{
if( (arriba.y + delta >= 0) && (arriba.y + delta < rows) &&
(arriba.x + delta >= 0) && (arriba.x + delta < cols) &&
(mapa[arriba.y + delta][arriba.x + delta] > distancia) )
{
mapa[arriba.y + delta][arriba.x + delta] = distancia;
flag = 1;
}
if( (arriba.y + delta >= 0) && (arriba.y + delta < rows) &&
(arriba.x - delta >= 0) && (arriba.x - delta < cols) &&
(mapa[arriba.y + delta][arriba.x - delta] > distancia) )
{
mapa[arriba.y + delta][arriba.x - delta] = distancia;
flag = 1;
}
}
for(int delta = r-1; delta >= 0; delta--)
{
if( (abajo.y - delta >= 0) && (abajo.y - delta < rows) &&
(abajo.x + delta >= 0) && (abajo.x + delta < cols) &&
(mapa[abajo.y - delta][abajo.x + delta] > distancia) )
{
mapa[abajo.y - delta][abajo.x + delta] = distancia;
flag = 1;
}
if( (abajo.y - delta >= 0) && (abajo.y - delta < rows) &&
(abajo.x - delta >= 0) && (abajo.x - delta < cols) &&
(mapa[abajo.y - delta][abajo.x - delta] > distancia) )
{
mapa[abajo.y - delta][abajo.x - delta] = distancia;
flag = 1;
}
}
if(flag == 0) break;
}
}
void actualizar_primera_antena(int ** mapa, int rows, int cols, Antena * antena)			
{
Antena miAntena = (* antena);
mapa[miAntena.y][miAntena.x] = 0;
#pragma omp parallel for
for(int i=0; i < rows; i++) {
for(int j=0; j < cols; j++) {
mapa[i][j] = manhattan(miAntena,i,j);
}
}
}
Antena calcular_max(int ** mapa, int rows, int cols, Registro * registros)
{
int maximo=0, numHilo;
#pragma omp parallel for shared(mapa,registros) firstprivate(maximo) private(numHilo)
for(int i=0; i < rows; i++)
{
numHilo = omp_get_thread_num();
for(int j=0; j < cols; j++)
{
if(mapa[i][j] > maximo)
{
maximo = mapa[i][j];
registros[numHilo].x = i;
registros[numHilo].y = j;
registros[numHilo].valor = mapa[i][j];
}
}
}
maximo=0;
for(int i=0; i < num_hilos; i++)
if(registros[i].valor > registros[maximo].valor)
maximo = i;
Antena antena = { registros[maximo].x, registros[maximo].y };
return antena;
}
int main(int nargs, char ** vargs)
{
if(nargs < 7)
{
fprintf(stderr,"Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",vargs[0]);
return -1;
}
int rows = atoi(vargs[1]);
int cols = atoi(vargs[2]);
int distMax = atoi(vargs[3]);
int nAntenas = atoi(vargs[4]);
if( nAntenas<1 || nargs != (nAntenas*2+5) )
{
fprintf(stderr, "Error en la lista de antenas\n");
return -1;
}
printf("Calculando el número de antenas necesarias para cubrir un mapa de"
" (%d x %d)\ncon una distancia máxima no superior a %d "
"y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);
Antena * antenas = malloc(sizeof(Antena) * (size_t) nAntenas);
if( !antenas )
{
fprintf(stderr, "Error al reservar memoria para las antenas inicales\n");
return -1;
}
for(int i=0; i < nAntenas; i++)
{
antenas[i].x = atoi(vargs[5+i*2]);
antenas[i].y = atoi(vargs[6+i*2]);
if( antenas[i].y<0 || antenas[i].y>=rows || antenas[i].x<0 || antenas[i].x>=cols )
{
fprintf(stderr, "Antena #%d está fuera del mapa\n", i);
return -1;
}
}
double tiempo = cp_Wtime();
num_hilos = 0;
#pragma omp parallel shared(num_hilos)
{
#pragma omp single nowait
{
num_hilos = omp_get_num_threads();
}
}
Registro * registros = malloc(sizeof(Registro) * (size_t) (num_hilos));	
if( !registros )
{
fprintf(stderr, "Error al reservar memoria para los registros iniciales\n");
return -1;
}
int ** mapa = (int**)malloc(cols*sizeof(int*));
for(int i=0; i < rows; i++){
mapa[i] = (int*)malloc(rows*sizeof(int*));
}
actualizar_primera_antena(mapa, rows, cols, &antenas[0]);
for(int i=1; i < nAntenas; i++)
{
actualizar(mapa, rows, cols, &antenas[i]);
}
#ifdef DEBUG
print_mapa(mapa, rows, cols, NULL);
#endif
int nuevas = 0;		
while(1)
{
Antena antena = calcular_max(mapa, rows, cols, registros);
if (mapa[antena.y][antena.x] <= distMax) break;	
nuevas++;
actualizar(mapa, rows, cols, &antena);
}
#ifdef DEBUG
print_mapa(mapa, rows, cols, NULL);
#endif
tiempo = cp_Wtime() - tiempo;	
printf("Result: %d\n", nuevas);
printf("Time: %f\n", tiempo);
print_mapa(mapa, rows, cols, NULL);
return 0;
}
