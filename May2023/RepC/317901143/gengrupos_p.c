#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "defineg.h"
#include "fun.h"
float  elem[MAXE][NCAR];               
struct lista_grupos listag[NGRUPOS];   
float  enf[MAXE][TENF];                
struct  analisis prob_enf[TENF];       
int main (int argc, char *argv[]) {
float   cent[NGRUPOS][NCAR], newcent[NGRUPOS][NCAR];    
float   densidad[NGRUPOS];	                            
int     popul[MAXE];                                    
double  additions[NGRUPOS][NCAR+1];
int     i, j, nelem, grupo, num;
int     fin = 0, num_ite = 0;
double  discent;
FILE   *fd;
struct timespec  t1, t2, t3;
double tlec, tclu, tord, tden, tenf, tesc, texe;
if ((argc < 3)  || (argc > 4)) {
printf("ERROR:  gengrupos bd_muestras bd_enfermedades [num_elem]\n");
exit(-1);
}
printf("\n >> Ejecucion paralela\n");
clock_gettime(CLOCK_REALTIME, &t1);
clock_gettime(CLOCK_REALTIME, &t2);
fd = fopen(argv[1], "r");
if (fd == NULL) {
printf("Error al abrir el fichero %s\n", argv[1]);
exit(-1);
}
fscanf(fd, "%d", &nelem);
if (argc == 4) {
nelem = atoi(argv[3]);
}
for (i = 0; i < nelem; i++) {
for (j = 0; j < NCAR; j++) {
fscanf(fd, "%f", &(elem[i][j]));
}
}
fclose(fd);
fd = fopen (argv[2], "r");
if (fd == NULL) {
printf("Error al abrir el fichero %s\n", argv[2]);
exit(-1);
}
for (i = 0; i < nelem; i++) {
for (j = 0; j < TENF; j++)
fscanf(fd, "%f", &(enf[i][j]));
}
fclose(fd);
clock_gettime (CLOCK_REALTIME, &t3);
tlec = (t3.tv_sec-t2.tv_sec) + (t3.tv_nsec-t2.tv_nsec)/(double)1e9;
clock_gettime (CLOCK_REALTIME, &t2);
srand (147);
for (i = 0; i < NGRUPOS; i++) {
for (j = 0; j < NCAR / 2; j++) {
cent[i][j] = (rand() % 10000) / 100.0;
cent[i][j + (NCAR / 2)] = cent[i][j];
}
}
num_ite = 0; fin = 0;
while ((fin == 0) && (num_ite < MAXIT)) {
grupo_cercano (nelem, elem, cent, popul);
#pragma omp parallel num_threads(32)
{
#pragma omp for private(i, j)  schedule(dynamic,2)
for (i = 0; i < NGRUPOS; i++) {
for (j = 0; j < NCAR + 1; j++) {
additions[i][j] = 0.0;
}
}
#pragma omp single
{
for (i = 0; i < nelem; i++) {
for (j = 0; j < NCAR; j++) {
additions[popul[i]][j] += elem[i][j];
}
additions[popul[i]][NCAR]++;
}
fin = 1;
}
#pragma omp for private(i, j, discent) schedule(dynamic,2)
for (i = 0; i < NGRUPOS; i++) {
if (additions[i][NCAR] > 0) {
for (j = 0; j < NCAR; j++) {
newcent[i][j] = additions[i][j] / additions[i][NCAR];
}
discent = gendist (&newcent[i][0], &cent[i][0]);
if (discent > DELTA) {
fin = 0;
}
for (j = 0; j < NCAR; j++) {
cent[i][j] = newcent[i][j];
}
}
}
#pragma omp single
{
num_ite++;
}
}
} 
clock_gettime(CLOCK_REALTIME, &t3);
tclu = (t3.tv_sec-t2.tv_sec) + (t3.tv_nsec-t2.tv_nsec)/(double)1e9;
clock_gettime(CLOCK_REALTIME, &t2);
#pragma omp parallel for private(i) schedule(static,2) num_threads(8)
for (i = 0; i < NGRUPOS; i++) {
listag[i].nelemg = 0;
}
for (i = 0; i < nelem; i++) {
grupo = popul[i];
num = listag[grupo].nelemg;
listag[grupo].elemg[num] = i;	
listag[grupo].nelemg++;
}
clock_gettime(CLOCK_REALTIME, &t3);
tord = (t3.tv_sec-t2.tv_sec) + (t3.tv_nsec-t2.tv_nsec)/(double)1e9;
clock_gettime(CLOCK_REALTIME, &t2);
calcular_densidad (elem, listag, densidad);
clock_gettime(CLOCK_REALTIME, &t3);
tden = (t3.tv_sec-t2.tv_sec) + (t3.tv_nsec-t2.tv_nsec)/(double)1e9;
clock_gettime(CLOCK_REALTIME, &t2);
analizar_enfermedades (listag, enf, prob_enf);
clock_gettime(CLOCK_REALTIME, &t3);
tenf = (t3.tv_sec-t2.tv_sec) + (t3.tv_nsec-t2.tv_nsec)/(double)1e9;
clock_gettime(CLOCK_REALTIME, &t2);
fd = fopen ("dbgen_p.out", "w");
if (fd == NULL) {
printf ("Error al abrir el fichero dbgen_p.out\n");
exit (-1);
}
fprintf (fd,">> Centroides de los clusters\n\n");
for (i=0; i<NGRUPOS; i++) {
for (j=0; j<NCAR; j++) fprintf (fd, "%7.3f", cent[i][j]);
fprintf (fd,"\n");
}
fprintf (fd,"\n\n>> Numero de elementos de cada cluster y densidad del cluster\n\n");
for (i=0; i<NGRUPOS; i++) {
fprintf(fd, " %6d  %.3f \n", listag[i].nelemg, densidad[i]);
}
fprintf (fd,"\n\n>> Analisis de enfermedades en los grupos\n\n");
for (i=0; i<TENF; i++) {
fprintf(fd, "Enfermedad: %2d - max: %4.2f (grupo %2d) - min: %4.2f (grupo %2d)\n",
i, prob_enf[i].max, prob_enf[i].gmax, prob_enf[i].min, prob_enf[i].gmin);
}
fclose (fd);
clock_gettime(CLOCK_REALTIME, &t3);
tenf = (t3.tv_sec-t2.tv_sec) + (t3.tv_nsec-t2.tv_nsec)/(double)1e9;
texe = (t3.tv_sec-t1.tv_sec) + (t3.tv_nsec-t1.tv_nsec)/(double)1e9;
printf ("\n>> Centroides 0, 40 y 80, y su valor de densidad\n ");
for (i = 0; i < NGRUPOS; i+=40) {
printf("\n  cent%2d -- ", i);
for (j = 0; j < NCAR; j++) {
printf("%5.1f", cent[i][j]);
}
printf("\n          %5.6f\n", densidad[i]);
}
printf("\n>> Tamano de los grupos \n");
for (i = 0; i < 10; i++) {
for (j = 0; j < 10; j++) {
printf("%7d", listag[10*i+j].nelemg);
}
printf("\n");
}
printf ("\n>> Analisis de enfermedades en los grupos\n");
for (i = 0; i < TENF; i++) {
printf("Enfermedad: %2d - max: %4.2f (grupo %2d) - min: %4.2f (grupo %2d)\n",
i, prob_enf[i].max, prob_enf[i].gmax, prob_enf[i].min, prob_enf[i].gmin);
}
printf ("\n >> Numero de iteraciones: %d\n", num_ite);
printf ("\n >> Tiempos de ejecución: ");
printf ("\n    - Lectura: %11.3f s", tlec);
printf ("\n    - Clustering: %8.3f s", tclu);
printf ("\n    - Ordenación: %8.3f s", tord);
printf ("\n    - Densidad: %10.3f s", tden);
printf ("\n    - Enfermedades: %6.3f s", tenf);
printf ("\n    - Escritura: %9.3f s", tesc);
printf ("\n    - Total: %13.3f s\n\n", texe);
return 0;
}
