#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "defineg.h"
#include "fun.h"
float  elem[MAXE][NCAR];               
struct lista_grupos cluster_data[MAX_GRUPOS];   
float  enf[MAXE][TENF];                
struct  analisis prob_enf[TENF];       
int nclusters = 35;
int main (int argc, char *argv[]) {
float a[MAX_GRUPOS]; 
int popul[MAXE]; 
float cent[MAX_GRUPOS][NCAR]; 
int i, j, nelem, grupo, num, ind;
int fin = 0, num_ite = 0;
int convergencia_cont;
double sil, sil_old, diff;
FILE *fd;
struct timespec t1, t2;
struct timespec t11, t12, t17, t20, t21;
double texe, t_lec, t_clust, t_enf, t_escri;
if ((argc < 3) || (argc > 4)) {
printf("[!] ERROR:  gengrupos bd_muestras bd_enfermedades [num_elem]\n");
exit(-1);
}
setbuf(stdout, NULL);
printf("\n*** Ejecucion en paralelo ***\n\n");
clock_gettime(CLOCK_REALTIME, &t1);
clock_gettime(CLOCK_REALTIME, &t11);
fd = fopen(argv[1], "r");
if (fd == NULL) {
printf("[!] Error al abrir el fichero %s\n", argv[1]);
exit(-1);
}
fscanf(fd, "%d", &nelem);
if (argc == 4) nelem = atoi(argv[3]);    
for (i = 0; i < nelem; i++)
for (j = 0; j < NCAR; j++)
fscanf(fd, "%f", &(elem[i][j]));
fclose(fd);
fd = fopen(argv[2], "r");
if (fd == NULL) {
printf("[!] Error al abrir el fichero %s\n", argv[2]);
exit(-1);
}
for (i = 0; i < nelem; i++) {
for (j = 0; j < TENF; j++)
fscanf(fd, "%f", &(enf[i][j]));
}
fclose(fd);
clock_gettime(CLOCK_REALTIME, &t12);
t_lec = (t12.tv_sec - t11.tv_sec) + (t12.tv_nsec - t11.tv_nsec) / (double) 1e9;
convergencia_cont = 0;
sil_old = -1;
omp_set_num_threads(64);
while (nclusters < MAX_GRUPOS && convergencia_cont < 1) {
inicializar_centroides(cent);
num_ite = 0;
fin = 0;
while ((fin == 0) && (num_ite < MAXIT)) {
nearest_cluster(nelem, elem, cent, popul);
fin = nuevos_centroides(elem, cent, popul, nelem);
num_ite++;
}
#pragma omp parallel default(none) private(i, grupo), shared(cluster_data, popul, nclusters, nelem)
{
#pragma omp for
for (i = 0; i < nclusters; i++) cluster_data[i].nelems = 0;
#pragma omp for private(grupo, i)
for (i = 0; i < nelem; i++) {
grupo = popul[i];
#pragma omp critical
{
cluster_data[grupo].elem_index[cluster_data[grupo].nelems] = i;
cluster_data[grupo].nelems++;
}
}
}
sil = silhouette_simple(elem, cluster_data, cent, a);
diff = sil - sil_old;
if (diff < DELTA2) convergencia_cont++;
else convergencia_cont = 0;
sil_old = sil;
nclusters = nclusters + 10;
}
nclusters = nclusters - 10;
clock_gettime (CLOCK_REALTIME, &t17);
t_clust = (t17.tv_sec-t12.tv_sec) + (t17.tv_nsec-t12.tv_nsec)/(double)1e9;
clock_gettime(CLOCK_REALTIME, &t20);
analisis_enfermedades(cluster_data, enf, prob_enf);
clock_gettime(CLOCK_REALTIME, &t21);
t_enf = (t21.tv_sec - t20.tv_sec) + (t21.tv_nsec - t20.tv_nsec) / (double) 1e9;
fd = fopen("dbgen_s.out", "w");
if (fd == NULL) {
printf("[!] Error al abrir el fichero dbgen_out.s\n");
exit(-1);
}
fprintf(fd, ">> Centroides de los clusters\n\n");
for (i = 0; i < nclusters; i++) {
for (j = 0; j < NCAR; j++) fprintf(fd, "%7.3f", cent[i][j]);
fprintf(fd, "\n");
}
fprintf(fd, "\n\n>> Numero de clusteres: %d. Numero de elementos en cada cluster:\n\n", nclusters);
ind = 0;
for (i = 0; i < nclusters / 10; i++) {
for (j = 0; j < 10; j++) {
fprintf(fd, "%6d", cluster_data[ind].nelems);
ind++;
}
fprintf(fd, "\n");
}
for (i = ind; i < nclusters; i++) fprintf(fd, "%6d", cluster_data[i].nelems);
fprintf(fd, "\n");
fprintf(fd, "\n>> Densidad de los clusters: b[i]\n\n");
ind = 0;
for (i = 0; i < nclusters / 10; i++) {
for (j = 0; j < 10; j++) {
fprintf(fd, "%9.2f", a[ind]);
ind++;
}
fprintf(fd, "\n");
}
for(i=ind; i < nclusters; i++) fprintf(fd, "%9.2f", a[i]);
fprintf(fd, "\n");
fprintf (fd,"\n\n>> Analisis de enfermedades (medianas) en los grupos\n\n");
for (i=0; i<TENF; i++)
fprintf (fd,"Enfermedad: %2d - mmax: %4.2f (grupo %2d) - mmin: %4.2f (grupo %2d)\n",
i, prob_enf[i].mmax, prob_enf[i].gmax, prob_enf[i].mmin, prob_enf[i].gmin);
fprintf (fd,"\n\n");
fclose (fd);
printf ("\n>> Centroides 0, 20, 40...\n");
for (i=0; i < nclusters; i+=20) {
printf ("\n  cent%2d -- ", i);
for (j=0; j<NCAR; j++) printf ("%5.1f", cent[i][j]);
printf("\n");
}
printf ("\n>> Numero de clusteres: %d. Tamanno de los grupos:\n\n", nclusters);
ind = 0;
for (i=0; i < nclusters / 10; i++) {
for (j=0; j<10; j++){
printf ("%6d", cluster_data[ind].nelems);
ind++;
}
printf ("\n");
}
for(i=ind; i < nclusters; i++) printf("%6d", cluster_data[i].nelems);
printf("\n");
printf("\n>> Densidad de los clusters: b[i]\n\n");
ind = 0;
for (i=0; i < nclusters / 10; i++) {
for (j=0; j<10; j++){
printf("%9.2f", a[ind]);
ind++;
}
printf("\n");
}
for(i=ind; i < nclusters; i++) printf("%9.2f", a[i]);
printf("\n");
printf ("\n>> Analisis de enfermedades en los grupos\n\n");
for (i=0; i<TENF; i++)
printf ("Enfermedad: %2d - max: %4.2f (grupo %2d) - min: %4.2f (grupo %2d)\n",
i, prob_enf[i].mmax, prob_enf[i].gmax, prob_enf[i].mmin, prob_enf[i].gmin);
clock_gettime (CLOCK_REALTIME, &t2);
t_escri = (t2.tv_sec-t21.tv_sec) + (t2.tv_nsec-t21.tv_nsec)/(double)1e9;
texe = (t2.tv_sec-t1.tv_sec) + (t2.tv_nsec-t1.tv_nsec)/(double)1e9;
printf("\n");
printf("t_lec,%f\n", t_lec);
printf("t_clust,%f\n", t_clust);
printf("t_enf,%f\n", t_enf);
printf("t_escri,%f\n", t_escri);
printf("Texe,%f\n", texe);
}
