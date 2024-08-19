#include <math.h>
#include <float.h>
#include <omp.h>
#include "defineg.h"    
double gendist (float *elem1, float *elem2) {
int i;
double acum = 0;
for (i = 0; i < NCAR; i++) {
double res = elem1[i] - elem2[i];
acum += pow(res, 2);
}
return sqrt(acum);
}
void grupo_cercano (int nelem, float elem[][NCAR], float cent[][NCAR], int *popul) {
int ngrupo, i, j;
double adis, dmin;
#pragma omp parallel for private(i, j, adis, dmin, ngrupo) schedule(dynamic,2) num_threads(32)
for (i = 0; i < nelem; i++) {
dmin = DBL_MAX;
for (j = 0; j < NGRUPOS; j++) {
adis = gendist(elem[i], cent[j]); 
if (adis < dmin) {
dmin = adis;
ngrupo = j;
}
}
popul[i] = ngrupo;
}
}
void calcular_densidad (float elem[][NCAR], struct lista_grupos *listag, float *densidad) {
int i, j, k, nelem, actg, othg;
double acum, cont;
for (i = 0; i < NGRUPOS; i++) {
nelem = listag[i].nelemg;
if (nelem < 2) {
densidad[i] = 0;
}
else {
acum = 0.0;
cont = 0.0;
#pragma omp parallel for private(j, k, actg, othg) reduction(+ : acum, cont) schedule(dynamic,2) num_threads(32)
for (j = 0; j < nelem; j++) {
actg = listag[i].elemg[j];
for (k = j + 1; k < nelem; k++) {
othg = listag[i].elemg[k];
acum += gendist(elem[actg], elem[othg]);
cont += 1.0;
}
}
densidad[i] = (float) (acum / cont);
}
}
}
void analizar_enfermedades (struct lista_grupos *listag, float enf[][TENF], struct analisis *prob_enf) {
int i, j, k, actg, nelem, gmax, gmin;
float mediaact, acum, mediamin, mediamax;
for (i = 0; i < TENF; i++) {
mediamin = FLT_MAX, mediamax = FLT_MIN;
for (j = 0; j < NGRUPOS; j++) {
nelem = listag[j].nelemg;
acum = 0;
#pragma omp parallel for private(k, actg, mediaact) shared(mediamin, mediamax, gmin, gmax) reduction(+ : acum) schedule(static) num_threads(2)
for (k = 0; k < nelem; k++) {
actg = listag[j].elemg[k];
acum += enf[actg][i];
}
mediaact = acum / nelem;
if (mediaact < mediamin) {
mediamin = mediaact;
gmin = j;
} else if (mediaact >= mediamax) {
mediamax = mediaact;
gmax = j;
}
}
prob_enf[i].max = mediamax;
prob_enf[i].min = mediamin;
prob_enf[i].gmax = gmax;
prob_enf[i].gmin = gmin;
}
}