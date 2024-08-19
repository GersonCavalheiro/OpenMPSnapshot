#include <math.h>
#include <float.h> 
#include <stdlib.h>
#include <stdio.h>
#include "defineg.h"           
#include <omp.h>
void swap(float *a, float *b) {
float t = *a;
*a = *b;
*b = t;
}
int partition(float array[], int low, int high) {
float pivot = array[high];
int i = (low - 1);
for (int j = low; j < high; j++) {
if (array[j] <= pivot) {
i++;
swap(&array[i], &array[j]);
}
}
swap(&array[i + 1], &array[high]);
return (i + 1);
}
void quickSort(float array[], int low, int high) {
while (low < high) {
int pivotIndex = partition(array, low, high);
quickSort(array, low, pivotIndex - 1);
low = pivotIndex + 1;
}
}
float sort_and_median(int n, float* disease_data)
{
quickSort(disease_data, 0, n - 1);
return disease_data[n/2];
}
double geneticdist(float *elem1, float *elem2)
{
double total = 0.0f;
for(int i = 0; i < NCAR; i++)
total += pow((double)(elem2[i] - elem1[i]), 2);
return sqrt(total);
}
void nearest_cluster(int nelem, float elem[][NCAR], float cent[][NCAR], int *samples)
{
double dist, mindist;
{
#pragma omp parallel for default(none) shared(nelem, elem, cent, samples, nclusters) private(dist, mindist)
for(int i = 0; i < nelem; i++){
mindist = DBL_MAX;
for(int k = 0; k < nclusters; k++) {
dist = geneticdist(elem[i], cent[k]);
if(dist < mindist){
mindist = dist;
samples[i] = k;
}
}
}
}
}
double silhouette_simple(float samples[][NCAR], struct lista_grupos *cluster_data, float centroids[][NCAR], float a[]) {
float b[nclusters];
float narista = 0;
double tmp = 0;
#pragma omp parallel default(none) shared(nclusters, b, a, cluster_data, samples, centroids, tmp) firstprivate(narista)
{
#pragma omp for
for (int k = 0; k < nclusters; k++) b[k] = 0.0f;
#pragma omp for
for (int k = 0; k < MAX_GRUPOS; k++) a[k] = 0.0f;
#pragma omp for nowait reduction(+: tmp)
for (int k = 0; k < nclusters; k++) {
tmp = 0;
for (int i = 0; i < cluster_data[k].nelems; i++) {
for (int j = i + 1; j < cluster_data[k].nelems; j++) {
tmp += geneticdist(samples[cluster_data[k].elem_index[i]],
samples[cluster_data[k].elem_index[j]]);
}
}
narista = ((float) (cluster_data[k].nelems * (cluster_data[k].nelems - 1)) / 2);
a[k] = cluster_data[k].nelems <= 1 ? 0 : (float) (tmp / narista);
}
#pragma omp for nowait
for (int k = 0; k < nclusters; k++) {
for (int j = 0; j < nclusters; j++) {
b[k] += (float) geneticdist(centroids[k], centroids[j]); 
}
b[k] /= (float) (nclusters - 1);
}
}
float max, sil = 0.0f;
for (int k = 0; k < nclusters; k++) {
max = a[k] >= b[k] ? a[k] : b[k];
if (max != 0.0f)
sil += (b[k] - a[k]) / max;
}
return (double)(sil / (float)nclusters);
}
void analisis_enfermedades(struct lista_grupos *cluster_data, float enf[][TENF], struct analisis *analysis)
{
int cluster_size;
float median;
float *disease_data;
#pragma omp parallel default(none) shared (analysis, nclusters, cluster_data, enf)
{
#pragma omp for nowait
for(int i = 0; i < TENF; i++){
analysis[i].mmin = FLT_MAX;
analysis[i].mmax = 0.0f;
}
#pragma omp for nowait private(cluster_size, disease_data, median) schedule(dynamic)
for(int k = 0; k < nclusters; k++){
cluster_size = cluster_data[k].nelems;
for(int j = 0; j < TENF; j++){
disease_data = malloc(sizeof(float) * cluster_size);
for(int i = 0; i < cluster_size; i++) disease_data[i] = 0.0f;
for(int i = 0; i < cluster_size; i++)
disease_data[i] = enf[cluster_data[k].elem_index[i]][j];
median = sort_and_median(cluster_size, disease_data);
if ((median > 0 && median < analysis[j].mmin) || ((median == analysis[j].mmin) && (k < analysis[j].gmin))){
analysis[j].mmin = median;
analysis[j].gmin = k;
}
if ((median > analysis[j].mmax) || ((median == analysis[j].mmax) && (k < analysis[j].gmax))){
analysis[j].mmax = median;
analysis[j].gmax = k;
}
free(disease_data);
}
}
}
}
void inicializar_centroides(float cent[][NCAR]){
int i, j;
srand (147);
for (i=0; i < nclusters; i++)
for (j=0; j<NCAR/2; j++){
cent[i][j] = (rand() % 10000) / 100.0;
cent[i][j+(NCAR/2)] = cent[i][j];
}
}
int nuevos_centroides(float elem[][NCAR], float cent[][NCAR], int samples[], int nelem){
int i, j, fin;
double discent;
double additions[nclusters][NCAR + 1];
float newcent[nclusters][NCAR];
#pragma omp parallel default(none) shared(nclusters, nelem, samples, elem, additions, fin, newcent, cent) private(i, j, discent)
{
#pragma omp for
for (i = 0; i < nclusters; i++)
for (j = 0; j < NCAR + 1; j++)
additions[i][j] = 0.0;
#pragma omp for nowait reduction(+: additions[:nclusters][:NCAR+1])
for (i = 0; i < nelem; i++) {
for (j = 0; j < NCAR; j++)
additions[samples[i]][j] += elem[i][j];
additions[samples[i]][NCAR]++;
}
#pragma omp single
fin = 1;
#pragma omp for
for (i = 0; i < nclusters; i++) {
if (additions[i][NCAR] > 0) {
for (j = 0; j < NCAR; j++)
newcent[i][j] = (float) (additions[i][j] / additions[i][NCAR]);
discent = geneticdist(&newcent[i][0], &cent[i][0]);
if (discent > DELTA1)
fin = 0;  
for (j = 0; j < NCAR; j++)
cent[i][j] = newcent[i][j];
}
}
}
return fin;
}