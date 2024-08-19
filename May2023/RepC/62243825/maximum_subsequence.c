#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <string.h>
struct tablo {
int * tab;
int size;
};
void printArray(struct tablo * tmp) {
printf("---- Array of size %i ---- \n", tmp->size);
int size = tmp->size;
int i;
for (i = 0; i < size; ++i) {
printf("%i ", tmp->tab[i]);
}
printf("\n");
}
struct tablo * allocateTablo(int size) {
struct tablo * tmp = malloc(sizeof(struct tablo));
tmp->size = size;
tmp->tab = malloc(size*sizeof(int));
int i;
#pragma omp parallel for
for(i = 0; i < tmp->size; i++){
tmp->tab[i] = 0;
}
return tmp;
}
void psum_montee(struct tablo * source, struct tablo * destination) {
int i;
#pragma omp parallel for
for(i = 0; i < source->size; i++){
destination->tab[destination->size / 2 + i] = source->tab[i];
}
int l, j;
for(l = log2(source->size); l > 0; l--){
#pragma omp parallel for
for(j = 1 << (l-1); j <= (1 << l) - 1; j++){
destination->tab[j] = destination->tab[2*j] + destination->tab[2*j+1];
}
}
}
void psum_descente(struct tablo * a, struct tablo * b) {
b->tab[1] = 0;
int l, j;
for(l = 2; l <= log2(a->size); l++){
#pragma omp parallel for
for(j = 1 << (l-1); j <= (1 << l) - 1;  j++){
if(j % 2 == 0){
b->tab[j] = b->tab[j/2];
} else {
b->tab[j] = b->tab[j/2] + a->tab[j-1];
}
}
}
}
void psum_final(struct tablo * a, struct tablo *b) {
int j;
int size = log2(a->size);
#pragma omp parallel for
for(j = 1 << (size - 1); j <= (1 << size) - 1; j++){
b->tab[j] = b->tab[j] + a->tab[j];
}
}
void ssum_montee(struct tablo * source, struct tablo * destination) {
int i;
#pragma omp parallel for
for(i = 0; i < source->size; i++){
destination->tab[destination->size - i - 1] = source->tab[i];
}
int l, j;
for(l = log2(source->size); l > 0; l--){
#pragma omp parallel for
for(j = 1 << (l-1); j <= (1 << l) - 1; j++){
destination->tab[j] = destination->tab[2*j] + destination->tab[2*j+1];
}
}
}
void ssum_descente(struct tablo * a, struct tablo * b) {
b->tab[1] = 0;
int l, j;
for(l = 2; l <= log2(a->size); l++){
#pragma omp parallel for
for(j = 1 << (l-1); j <= (1 << l) - 1;  j++){
if(j % 2 == 0){
b->tab[j] = b->tab[j/2];
} else {
b->tab[j] = b->tab[j/2] + a->tab[j-1];
}
}
}
}
void ssum_final(struct tablo * a, struct tablo *b) {
int j;
int size = log2(a->size);
#pragma omp parallel for
for(j = 1 << (size - 1); j <= (1 << size) - 1; j++){
b->tab[j] = b->tab[j] + a->tab[j];
}
}
void pmax_montee(struct tablo * source, struct tablo * destination) {
int i;
#pragma omp parallel for
for(i = 0; i < source->size/2; i++){
destination->tab[destination->size / 2 + i] = source->tab[source->size - i - 1];
}
int l, j;
for(l = log2(source->size/2); l > 0; l--){
#pragma omp parallel for
for(j = 1 << (l-1); j <= (1 << l) - 1; j++){
destination->tab[j] = fmax(destination->tab[2*j], destination->tab[2*j+1]);
}
}
}
void pmax_descente(struct tablo * a, struct tablo * b) {
b->tab[1] = INT_MIN;
int l, j;
for(l = 2; l <= log2(a->size); l++){
#pragma omp parallel for
for(j = 1<< (l-1); j <= (1 << l) - 1;  j++){
if(j % 2 == 0){
b->tab[j] = b->tab[j/2];
} else {
b->tab[j] = fmax(b->tab[j/2], a->tab[j-1]);
}
}
}
}
void pmax_final(struct tablo * a, struct tablo *b) {
int j;
int size = log2(a->size);
#pragma omp parallel for
for(j = 1 << (size - 1); j <= (1 << size) - 1; j++){
b->tab[j] = fmax(b->tab[j], a->tab[j]);
}
}
void smax_montee(struct tablo * source, struct tablo * destination) {
int i;
#pragma omp parallel for
for(i = 0; i < source->size/2; i++){
destination->tab[destination->size-i-1] = source->tab[source->size/2+i];
}
int l, j;
for(l = log2(source->size/2); l > 0; l--){
#pragma omp parallel for
for(j = 1 << (l-1); j <= (1 << l) - 1; j++){
destination->tab[j] = fmax(destination->tab[2*j], destination->tab[2*j+1]);
}
}
}
void smax_descente(struct tablo * a, struct tablo * b) {
b->tab[1] = INT_MIN;
int l, j;
for(l = 2; l <= log2(a->size); l++){
#pragma omp parallel for
for(j = 1 << (l-1); j <= (1 << l) - 1;  j++){
if(j % 2 == 0){
b->tab[j] = b->tab[j/2];
} else {
b->tab[j] = fmax(b->tab[j/2], a->tab[j-1]);
}
}
}
}
void smax_final(struct tablo * a, struct tablo *b) {
int j;
int size = log2(a->size);
#pragma omp parallel for
for(j = 1 << (size - 1); j <= (1 << size) - 1; j++){
b->tab[j] = fmax(b->tab[j], a->tab[j]);
}
}
void make_max(struct tablo * q, struct tablo * pmax, struct tablo * smax, struct tablo * ssum, struct tablo * psum, struct tablo * m){
int i;
struct tablo * ms = allocateTablo(q->size);
struct tablo * mp = allocateTablo(q->size);
#pragma omp parallel for
for(i = 0; i < q->size; i++){
ms->tab[i] = pmax->tab[pmax->size/2+i] - ssum->tab[ssum->size-i-1];
mp->tab[i] = smax->tab[smax->size-i-1] - psum->tab[psum->size/2+i];
m->tab[i] = ms->tab[i] + mp->tab[i] + q->tab[i];
}
free(ms->tab);
free(ms);
free(mp->tab);
free(mp);
}
void createArray(struct tablo * s, char * file){
FILE * fp;
int value;
int i = 0;
int size = 0;
fp = fopen(file, "r");
fseek(fp, 0L, SEEK_END);
size = ftell(fp);
fseek(fp, 0L, SEEK_SET);
s->tab = malloc(sizeof(int) * size/2);
rewind(fp);
while(fscanf(fp, "%d", &value) == 1){
s->tab[i] = value;
i++;
}
s->size = i;
fclose(fp);
}
void find_max(struct tablo * m, int * maxIndex, int * minIndex){
int i;
int value = INT_MIN;
int value_min = INT_MAX;
int value_max = INT_MIN;
int * maxIndexA = malloc(m->size*sizeof(int));
int * minIndexA = malloc(m->size*sizeof(int));
#pragma omp parallel for
for(i = 0; i < m->size; i++){
maxIndexA[i] = INT_MIN;
minIndexA[i] = INT_MAX;
}
#pragma omp parallel for reduction(max: value)
for(i = 0; i < m->size; i++){
if(m->tab[i] > value){
value = m->tab[i];
minIndexA[i] = i;
maxIndexA[i] = i;
} else if(m->tab[i] == value){
maxIndexA[i] = i;
}
}
#pragma omp parallel for reduction(max: value_max) reduction(min: value_min)
for(i = 0; i < m->size; i++){
if((maxIndexA[i] > value_max) && (value == m->tab[maxIndexA[i]])){
value_max = maxIndexA[i];
}
if((minIndexA[i] < value_min) && (value == m->tab[minIndexA[i]])){
value_min = minIndexA[i];
}
}
*maxIndex = value_max;
*minIndex = value_min;
free(maxIndexA);
free(minIndexA);
}
int main(int argc, char **argv) {
struct tablo source;
createArray(&source, argv[1]);
#ifdef DEBUG
printf("SOURCE\n");
printArray(&source);
printf("PSUM\n");
#endif
struct tablo * tmp = allocateTablo(source.size*2);
psum_montee(&source, tmp);
#ifdef DEBUG
printArray(tmp);
#endif
struct tablo * psum = allocateTablo(source.size*2);
psum_descente(tmp, psum);
#ifdef DEBUG
printArray(psum);
#endif
psum_final(tmp,psum);
#ifdef DEBUG
printArray(psum);
#endif
#ifdef DEBUG
printf("\nSSUM\n");
#endif
ssum_montee(&source, tmp);
#ifdef DEBUG
printArray(tmp);
#endif
struct tablo * ssum = allocateTablo(source.size*2);
ssum_descente(tmp, ssum);
#ifdef DEBUG
printArray(ssum);
#endif
ssum_final(tmp, ssum);
#ifdef DEBUG
printArray(ssum);
#endif
free(tmp->tab);
free(tmp);
#ifdef DEBUG
printf("\nSMAX\n");
#endif
struct tablo * tmp2 = allocateTablo(source.size*2);
smax_montee(psum, tmp2);
#ifdef DEBUG
printArray(tmp2);
#endif
struct tablo * smax = allocateTablo(source.size*2);
smax_descente(tmp2, smax);
#ifdef DEBUG
printArray(smax);
#endif
smax_final(tmp2, smax);
#ifdef DEBUG
printArray(smax);
#endif
#ifdef DEBUG
printf("\nPMAX\n");
#endif
tmp2 = allocateTablo(source.size*2);
pmax_montee(ssum, tmp2);
#ifdef DEBUG
printArray(tmp2);
#endif
struct tablo * pmax = allocateTablo(source.size*2);
pmax_descente(tmp2, pmax);
#ifdef DEBUG
printArray(pmax);
#endif
pmax_final(tmp2, pmax);
#ifdef DEBUG
printArray(pmax);
#endif
free(tmp2->tab);
free(tmp2);
#ifdef DEBUG
printf("\nMAX\n");
#endif
struct tablo * m = allocateTablo(source.size);
make_max(&source, pmax, smax, ssum, psum, m);
#ifdef DEBUG
printArray(m);
printf("\nRESULT\n");
#endif
free(pmax->tab);
free(pmax);
free(smax->tab);
free(smax);
free(ssum->tab);
free(ssum);
free(psum->tab);
free(psum);
int i;
int maxIndex = -1, minIndex = -1;
find_max(m, &maxIndex, &minIndex);
printf("%d", m->tab[minIndex]);
for(i = minIndex; i <= maxIndex; i++){
if(m->tab[i] == m->tab[minIndex]){
printf(" %d", source.tab[i]);
} else {
i = maxIndex+1;
}
}
printf("\n");
free(source.tab);
free(m->tab);
free(m);
}
