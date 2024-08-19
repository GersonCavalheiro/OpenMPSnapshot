#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "tris.h"
#define CHUNK_SIZE 6
#define THREAD_NUM 2
#define ARRAY_SIZE 10
#define DEBUG 0
void bubble_sort(int * tab, int tab_size){
for(int i = tab_size - 1; i>1; i--){
for(int j=0; j<=i-1; j++){
if(tab[j+1] < tab[j]){
int t = tab[j+1];
tab[j+1] = tab[j];
tab[j] = t;
}
}
}
}
void bulle_trie (int * tab, int tab_size){
for(int i = 0; i<tab_size-1; i++){
if(tab[i]>tab[i+1]){
bubble_sort(tab, tab_size);
}
}
}
void bubble_sort_omp (int * tab, int tab_size){
omp_set_dynamic(0);
omp_set_num_threads(2);
int base;
#pragma omp parallel for private(base)
for(int o=0; o<THREAD_NUM; o++){
base = (CHUNK_SIZE * o);
for(int i = CHUNK_SIZE - 1; i>1; i--){
for(int j=0; j<=i-1; j++){
if(j+base+1 >= tab_size)
break;
if(tab[j+base+1] < tab[j+base]){
int t = tab[j+base+1];
tab[j+base+1] = tab[j+base];
tab[j+base] = t;
}
}
}
}
}