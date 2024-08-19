#ifndef GoL_LIFE_INIT_H
#define GoL_LIFE_INIT_H
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "life.h"
#include "../utils/func.h"
FILE* set_grid_dimens_from_file(life_t *life) {
FILE *file_ptr;
if (life->infile != NULL) {
if ((file_ptr = fopen(life->infile, "r")) == NULL) {
perror("[*] Failed to open the input file.\n");
perror("[*] Launching the evolution in default configuration...\n");
} else if (fscanf(file_ptr, "%d %d\n", &life->nrows, &life->ncols) == EOF) {
perror("[*] The input file only defines GoL board's dimensions!\n");
perror("[*] Launching the evolution in default configuration...\n");
} else{
return file_ptr;
}
}
return NULL;
}
void malloc_grid(life_t *life) {
int ncols = life->ncols;
int nrows = life->nrows;
#ifdef GoL_CUDA
life->grid = (bool *) malloc(nrows*ncols * sizeof(bool));
if (life->grid == NULL) {
perror("[*] GoL's board allocation failed!\n");
exit(EXIT_FAILURE);
}
#else
int i;
life->grid      = (bool **) malloc(sizeof(bool *) * nrows);
life->next_grid = (bool **) malloc(sizeof(bool *) * nrows);
if (life->grid == NULL
|| life->next_grid == NULL) {
perror("[*] GoL's board allocation failed!\n");
exit(EXIT_FAILURE);
}
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (i = 0; i < nrows; i++) {
life->grid[i]      = (bool *) malloc(sizeof(bool) * (ncols));
life->next_grid[i] = (bool *) malloc(sizeof(bool) * (ncols));
if (life->grid[i] == NULL
|| life->next_grid[i] == NULL) {
perror("[*] GoL's board allocation failed!\n");
exit(EXIT_FAILURE);
}
}
#endif
}
void init_empty_grid(life_t *life) {
int i;
#ifdef GoL_CUDA
for (i = 0; i < life->nrows*life->ncols; i++)
life->grid[i] = DEAD;
#else
int j;
#ifdef _OPENMP
#pragma omp parallel for private(j)
#endif
for (i = 0; i < life->nrows; i++)
for (j = 0; j < life->ncols; j++) {
life->grid[i][j]      = DEAD;
life->next_grid[i][j] = DEAD;
}
#endif
}
void init_from_file(life_t *life, FILE *file_ptr) {
int i, j;
if (life->infile != NULL) {
char *line = NULL;
size_t buf_size = 0; 
ssize_t len = 0;     
i = 0;
while ((len = getline(&line, &buf_size, file_ptr)) != -1) {
if (i >= life->nrows){
perror("[*] GoL's input file exceeds the number of rows!\n");
exit(EXIT_FAILURE);
}
if (len != life->ncols + 1) { 
fprintf(stderr, "[*] Row #%d does not respect the number of columns!\n", i);
exit(EXIT_FAILURE);
}
for (j = 0; j < len - 1; j++) {
if(line[j] == 'X') {
#ifdef GoL_CUDA
life->grid[i*life->ncols + j] = ALIVE;
#else
life->grid[i][j] = ALIVE;
#endif
}
}
i++;
}
free(line);
line = NULL;
if (i != life->nrows) {
perror("[*] GoL's input file does not respect the number of rows!\n");
exit(EXIT_FAILURE);
}
}
fflush(file_ptr);
fclose(file_ptr);
}
void init_random(life_t *life) {
int i;
#ifdef GoL_CUDA
for (i = 0; i < life->nrows*life->ncols; i++)
if (rand_double(0., 1.) < life->init_prob)
life->grid[i] = ALIVE;
#else
int j;
for (i = 0; i < life->nrows; i++) 
for (j = 0; j < life->ncols; j++) { 
if (rand_double(0., 1.) < life->init_prob)
life->grid[i][j] = ALIVE;
}
#endif
}
#endif
