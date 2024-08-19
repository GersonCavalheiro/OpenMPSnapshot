#ifndef GoL_LIFE_H
#define GoL_LIFE_H
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../globals.h"
typedef struct life {
int ncols;         
int nrows;         
int timesteps;     
double init_prob;  
#ifdef _OPENMP
int nthreads;      
#endif
#ifdef GoL_CUDA
int block_size;    
#endif
unsigned int seed; 
#ifdef GoL_CUDA
bool *grid;        
#else
bool **grid;       
bool **next_grid;  
#endif
char *infile;      
char *outfile;     
} life_t;
void initialize(life_t *life);
double game(life_t *life);
#ifdef GoL_CUDA
__global__ void evolve(bool *gpu_grid,
bool *gpu_next_grid, int nrows, int ncols);
#else
void evolve(life_t *life);
#endif
void cleanup(life_t *life);
#ifdef GoL_DEBUG
void show_grid_status(life_t life) {
int i, j;
int ncols = life.ncols;
int nrows = life.nrows;
int n_alive = 0;
int n_dead  = 0;
#ifdef _OPENMP
#pragma omp parallel for private(j) reduction(+:n_alive, n_dead)
#endif
for (i = 0; i < nrows; i++) 
for (j = 0; j < ncols; j++) {
#ifdef GoL_CUDA
life.grid[i*ncols + j] == ALIVE \
? n_alive++ : n_dead++;
#else
life.grid[i][j] == ALIVE \
? n_alive++ : n_dead++;
#endif
}
printf("Number of ALIVE cells: %d\n",  n_alive);
printf("Number of DEAD cells: %d\n\n", n_dead);
fflush(stdout);
usleep(320000);
}
void debug(life_t life) {
printf("Number of cols: %d\n", life.ncols);
printf("Number of rows: %d\n", life.nrows);
printf("Number of timesteps: %d\n", life.timesteps);
printf("Probability for grid initialization: %f\n", life.init_prob);
printf("Random seed initializer: %d\n", life.seed);
#ifdef _OPENMP
printf("Number of total OpenMP threads: %d\n", life.nthreads);
#endif
#ifdef GoL_CUDA
printf("Number of threads per CUDA block: %d\n", life.block_size);
#endif
printf("Input file: %s\n", life.infile == NULL ? "None" : life.infile);
printf("Output file: %s\n\n", life.outfile);
fflush(stdout);
}
#endif
bool is_big(life_t life) {
return life.nrows * life.ncols > DEFAULT_MAX_SIZE;
}
void show(life_t life) {
int i, j;
int ncols = life.ncols;
int nrows = life.nrows;
printf("\033[H\033[J");
for (i = 0; i < nrows; i++) {
for (j = 0; j < ncols; j++) {
#ifdef GoL_CUDA
printf(life.grid[i*ncols + j] == ALIVE
? "\033[07m  \033[m" : "  ");
#else
printf(life.grid[i][j] == ALIVE
? "\033[07m  \033[m" : "  ");
#endif
}
printf("\033[E"); 
}
fflush(stdout);
usleep(160000);
}
void printbig(life_t life, bool append) {
int i, j;
int ncols = life.ncols;
int nrows = life.nrows;
FILE *out_ptr = append \
? fopen(life.outfile, "a" ) \
: fopen(life.outfile, "w" );
if (out_ptr == NULL) {
perror("[*] Failed to open the output file.");
exit(EXIT_FAILURE);
}
if (!append) 
fprintf(out_ptr, "%d %d\n", nrows, ncols);
for (i = 0; i < nrows; i++) {
for (j = 0; j < ncols; j++) {
#ifdef GoL_CUDA
fprintf(out_ptr, "%c", life.grid[i*ncols + j] == ALIVE
? 'X' : ' ');
#else
fprintf(out_ptr, "%c", life.grid[i][j] == ALIVE
? 'X' : ' ');
#endif
}  
fprintf(out_ptr, "\n");
}
fprintf(out_ptr, "****************************************************************************************************\n");
fflush(out_ptr);
fclose(out_ptr);
}
void display(life_t life, bool append) {
if(is_big(life)) printbig(life, append);
else show(life);
}
#endif
