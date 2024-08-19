#ifndef GoL_H
#define GoL_H
#include <stdlib.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h> 
#endif
#ifdef GoL_MPI
#include <mpi.h> 
#endif
#include "../../include/globals.h"
#include "../../include/utils/log.h"
#include "../../include/utils/func.h"
#include "../../include/utils/parse.h"
#include "../../include/life/init.h"
void swap_grids(bool ***old, bool ***new) {
bool **temp = *old;
*old = *new;
*new = temp;
}
void get_grid_status(life_t life) {
int i, j;
int ncols = life.ncols;
int nrows = life.nrows;
int n_alive = 0;
int n_dead  = 0;
#ifdef _OPENMP
#pragma omp parallel for private(j) reduction(+:n_alive, n_dead)
#endif
for (i = 0; i < nrows; i++) 
for (j = 0; j < ncols; j++)
life.grid[i][j] == ALIVE ? n_alive++ : n_dead++;
printf("Number of ALIVE cells: %d\n",  n_alive);
printf("Number of DEAD cells: %d\n\n", n_dead);
fflush(stdout);
usleep(320000);
}
#ifdef GoL_MPI
#include "../../include/chunk/init.h"
void initialize_chunk(chunk_t *chunk, life_t life,
FILE *input_ptr, int from, int to) {
srand(life.seed);
malloc_chunk(chunk);
init_empty_chunk(chunk);
if (input_ptr != NULL) { 
init_chunk_from_file(chunk, life.nrows, life.ncols,
input_ptr, from, to);
} else {  
init_random_chunk(chunk, life, from, to);
}
#ifdef GoL_DEBUG
debug_chunk(*chunk);
usleep(1000000);
#endif
}
double game_chunk(chunk_t *chunk, life_t life) {
int i;
MPI_Status status;
int timesteps = life.timesteps;
int tot_rows  = life.nrows;
char *outfile = life.outfile;
bool big = is_big(life);
struct timeval gstart, gend;
double cur_gene_time = 0.0;
double tot_gene_time = 0.0;
display_chunk(chunk, big, tot_rows,
outfile, false);
for (i = 0; i < timesteps; i++) {
MPI_Barrier(MPI_COMM_WORLD);
if (chunk->rank == 0)
gettimeofday(&gstart, NULL);
evolve_chunk(chunk);
int prev_rank = (chunk->rank - 1 + chunk->size) % chunk->size;
int next_rank = (chunk->rank + 1) % chunk->size;
MPI_Sendrecv(&chunk->slice[1][0], chunk->ncols, MPI_C_BOOL, prev_rank, TOP,
&chunk->slice[chunk->nrows + 1][0], chunk->ncols, MPI_C_BOOL, next_rank, TOP,
MPI_COMM_WORLD, &status);
MPI_Sendrecv(&chunk->slice[chunk->nrows][0], chunk->ncols, MPI_C_BOOL, next_rank, BOTTOM,
&chunk->slice[0][0], chunk->ncols, MPI_C_BOOL, prev_rank, BOTTOM,
MPI_COMM_WORLD, &status);
MPI_Barrier(MPI_COMM_WORLD);
if (chunk->rank == 0) {
gettimeofday(&gend, NULL);
cur_gene_time = elapsed_wtime(gstart, gend);
tot_gene_time += cur_gene_time;
}
if(big) {
if (chunk->rank == 0)
printf("Generation #%d took %.5f ms on process 0\n", i, cur_gene_time);  
if (i == timesteps - 1) {
display_chunk(chunk, big, tot_rows,
outfile, true);
}
} else {
display_chunk(chunk, big, tot_rows,
outfile, true);
}
}
if (chunk->rank == 0)
printf("\nEvolved GoL's grid for %d generations - ETA: %.5f ms\n",
timesteps, tot_gene_time);
return tot_gene_time;
}
void evolve_chunk(chunk_t *chunk) {
int x, y, i, j, r, c;
int alive_neighbs; 
int ncols = chunk->ncols;
int nrows = chunk->nrows;
#ifdef _OPENMP
#pragma omp parallel for private(alive_neighbs, y, i, j, r, c)
#endif
for (x = 1; x < nrows + 1; x++) 
for (y = 0; y < ncols; y++) {
alive_neighbs = 0;
for (i = x - 1; i <= x + 1; i++)
for (j = y - 1; j <= y + 1; j++) {
c = (j + ncols) % ncols;
if (!(i == x && j == y) 
&& chunk->slice[i][c] == ALIVE)
alive_neighbs++;
}
chunk->next_slice[x][y] = (alive_neighbs == 3
|| (alive_neighbs == 2
&& chunk->slice[x][y] == ALIVE)) \
? ALIVE : DEAD;
}
swap_grids(&chunk->slice, &chunk->next_slice);
}
void cleanup_chunk(chunk_t *chunk) {
int i;
free(chunk->slice);
free(chunk->next_slice);
}
#endif
#endif
