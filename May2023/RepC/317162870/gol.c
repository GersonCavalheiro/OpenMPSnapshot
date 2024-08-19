#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "gol.h"
void initialize(life_t *life) {
srand(life->seed);
FILE *input_ptr = set_grid_dimens_from_file(life);
malloc_grid(life);
init_empty_grid(life);
if (input_ptr != NULL) { 
init_from_file(life, input_ptr);
} else {  
init_random(life);
}
#ifdef GoL_DEBUG
debug(*life);
usleep(1000000);
#endif
}
double game(life_t *life) {
int x, y, t;
struct timeval gstart, gend;
initialize(life);
int ncols = life->ncols;
int nrows = life->nrows;
double tot_gene_time = 0.;
double cur_gene_time = 0.;
display(*life, false);
for(t = 0; t < life->timesteps; t++) { 
gettimeofday(&gstart, NULL);
evolve(life);
gettimeofday(&gend, NULL);
cur_gene_time = elapsed_wtime(gstart, gend);
tot_gene_time += cur_gene_time;
if (is_big(*life)) {
printf("Generation #%d took %.5f ms\n", t, cur_gene_time);  
if (t == life->timesteps - 1) {
display(*life, true);
}
} else {
display(*life, true);
}
#ifdef GoL_DEBUG
get_grid_status(*life);
#endif
}
printf("\nEvolved GoL's grid for %d generations - ETA: %.5f ms\n",
life->timesteps, tot_gene_time);
return tot_gene_time;
}
void evolve(life_t *life) {
int x, y, i, j, r, c;
int alive_neighbs; 
int ncols = life->ncols;
int nrows = life->nrows;
#ifdef _OPENMP
#pragma omp parallel for private(alive_neighbs, y, i, j, r, c)
#endif
for (x = 0; x < nrows; x++) 
for (y = 0; y < ncols; y++) {
alive_neighbs = 0;
for (i = x - 1; i <= x + 1; i++)
for (j = y - 1; j <= y + 1; j++) {
c = (i + nrows) % nrows;
r = (j + ncols) % ncols;
if (!(i == x && j == y) 
&& life->grid[c][r] == ALIVE)
alive_neighbs++;
}
life->next_grid[x][y] = (alive_neighbs == 3
|| (alive_neighbs == 2
&& life->grid[x][y] == ALIVE)) \
? ALIVE : DEAD;
}
swap_grids(&life->grid, &life->next_grid);
}
void cleanup(life_t *life) {
int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (i = 0; i < life->nrows; i++) {
free(life->grid[i]);
free(life->next_grid[i]);
}
free(life->grid);
free(life->next_grid);
}
int main(int argc, char **argv) {
struct timeval start, end;
double cum_gene_time, elapsed_prog_wtime;
int nprocs = 1; 
life_t life;    
gettimeofday(&start, NULL);
parse_args(&life, argc, argv);
#ifdef _OPENMP
omp_set_num_threads(life.nthreads);
#endif
FILE *input_ptr = set_grid_dimens_from_file(&life);
#ifdef GoL_MPI 
int rows_per_process;
int from; 
int to;   
int status = MPI_Init(&argc, &argv);
if (status != MPI_SUCCESS) {
fprintf(stderr, "[*] Failed to initialize MPI environment - errcode %d", status);
MPI_Abort(MPI_COMM_WORLD, 1);
}
chunk_t chunk; 
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &chunk.rank);
chunk.size = nprocs;
if (chunk.size != 1) { 
MPI_Barrier(MPI_COMM_WORLD);
rows_per_process = (int) life.nrows/chunk.size;
chunk.displacement = life.nrows % chunk.size;
from = chunk.rank * rows_per_process;
if (chunk.rank == chunk.size - 1) { 
to = life.nrows - 1;
chunk.nrows = life.nrows - from;
} else {
to = (chunk.rank + 1) * rows_per_process - 1;
chunk.nrows = rows_per_process;
}
chunk.ncols = life.ncols; 
initialize_chunk(&chunk, life,
input_ptr, from, to);
double tot_gtime = game_chunk(&chunk, life);
if (chunk.rank == 0) {
cum_gene_time = tot_gtime;
}
MPI_Barrier(MPI_COMM_WORLD);
cleanup_chunk(&chunk);
if(chunk.rank == 0) {
gettimeofday(&end, NULL);
elapsed_prog_wtime = elapsed_wtime(start, end);
}
} else { 
cum_gene_time = game(&life);
cleanup(&life);
gettimeofday(&end, NULL);
elapsed_prog_wtime = elapsed_wtime(start, end);
}
status = MPI_Finalize();
if (status != MPI_SUCCESS) {
fprintf(stderr, "[*] Failed to finalize MPI environment - errcode %d", status);
MPI_Abort(MPI_COMM_WORLD, 1);
}
#else 
cum_gene_time = game(&life);
cleanup(&life);
gettimeofday(&end, NULL);
elapsed_prog_wtime = elapsed_wtime(start, end);
#endif
#ifdef GoL_LOG
#ifdef GoL_MPI
if (chunk.rank == 0) { 
#endif
FILE *log_ptr = init_log_file(life, nprocs);
log_data(log_ptr, life.timesteps, cum_gene_time,
elapsed_prog_wtime);
fflush(log_ptr);
fclose(log_ptr);
#ifdef GoL_MPI
}
#endif
#endif
#ifdef GoL_MPI
if (chunk.rank == 0) {
#endif
printf("\nFinalized the program - ETA: %.5f ms\n\n", elapsed_prog_wtime);
#ifdef GoL_MPI
}
#endif
}