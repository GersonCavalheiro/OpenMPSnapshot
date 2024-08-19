#ifndef mpi_h_
#define mpi_h_

#include "./header.h"
#include "./utilities.h"
#include "./grid.h"

#ifdef MPI_OPENMP
#define MPI_OPENMP 1
#endif

static inline void calculate_neighbours(struct neighbour_processes *neighbour_processes, int rank_of_the_process, int process_grid_dimension) {
int periods[2]= {1,1}, dim[2] = { process_grid_dimension, process_grid_dimension}, coordinates[2], reorder = 1, number_of_dimensions = 2;
MPI_Comm comm;
MPI_Cart_create(MPI_COMM_WORLD, number_of_dimensions, dim, periods, reorder, &comm);
MPI_Cart_coords(comm, rank_of_the_process, 2, coordinates);
neighbour_processes->bottom_neighbour_coordinates[0] = coordinates[0] + 1;
neighbour_processes->bottom_neighbour_coordinates[1] = coordinates[1];
neighbour_processes->bottom_right_neighbour_coordinates[0] = coordinates[0] + 1;
neighbour_processes->bottom_right_neighbour_coordinates[1] = coordinates[1] + 1;
neighbour_processes->bottom_left_neighbour_coordinates[0] = coordinates[0] + 1;
neighbour_processes->bottom_left_neighbour_coordinates[1] = coordinates[1] - 1;
neighbour_processes->top_neighbour_coordinates[0] = coordinates[0] - 1;
neighbour_processes->top_neighbour_coordinates[1] = coordinates[1];
neighbour_processes->top_right_neighbour_coordinates[0] = coordinates[0] - 1;
neighbour_processes->top_right_neighbour_coordinates[1] = coordinates[1] + 1;
neighbour_processes->top_left_neighbour_coordinates[0] = coordinates[0] - 1;
neighbour_processes->top_left_neighbour_coordinates[1] = coordinates[1] - 1;
neighbour_processes->right_neighbour_coordinates[0] = coordinates[0];
neighbour_processes->right_neighbour_coordinates[1] = coordinates[1] + 1;
neighbour_processes->left_neighbour_coordinates[0] = coordinates[0];
neighbour_processes->left_neighbour_coordinates[1] = coordinates[1] - 1;
MPI_Cart_rank(comm, neighbour_processes->bottom_neighbour_coordinates, &neighbour_processes->bottom_neighbour_rank);
MPI_Cart_rank(comm, neighbour_processes->bottom_right_neighbour_coordinates, &neighbour_processes->bottom_right_neighbour_rank);
MPI_Cart_rank(comm, neighbour_processes->bottom_left_neighbour_coordinates, &neighbour_processes->bottom_left_neighbour_rank);
MPI_Cart_rank(comm, neighbour_processes->top_neighbour_coordinates, &neighbour_processes->top_neighbour_rank);
MPI_Cart_rank(comm, neighbour_processes->top_right_neighbour_coordinates, &neighbour_processes->top_right_neighbour_rank);
MPI_Cart_rank(comm, neighbour_processes->top_left_neighbour_coordinates, &neighbour_processes->top_left_neighbour_rank);
MPI_Cart_rank(comm, neighbour_processes->right_neighbour_coordinates, &neighbour_processes->right_neighbour_rank);
MPI_Cart_rank(comm, neighbour_processes->left_neighbour_coordinates, &neighbour_processes->left_neighbour_rank);
}

static inline void print_neighbour_ranks(struct neighbour_processes neighbour_processes, int rank_of_the_process) {
printf("Process rank %d\nneighbours Ranks: bottom = %d, bottom_right = %d, bottom_left = %d, top = %d, top_right = %d, top_left = %d, right = %d, left = %d \n"
,rank_of_the_process
,neighbour_processes.bottom_neighbour_rank,neighbour_processes.bottom_right_neighbour_rank
,neighbour_processes.bottom_left_neighbour_rank,neighbour_processes.top_neighbour_rank
,neighbour_processes.top_right_neighbour_rank,neighbour_processes.top_left_neighbour_rank,
neighbour_processes.right_neighbour_rank,neighbour_processes.left_neighbour_rank);
}

static inline char apply_rules(char state, int neighbours) {
if( (state == 0) && (neighbours == 3))
return 1;
else if( (state == 1) && ((neighbours < 2) || (neighbours > 3)) )
return 0;
else
return state;
}

static inline void receive(struct grid_side_dimensions *grid_side_dimensions,struct grid *current_generation,struct neighbour_processes neighbour_processes,MPI_Request *request) {
MPI_Irecv(grid_side_dimensions->bottom_dimension, current_generation->dimension, MPI_CHAR, neighbour_processes.top_neighbour_rank, 0,MPI_COMM_WORLD, &request[8]);
MPI_Irecv(grid_side_dimensions->top_dimension, current_generation->dimension, MPI_CHAR, neighbour_processes.bottom_neighbour_rank, 0,MPI_COMM_WORLD, &request[9]);
MPI_Irecv(grid_side_dimensions->right_dimension, current_generation->dimension, MPI_CHAR, neighbour_processes.left_neighbour_rank, 0,MPI_COMM_WORLD, &request[10]);
MPI_Irecv(grid_side_dimensions->left_dimension, current_generation->dimension, MPI_CHAR, neighbour_processes.right_neighbour_rank, 0,MPI_COMM_WORLD, &request[11]);
MPI_Irecv(&grid_side_dimensions->bottom_right_corner, 1, MPI_CHAR, neighbour_processes.top_left_neighbour_rank, 0, MPI_COMM_WORLD, &request[12]);
MPI_Irecv(&grid_side_dimensions->bottom_left_corner, 1, MPI_CHAR, neighbour_processes.top_right_neighbour_rank, 0, MPI_COMM_WORLD, &request[13]);
MPI_Irecv(&grid_side_dimensions->top_right_corner, 1, MPI_CHAR, neighbour_processes.bottom_left_neighbour_rank, 0, MPI_COMM_WORLD, &request[14]);
MPI_Irecv(&grid_side_dimensions->top_left_corner, 1, MPI_CHAR, neighbour_processes.bottom_right_neighbour_rank, 0, MPI_COMM_WORLD, &request[15]);
}

static inline void send(struct grid *current_generation,struct neighbour_processes neighbour_processes,MPI_Datatype columns,MPI_Request *request) {
MPI_Isend(&current_generation->array[0][0], current_generation->dimension, MPI_CHAR, neighbour_processes.top_neighbour_rank, 0, MPI_COMM_WORLD, &request[0]);
MPI_Isend(&current_generation->array[current_generation->dimension-1][0], current_generation->dimension, MPI_CHAR, neighbour_processes.bottom_neighbour_rank, 0, MPI_COMM_WORLD, &request[1]);
MPI_Isend(&current_generation->array[0][0], 1, columns, neighbour_processes.left_neighbour_rank, 0,MPI_COMM_WORLD, &request[2]);
MPI_Isend(&current_generation->array[0][current_generation->dimension-1], 1, columns, neighbour_processes.right_neighbour_rank, 0, MPI_COMM_WORLD, &request[3]);
MPI_Isend(&current_generation->array[0][0], 1, MPI_CHAR, neighbour_processes.top_left_neighbour_rank, 0,MPI_COMM_WORLD, &request[4]);
MPI_Isend(&current_generation->array[0][current_generation->dimension-1], 1, MPI_CHAR, neighbour_processes.top_right_neighbour_rank, 0, MPI_COMM_WORLD, &request[5]);
MPI_Isend(&current_generation->array[current_generation->dimension-1][0], 1, MPI_CHAR, neighbour_processes.bottom_left_neighbour_rank, 0,MPI_COMM_WORLD, &request[6]);
MPI_Isend(&current_generation->array[current_generation->dimension-1][current_generation->dimension-1],1, MPI_CHAR, neighbour_processes.bottom_right_neighbour_rank, 0, MPI_COMM_WORLD, &request[7]);
}

static inline void calculate_intermidiate_elements(struct grid *current_generation,struct grid *next_generation, int *different_generations) {
int i, j, neighbours=0;
#if MPI_OPENMP
#pragma omp parallel for shared (current_generation, next_generation, different_generations) private(i, j) reduction(+:neighbours) collapse(2)
#endif
for( i = 1 ; i < current_generation->dimension-1 ; i++ ) {
for( j = 1 ; j < current_generation->dimension-1 ; j++ ) {
neighbours = (current_generation->array[i-1][j-1])+(current_generation->array[i-1][j])+(current_generation->array[i-1][j+1])
+(current_generation->array[i][j-1])+(current_generation->array[i][j+1])
+(current_generation->array[i+1][j-1])+(current_generation->array[i+1][j])+(current_generation->array[i+1][j+1]);
next_generation->array[i][j] = apply_rules(current_generation->array[i][j],neighbours);

if( (*different_generations == 0) && (next_generation->array[i][j] != current_generation->array[i][j]) )
*different_generations = 1;
}
}
}

static inline void calculate_outline_elements(struct grid *current_generation,struct grid *next_generation, struct grid_side_dimensions *grid_side_dimensions,int *different_generations, int last) {
int i, neighbours=0;
#if MPI_OPENMP
#pragma omp parallel for shared (current_generation, next_generation, grid_side_dimensions, different_generations) private(i) reduction(+:neighbours)
#endif
for( i = 1 ; i < current_generation->dimension-1 ; i++ ) {
neighbours = (current_generation->array[0][i-1])+(current_generation->array[0][i+1])
+(current_generation->array[1][i-1])+(current_generation->array[1][i])+(current_generation->array[1][i+1])
+(grid_side_dimensions->top_dimension[i-1])+(grid_side_dimensions->top_dimension[i])+(grid_side_dimensions->top_dimension[i+1]);
next_generation->array[0][i] = apply_rules(current_generation->array[0][i],neighbours);

neighbours = (current_generation->array[last][i-1])+(current_generation->array[last][i+1])
+(current_generation->array[last-1][i-1])+(current_generation->array[last-1][i])+(current_generation->array[last-1][i+1])
+(grid_side_dimensions->bottom_dimension[i-1])+(grid_side_dimensions->bottom_dimension[i])+(grid_side_dimensions->bottom_dimension[i+1]);
next_generation->array[last][i] = apply_rules(current_generation->array[last][i],neighbours);

neighbours = (current_generation->array[i-1][0])+(current_generation->array[i+1][0])
+(current_generation->array[i-1][1])+(current_generation->array[i][1])+(current_generation->array[i+1][1])
+(grid_side_dimensions->left_dimension[i-1])+(grid_side_dimensions->left_dimension[i])+(grid_side_dimensions->left_dimension[i+1]);
next_generation->array[i][0] = apply_rules(current_generation->array[i][0],neighbours);

neighbours = (current_generation->array[i-1][last])+(current_generation->array[i+1][last])
+(current_generation->array[i-1][last-1])+(current_generation->array[i][last-1])+(current_generation->array[i+1][last-1])
+(grid_side_dimensions->right_dimension[i-1])+(grid_side_dimensions->right_dimension[i])+(grid_side_dimensions->right_dimension[i+1]);
next_generation->array[i][last] = apply_rules(current_generation->array[i][last],neighbours);

if( *different_generations == 0 ) {
if( (next_generation->array[0][i] != current_generation->array[0][i]) || (next_generation->array[last][i] != current_generation->array[last][i]) ||
(next_generation->array[i][0] != current_generation->array[i][0]) || (next_generation->array[i][last] != current_generation->array[i][last]) ) {
*different_generations = 1;
}
}
}
neighbours = (current_generation->array[0][1])+(current_generation->array[1][0])+(current_generation->array[1][1])
+(grid_side_dimensions->top_dimension[0])+(grid_side_dimensions->top_dimension[1])
+(grid_side_dimensions->left_dimension[0])+(grid_side_dimensions->left_dimension[1])
+(grid_side_dimensions->top_left_corner);
next_generation->array[0][0] = apply_rules(current_generation->array[0][0],neighbours);

neighbours = (current_generation->array[0][last-1])+(current_generation->array[1][last])+(current_generation->array[1][last-1])
+(grid_side_dimensions->top_dimension[last])+(grid_side_dimensions->top_dimension[last-1])
+(grid_side_dimensions->right_dimension[0])+(grid_side_dimensions->right_dimension[1])
+(grid_side_dimensions->top_right_corner);
next_generation->array[0][last] = apply_rules(current_generation->array[0][last],neighbours);

neighbours = (current_generation->array[last-1][0])+(current_generation->array[last-1][1])+(current_generation->array[last][1])
+(grid_side_dimensions->bottom_dimension[0])+(grid_side_dimensions->bottom_dimension[1])
+(grid_side_dimensions->left_dimension[last])+(grid_side_dimensions->left_dimension[last-1])
+(grid_side_dimensions->bottom_left_corner);

next_generation->array[last][0] = apply_rules(current_generation->array[last][0],neighbours);
neighbours = (current_generation->array[last-1][last])+(current_generation->array[last-1][last-1])+(current_generation->array[last][last-1])
+(grid_side_dimensions->bottom_dimension[last])+(grid_side_dimensions->bottom_dimension[last-1])
+(grid_side_dimensions->right_dimension[last])+(grid_side_dimensions->right_dimension[last-1])
+(grid_side_dimensions->bottom_right_corner);
next_generation->array[last][last] = apply_rules(current_generation->array[last][last],neighbours);

if( *different_generations == 0 ) {
if( (next_generation->array[0][0] != current_generation->array[0][0]) || (next_generation->array[0][last] != current_generation->array[0][last]) ||
(next_generation->array[last][0] != current_generation->array[last][0]) || (next_generation->array[last][last] != current_generation->array[last][last]) ) {
*different_generations = 1;
}
}
}

#endif