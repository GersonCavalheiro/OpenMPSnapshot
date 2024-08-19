#include "hashtable.h"
#include "cell.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
const int OVERLAP = 2;
static unsigned int next_generation(const hashtable_t *now, hashtable_t *next, unsigned int size);
static unsigned int get_cells_in_region(const hashtable_t *ht, cell_t lower_bound, cell_t upper_bound,
unsigned int size, int overlap, cell_t *out);
void
get_num_cells_to_send(hashtable_t *ht, cell_t lower_bound, cell_t upper_bound, int overlap, unsigned int *num_cells);
void get_cells_to_send(hashtable_t *ht, cell_t lower_bound, cell_t upper_bound, int overlap, cell_t **cells);
unsigned int sendrecv_boundary_cells(MPI_Comm comm, cell_t **sendbuf, const unsigned int *sendcount, cell_t **received);
void life3d_run(unsigned int size, hashtable_t *state, unsigned int num_cells, unsigned long generations) {
int grid_rank, num_procs;
int dims[3] = {0, 0, 0};
int coords[3] = {0, 0, 0};
const int periodic[] = {1, 1, 1};
MPI_Comm grid_comm;
MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
MPI_Dims_create(num_procs, 3, dims);
MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, 1, &grid_comm);
MPI_Comm_rank(grid_comm, &grid_rank);
MPI_Cart_coords(grid_comm, grid_rank, 3, coords);
cell_t lower_bound = cell_block_low(coords, dims, size);
cell_t upper_bound = cell_block_high(coords, dims, size);
cell_t *my_cells = (cell_t *) calloc(num_cells, sizeof(cell_t));
num_cells = get_cells_in_region(state, lower_bound, upper_bound, size, OVERLAP, my_cells);
HT_free(state);
state = HT_create(num_cells * 3);
HT_set_all(state, my_cells, num_cells);
free(my_cells);
hashtable_t *next_state;
for (unsigned long gen = 0; gen < generations; gen++) {
next_state = HT_create(num_cells * 6);
num_cells = next_generation(state, next_state, size);
HT_free(state);
state = next_state;
if (!((gen + 1) % OVERLAP)) {
unsigned int num_cells_to_send[26] = {0};
get_num_cells_to_send(state, lower_bound, upper_bound, OVERLAP, num_cells_to_send);
cell_t *cells_to_send[26];
for (int i = 0; i < 26; i++) {
cells_to_send[i] = (cell_t *) calloc((size_t) num_cells_to_send[i], sizeof(cell_t));
}
get_cells_to_send(state, lower_bound, upper_bound, OVERLAP, cells_to_send);
cell_t *cells_received;
unsigned int num_cells_recv = sendrecv_boundary_cells(grid_comm, cells_to_send, num_cells_to_send,
&cells_received);
for (int i = 0; i < 26; i++) {
free(cells_to_send[i]);
}
my_cells = (cell_t *) calloc(num_cells, sizeof(cell_t));
num_cells = get_cells_in_region(state, lower_bound, upper_bound, size, 0, my_cells);
HT_free(state);
state = HT_create((num_cells + num_cells_recv) * 3);
HT_set_all(state, my_cells, num_cells);
free(my_cells);
HT_set_all(state, cells_received, num_cells_recv);
}
}
my_cells = (cell_t *) calloc(num_cells, sizeof(cell_t));
num_cells = get_cells_in_region(state, lower_bound, upper_bound, size, 0, my_cells);
int num_cells_by_process[num_procs];
MPI_Gather(&num_cells, 1, MPI_UNSIGNED, num_cells_by_process, 1, MPI_INT, 0, grid_comm);
int displs[num_procs];
displs[0] = 0;
for (int p = 1; p < num_procs; p++) {
displs[p] = displs[p - 1] + num_cells_by_process[p - 1];
}
unsigned int total_cells = (unsigned int) displs[num_procs - 1] + num_cells_by_process[num_procs - 1];
cell_t *all_cells = NULL;
if (grid_rank == 0) {
all_cells = (cell_t *) calloc(total_cells, sizeof(cell_t));
}
qsort(my_cells, num_cells, sizeof(cell_t), compare_cells);
MPI_Gatherv(my_cells, num_cells, MPI_UNSIGNED_LONG_LONG, all_cells, num_cells_by_process, displs,
MPI_UNSIGNED_LONG_LONG, 0, grid_comm);
if (grid_rank != 0)
return;
qsort(all_cells, total_cells, sizeof(cell_t), compare_cells);
for (unsigned int i = 0; i < total_cells; i++) {
fprintf(stdout, "%u %u %u\n", CELL_X(all_cells[i]), CELL_Y(all_cells[i]), CELL_Z(all_cells[i]));
}
}
int in_boundary_region(cell_t c, cell_t lower_bound, cell_t upper_bound, int boundary_size, const int *direction) {
for (int dim = 0; dim < 3; dim++) {
if (CELL_COORD(lower_bound, dim) > CELL_COORD(c, dim) || CELL_COORD(c, dim) > CELL_COORD(upper_bound, dim)) {
return 0;
}
if (direction[dim] == -1 && CELL_COORD(c, dim) > CELL_COORD(lower_bound, dim) + boundary_size) {
return 0;
}
if (direction[dim] == 1 && CELL_COORD(c, dim) < CELL_COORD(upper_bound, dim) - boundary_size) {
return 0;
}
}
return 1;
}
void
get_num_cells_to_send(hashtable_t *ht, cell_t lower_bound, cell_t upper_bound, int overlap, unsigned int *num_cells) {
int direction[3];
for (int i = 0; i < ht->capacity; i++) {
cell_t c = ht->table[i];
if (c == 0) {
continue;
}
int shift = 0;
for (direction[0] = -1; direction[0] < 2; direction[0]++) {
for (direction[1] = -1; direction[1] < 2; direction[1]++) {
for (direction[2] = -1; direction[2] < 2; direction[2]++) {
if (!(direction[0] || direction[1] || direction[2])) {
continue;
}
if (in_boundary_region(c, lower_bound, upper_bound, overlap, direction)) {
num_cells[shift]++;
}
shift++;
}
}
}
}
}
void get_cells_to_send(hashtable_t *ht, cell_t lower_bound, cell_t upper_bound, int overlap, cell_t **cells) {
int direction[3];
int num_cells[26] = {0};
for (int i = 0; i < ht->capacity; i++) {
cell_t c = ht->table[i];
if (c == 0) {
continue;
}
int shift = 0;
for (direction[0] = -1; direction[0] < 2; direction[0]++) {
for (direction[1] = -1; direction[1] < 2; direction[1]++) {
for (direction[2] = -1; direction[2] < 2; direction[2]++) {
if (!(direction[0] || direction[1] || direction[2])) {
continue;
}
if (in_boundary_region(c, lower_bound, upper_bound, overlap, direction)) {
cells[shift][num_cells[shift]++] = c;
}
shift++;
}
}
}
}
}
unsigned int
sendrecv_boundary_cells(MPI_Comm comm, cell_t **sendbuf, const unsigned int *sendcount, cell_t **received) {
int rank, my_coords[3];
MPI_Comm_rank(comm, &rank);
MPI_Cart_coords(comm, rank, 3, my_coords);
int shift = 0;
int direction[3];
cell_t *recvbufs[26];
unsigned int recvcount[26];
for (direction[0] = -1; direction[0] < 2; direction[0]++) {
for (direction[1] = -1; direction[1] < 2; direction[1]++) {
for (direction[2] = -1; direction[2] < 2; direction[2]++) {
if (!(direction[0] || direction[1] || direction[2])) {
continue;
}
int src_coords[3] = {my_coords[0] - direction[0],
my_coords[1] - direction[1],
my_coords[2] - direction[2]};
int dest_coords[3] = {my_coords[0] + direction[0],
my_coords[1] + direction[1],
my_coords[2] + direction[2]};
int source, dest;
MPI_Cart_rank(comm, src_coords, &source);
MPI_Cart_rank(comm, dest_coords, &dest);
if (source == rank && dest == rank) {
recvbufs[shift] = malloc(0);
recvcount[shift] = 0;
shift++;
continue;
}
MPI_Sendrecv(&sendcount[shift], 1, MPI_UNSIGNED, dest, 18361,
&recvcount[shift], 1, MPI_UNSIGNED, source, 18361,
comm, NULL);
recvbufs[shift] = calloc((size_t) recvcount[shift], sizeof(cell_t));
MPI_Sendrecv(sendbuf[shift], sendcount[shift], MPI_UNSIGNED_LONG_LONG, dest, 25341,
recvbufs[shift], recvcount[shift], MPI_UNSIGNED_LONG_LONG, source, 25341,
comm, NULL);
shift++;
}
}
}
unsigned int total_cells_received = 0;
for (int i = 0; i < 26; i++) {
total_cells_received += recvcount[i];
}
*received = (cell_t *) calloc(total_cells_received, sizeof(cell_t));
int copied_so_far = 0;
for (int i = 0; i < 26; i++) {
for (int j = 0; j < recvcount[i]; j++) {
(*received)[copied_so_far++] = recvbufs[i][j];
}
free(recvbufs[i]);
}
return total_cells_received;
}
static unsigned int get_cells_in_region(const hashtable_t *ht, cell_t lower_bound, cell_t upper_bound,
unsigned int size, int overlap, cell_t *out) {
unsigned int num_cells = 0;
cell_t c;
for (unsigned int i = 0; i < ht->capacity; i++) {
c = ht->table[i];
if (c == 0)
continue;
if (in_region(c, lower_bound, upper_bound, size, overlap))
out[num_cells++] = c;
}
return num_cells;
}
static unsigned int next_generation(const hashtable_t *now, hashtable_t *next, unsigned int size) {
unsigned int ncells_next = 0;
#pragma omp parallel for reduction(+:ncells_next) shared(now, next)
for (unsigned int i = 0; i < now->capacity; i++) {
cell_t c = now->table[i];
if (c == 0) {
continue;
}
cell_t neighbors[6];
cell_get_neighbors(c, neighbors, size);
if (cell_next_state(c, neighbors, now)) {
HT_set_atomic(next, c);
ncells_next++;
}
for (size_t j = 0; j < 6; j++) {
c = neighbors[j];
if (!(HT_contains(now, c)) && !(HT_contains(next, c))) {
cell_t buf[6];
cell_get_neighbors(c, buf, size);
if (cell_next_state(c, buf, now)) {
HT_set_atomic(next, c);
ncells_next++;
}
}
}
}
return ncells_next;
}
