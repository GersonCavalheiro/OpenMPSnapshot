#include <mpi.h>
#include <omp.h>
#include "array_list.h"
#include "structures.h"
#define NUM_PARTICLES_BUFFER 10000
#define CONVERT_TO_LOCAL(id, p, n, x) (x - BLOCK_LOW(id, p, n) + 1)
enum { TAG_INIT_PARTICLES, TAG_SEND_CENTER_OF_MASS, TAG_SEND_PARTICLES };
node_t* adjacent_processes[8] = {0};
array_list_t* particles;
cell_t** cells;
int myRank, number_processors;
int size_processor_grid[2] = {0, 0};
int my_coordinates[2];
node_t* processes_buffers;
int size_local_cell_matrix[2];
MPI_Comm cart_comm;
void create_cartesian_communicator(int grid_size) {
int periodic[2] = {1, 1};
if (number_processors <= 3) {
MPI_Dims_create(number_processors, 2, size_processor_grid);
} else {
size_processor_grid[1] = sqrt(number_processors);
if (size_processor_grid[1] >= grid_size) {
size_processor_grid[0] = size_processor_grid[1] = grid_size;
} else {
size_processor_grid[0] = number_processors / size_processor_grid[1];
}
}
MPI_Cart_create(MPI_COMM_WORLD, 2, size_processor_grid, periodic, 1, &cart_comm);
if (size_processor_grid[0] * size_processor_grid[1] <= myRank) {
MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
exit(0);
}
MPI_Comm_rank(cart_comm, &myRank);
MPI_Cart_coords(cart_comm, myRank, 2, my_coordinates);
int adjacent_processes_rank[8];
int counter = 0;
for (int i = -1; i <= 1; i++) {
for (int j = -1; j <= 1; j++) {
if (i == 0 && j == 0) {
continue;
}
int coords[2];
coords[0] = my_coordinates[0] + i;
coords[1] = my_coordinates[1] + j;
MPI_Cart_rank(cart_comm, coords, &adjacent_processes_rank[counter]);
counter++;
}
}
for (int i = 0; i < 8; i++) {
if (adjacent_processes[i] != NULL) {
continue;
}
adjacent_processes[i] = (node_t*)calloc(1, sizeof(node_t));
adjacent_processes[i]->rank = adjacent_processes_rank[i];
for (int j = i + 1; j < 8; j++) {
if (adjacent_processes_rank[j] == adjacent_processes_rank[i]) {
adjacent_processes[j] = adjacent_processes[i];
}
}
}
processes_buffers = (node_t*)calloc(size_processor_grid[0] * size_processor_grid[1], sizeof(node_t));
number_processors = size_processor_grid[0] * size_processor_grid[1];
}
void init_cells(int grid_size) {
size_local_cell_matrix[0] = BLOCK_SIZE(my_coordinates[0], size_processor_grid[0], grid_size) + 2;
size_local_cell_matrix[1] = BLOCK_SIZE(my_coordinates[1], size_processor_grid[1], grid_size) + 2;
cells = (cell_t**)malloc(sizeof(cell_t*) * size_local_cell_matrix[0]);
cell_t* cells_chunk = (cell_t*)calloc(size_local_cell_matrix[0] * size_local_cell_matrix[1], sizeof(cell_t));
for (int i = 0; i < size_local_cell_matrix[0]; i++) {
cells[i] = &cells_chunk[i * size_local_cell_matrix[1]];
}
}
void init_particles(long seed, long grid_size, long long number_particles) {
long long i;
int number_processors_grid = size_processor_grid[0] * size_processor_grid[1];
int* counters = (int*)calloc(number_processors_grid, sizeof(int));
particle_t* buffer_space =
(particle_t*)malloc(sizeof(particle_t) * NUM_PARTICLES_BUFFER * (number_processors_grid));
particle_t** buffers = (particle_t**)malloc(sizeof(particle_t*) * (number_processors_grid));
for (int i = 1; i < number_processors_grid; i++) {
buffers[i - 1] = &buffer_space[NUM_PARTICLES_BUFFER * (i - 1)];
}
srandom(seed);
for (i = 0; i < number_particles; i++) {
particle_t particle;
particle.index = i;
particle.position.x = RND0_1;
particle.position.y = RND0_1;
particle.velocity.x = RND0_1 / grid_size / 10.0;
particle.velocity.y = RND0_1 / grid_size / 10.0;
particle.mass = RND0_1 * grid_size / (G * 1e6 * number_particles);
int cell_coordinate_x = particle.position.x * grid_size;
int cell_coordinate_y = particle.position.y * grid_size;
int coords_proc_grid[2];
coords_proc_grid[0] = BLOCK_OWNER(cell_coordinate_x, size_processor_grid[0], grid_size);
coords_proc_grid[1] = BLOCK_OWNER(cell_coordinate_y, size_processor_grid[1], grid_size);
int proc_id_to_send;
MPI_Cart_rank(cart_comm, coords_proc_grid, &proc_id_to_send);
if (proc_id_to_send == 0) {
append(particles, particle);
continue;
}
buffers[proc_id_to_send - 1][counters[proc_id_to_send - 1]] = particle;
counters[proc_id_to_send - 1]++;
if (counters[proc_id_to_send - 1] == NUM_PARTICLES_BUFFER) {
MPI_Send(buffers[proc_id_to_send - 1], SIZEOF_PARTICLE(NUM_PARTICLES_BUFFER), MPI_DOUBLE, proc_id_to_send,
TAG_INIT_PARTICLES, cart_comm);
counters[proc_id_to_send - 1] = 0;
}
}
for (int i = 1; i < number_processors_grid; i++) {
if (counters[i - 1] != 0) {
MPI_Send(buffers[i - 1], SIZEOF_PARTICLE(counters[i - 1]), MPI_DOUBLE, i, TAG_INIT_PARTICLES, cart_comm);
} else {
double endFlag[1] = {-1};
MPI_Send(endFlag, 1, MPI_DOUBLE, i, TAG_INIT_PARTICLES, cart_comm);
}
}
free(counters);
free(buffers[0]);
free(buffers);
}
void receiveParticles() {
int number_elements_received;
MPI_Status status;
while (1) {
MPI_Probe(0, TAG_INIT_PARTICLES, cart_comm, &status);
MPI_Get_count(&status, MPI_DOUBLE, &number_elements_received);
if (number_elements_received == 1) {
break;
}
MPI_Recv((void*)allocate_array(particles, GET_NUMBER_PARTICLE(number_elements_received)),
SIZEOF_PARTICLE(NUM_PARTICLES_BUFFER), MPI_DOUBLE, 0, TAG_INIT_PARTICLES, cart_comm, &status);
MPI_Get_count(&status, MPI_DOUBLE, &number_elements_received);
if (GET_NUMBER_PARTICLE(number_elements_received) != NUM_PARTICLES_BUFFER) {
break;
}
}
}
void calculate_centers_of_mass(int grid_size) {
#pragma omp parallel for
for (int i = 0; i < particles->length; i++) {
particle_t* particle = list_get(particles, i);
int global_cell_index_x = particle->position.x * grid_size;
int global_cell_index_y = particle->position.y * grid_size;
int local_cell_index_x =
CONVERT_TO_LOCAL(my_coordinates[0], size_processor_grid[0], grid_size, global_cell_index_x);
int local_cell_index_y =
CONVERT_TO_LOCAL(my_coordinates[1], size_processor_grid[1], grid_size, global_cell_index_y);
cell_t* cell = &cells[local_cell_index_x][local_cell_index_y];
#pragma omp atomic
cell->mass_sum += particle->mass;
#pragma omp atomic
cell->center_of_mass.x += particle->mass * particle->position.x;
#pragma omp atomic
cell->center_of_mass.y += particle->mass * particle->position.y;
}
for (int i = 1; i <= size_local_cell_matrix[0] - 2; i++) {
for (int j = 1; j <= size_local_cell_matrix[1] - 2; j++) {
cell_t* cell = &cells[i][j];
if (cell->mass_sum != 0) {
cell->center_of_mass.x /= cell->mass_sum;
cell->center_of_mass.y /= cell->mass_sum;
}
}
}
}
void send_recv_centers_of_mass() {
int cells_size_x = size_local_cell_matrix[0] - 2;
int cells_size_y = size_local_cell_matrix[1] - 2;
int number_cells_max = cells_size_x * 2 + cells_size_y * 2 + 4;
for (int i = 0; i < 8; i++) {
if (adjacent_processes[i]->cells_buffer_send == NULL) {
adjacent_processes[i]->cells_buffer_send = (cell_t*)calloc(number_cells_max, sizeof(cell_t));
}
if (adjacent_processes[i]->cells_buffer_recv == NULL) {
adjacent_processes[i]->cells_buffer_recv = (cell_t*)calloc(number_cells_max, sizeof(cell_t));
}
}
for (int i = 7; i >= 0; i--) {
switch (i) {
case DIAGONAL_UP_LEFT:
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[1][1];
adjacent_processes[i]->length_send_buffer++;
break;
case DIAGONAL_UP_RIGHT:
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] =
cells[1][cells_size_y];
adjacent_processes[i]->length_send_buffer++;
break;
case DIAGONAL_DOWN_LEFT:
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] =
cells[cells_size_x][1];
adjacent_processes[i]->length_send_buffer++;
break;
case DIAGONAL_DOWN_RIGHT:
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] =
cells[cells_size_x][cells_size_y];
adjacent_processes[i]->length_send_buffer++;
break;
case LEFT:
for (int j = 1; j <= cells_size_x; j++) {
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[j][1];
adjacent_processes[i]->length_send_buffer++;
}
break;
case RIGHT:
for (int j = 1; j <= cells_size_x; j++) {
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] =
cells[j][cells_size_y];
adjacent_processes[i]->length_send_buffer++;
}
break;
case UP:
for (int j = 1; j <= cells_size_y; j++) {
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[1][j];
adjacent_processes[i]->length_send_buffer++;
}
break;
case DOWN:
for (int j = 1; j <= cells_size_y; j++) {
adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] =
cells[cells_size_x][j];
adjacent_processes[i]->length_send_buffer++;
}
break;
default:
printf("[%d] Default case in send send_recv_centers_of_mass\n", myRank);
fflush(stdout);
break;
}
}
MPI_Request request[8];
MPI_Status status[8];
for (int i = 0; i < 8; i++) {
if (adjacent_processes[i]->received == 1) {
request[i] = MPI_REQUEST_NULL;
continue;
}
MPI_Irecv(adjacent_processes[i]->cells_buffer_recv, SIZEOF_CELL(number_cells_max), MPI_DOUBLE,
adjacent_processes[i]->rank, TAG_SEND_CENTER_OF_MASS, cart_comm, &request[i]);
adjacent_processes[i]->received = 1;
}
for (int i = 0; i < 8; i++) {
if (adjacent_processes[i]->sent == 1) {
continue;
}
MPI_Send(adjacent_processes[i]->cells_buffer_send, SIZEOF_CELL(adjacent_processes[i]->length_send_buffer),
MPI_DOUBLE, adjacent_processes[i]->rank, TAG_SEND_CENTER_OF_MASS, cart_comm);
adjacent_processes[i]->sent = 1;
}
MPI_Waitall(1, request, status);
for (int i = 0; i < 8; i++) {
switch (i) {
case DIAGONAL_UP_LEFT:
cells[0][0] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
break;
case DIAGONAL_UP_RIGHT:
cells[0][cells_size_y + 1] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
break;
case DIAGONAL_DOWN_LEFT:
cells[cells_size_x + 1][0] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
break;
case DIAGONAL_DOWN_RIGHT:
cells[cells_size_x + 1][cells_size_y + 1] =
adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
break;
case LEFT:
for (int j = 1; j <= cells_size_x; j++) {
cells[j][0] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
}
break;
case RIGHT:
for (int j = 1; j <= cells_size_x; j++) {
cells[j][cells_size_y + 1] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
}
break;
case UP:
for (int j = 1; j <= cells_size_y; j++) {
cells[0][j] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
}
break;
case DOWN:
for (int j = 1; j <= cells_size_y; j++) {
cells[cells_size_x + 1][j] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
adjacent_processes[i]->index++;
}
break;
default:
printf("[%d] Default case in send Reconstruct Data\n", myRank);
fflush(stdout);
break;
}
}
for (int i = 0; i < 8; i++) {
adjacent_processes[i]->cells_buffer_send = NULL;
adjacent_processes[i]->cells_buffer_recv = NULL;
adjacent_processes[i]->length_send_buffer = 0;
adjacent_processes[i]->sent = 0;
adjacent_processes[i]->received = 0;
adjacent_processes[i]->index = 0;
}
}
void calculate_new_iteration(int grid_size) {
for (int i = 0; i < particles->length; i++) {
particle_t* particle = list_get(particles, i);
coordinate_t force = {0}, acceleration = {0};
int global_cell_index_x = particle->position.x * grid_size;
int global_cell_index_y = particle->position.y * grid_size;
int local_cell_index_x =
CONVERT_TO_LOCAL(my_coordinates[0], size_processor_grid[0], grid_size, global_cell_index_x);
int local_cell_index_y =
CONVERT_TO_LOCAL(my_coordinates[1], size_processor_grid[1], grid_size, global_cell_index_y);
for (int i = -1; i <= 1; i++) {
for (int j = -1; j <= 1; j++) {
int adjacent_cell_index_x = local_cell_index_x + i;
int adjacent_cell_index_y = local_cell_index_y + j;
cell_t* adjacent_cell = &cells[adjacent_cell_index_x][adjacent_cell_index_y];
coordinate_t force_a_b;
force_a_b.x = adjacent_cell->center_of_mass.x - particle->position.x;
force_a_b.y = adjacent_cell->center_of_mass.y - particle->position.y;
double distance_squared = force_a_b.x * force_a_b.x + force_a_b.y * force_a_b.y;
if (distance_squared < EPSLON * EPSLON) {
continue;
}
double scalar_force =
G * particle->mass * adjacent_cell->mass_sum / (distance_squared * sqrt(distance_squared));
force.x += force_a_b.x * scalar_force;
force.y += force_a_b.y * scalar_force;
}
}
acceleration.x = force.x / particle->mass;
acceleration.y = force.y / particle->mass;
particle->position.x += particle->velocity.x + acceleration.x * 0.5 + 1;
particle->position.y += particle->velocity.y + acceleration.y * 0.5 + 1;
particle->position.x -= (int)particle->position.x;
particle->position.y -= (int)particle->position.y;
particle->velocity.x += acceleration.x;
particle->velocity.y += acceleration.y;
int new_cell_position_x = particle->position.x * grid_size;
int new_cell_position_y = particle->position.y * grid_size;
int processor_coordinates[2];
processor_coordinates[0] = BLOCK_OWNER(new_cell_position_x, size_processor_grid[0], grid_size);
processor_coordinates[1] = BLOCK_OWNER(new_cell_position_y, size_processor_grid[1], grid_size);
int rank;
MPI_Cart_rank(cart_comm, processor_coordinates, &rank);
if (rank != myRank) {
if (processes_buffers[rank].particles_buffer_send == NULL) {
processes_buffers[rank].particles_buffer_send = create_array_list(1024);
}
append(processes_buffers[rank].particles_buffer_send, *particle);
list_remove(particles, i);
i--;
}
}
}
void send_recv_particles() {
MPI_Request* request = malloc(sizeof(MPI_Request) * number_processors);
for (int i = 0; i < number_processors; i++) {
if (i == myRank) {
continue;
}
if (processes_buffers[i].sent == 1) {
request[i] = MPI_REQUEST_NULL;
continue;
}
if (processes_buffers[i].particles_buffer_send == NULL) {
processes_buffers[i].particles_buffer_send = create_array_list(1024);
}
if (processes_buffers[i].particles_buffer_send->length == 0) {
MPI_Isend(processes_buffers[i].particles_buffer_send->particles, 1, MPI_DOUBLE, i, TAG_SEND_PARTICLES,
cart_comm, &request[i]);
} else {
MPI_Isend(processes_buffers[i].particles_buffer_send->particles,
SIZEOF_PARTICLE(processes_buffers[i].particles_buffer_send->length), MPI_DOUBLE, i,
TAG_SEND_PARTICLES, cart_comm, &request[i]);
}
processes_buffers[i].sent = 1;
}
free(request);
for (int i = 0; i < number_processors; i++) {
if (i == myRank || processes_buffers[i].received == 1) {
continue;
}
MPI_Status status;
int number_elements_received;
MPI_Probe(i, TAG_SEND_PARTICLES, cart_comm, &status);
MPI_Get_count(&status, MPI_DOUBLE, &number_elements_received);
if (number_elements_received == 1) {
double buffer[1];
MPI_Recv(buffer, 1, MPI_DOUBLE, i, TAG_SEND_PARTICLES, cart_comm, &status);
processes_buffers[i].received = 1;
continue;
}
MPI_Recv((void*)allocate_array(particles, GET_NUMBER_PARTICLE(number_elements_received)),
number_elements_received, MPI_DOUBLE, i, TAG_SEND_PARTICLES, cart_comm, &status);
MPI_Get_count(&status, MPI_DOUBLE, &number_elements_received);
processes_buffers[i].received = 1;
}
for (int i = 0; i < number_processors; i++) {
if (processes_buffers[i].particles_buffer_send != NULL) {
list_free(processes_buffers[i].particles_buffer_send);
processes_buffers[i].particles_buffer_send = NULL;
}
processes_buffers[i].received = 0;
processes_buffers[i].sent = 0;
}
}
void calculate_overall_center_of_mass() {
double center_of_mass_x = 0, center_of_mass_y = 0;
double total_mass = 0;
#pragma omp parallel for reduction(+ : total_mass, center_of_mass_x, center_of_mass_y)
for (int i = 0; i < particles->length; i++) {
particle_t* particle = list_get(particles, i);
if ((int)particle->index == 0) {
printf("%.2f %.2f\n", particle->position.x, particle->position.y);
fflush(stdout);
}
total_mass += particle->mass;
center_of_mass_x += particle->mass * particle->position.x;
center_of_mass_y += particle->mass * particle->position.y;
}
double global_total_mass;
double global_center_of_mass_x;
double global_center_of_mass_y;
MPI_Reduce(&total_mass, &global_total_mass, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
MPI_Reduce(&center_of_mass_x, &global_center_of_mass_x, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
MPI_Reduce(&center_of_mass_y, &global_center_of_mass_y, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
if (myRank == 0) {
center_of_mass_x = global_center_of_mass_x / global_total_mass;
center_of_mass_y = global_center_of_mass_y / global_total_mass;
printf("%.2f %.2f\n", center_of_mass_x, center_of_mass_y);
}
}
int main(int argc, char* argv[]) {
if (argc != 5) {
printf("Expected 5 arguments, but %d were given\n", argc);
exit(1);
}
int grid_size = atoi(argv[2]);
int number_particles = atoi(argv[3]);
int n_time_steps = atoi(argv[4]);
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &number_processors);
MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
particles = create_array_list(2048);
create_cartesian_communicator(grid_size);
init_cells(grid_size);
if (myRank == 0) {
init_particles(atoi(argv[1]), grid_size, number_particles);
} else {
receiveParticles();
}
for (int n = 0; n < n_time_steps; n++) {
calculate_centers_of_mass(grid_size);
send_recv_centers_of_mass();
calculate_new_iteration(grid_size);
send_recv_particles();
memset(cells[0], 0, sizeof(cell_t) * size_local_cell_matrix[0] * size_local_cell_matrix[1]);
}
calculate_overall_center_of_mass();
MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
return 0;
}