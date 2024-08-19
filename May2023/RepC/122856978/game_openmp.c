#define _DEFAULT_SOURCE
#define GEN_LIMIT 1000
#define CHECK_SIMILARITY
#define SIMILARITY_FREQUENCY 3
#define THREADS 4
#define true 1
#define false 0
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
void perror_exit(const char *message)
{
perror(message);
exit(EXIT_FAILURE);
}
void evolve(char **local, char **new, int width_local, int height_local)
{
#pragma omp parallel for num_threads(THREADS) firstprivate(local, height_local, width_local)
for (int y = 1; y <= height_local; y++)
{
for (int x = 1; x <= width_local; x++)
{
int neighbors = 0;
neighbors = local[y - 1][x - 1] + local[y - 1][x] +
local[y - 1][x + 1] + local[y][x - 1] +
local[y][x + 1] + local[y + 1][x - 1] +
local[y + 1][x] + local[y + 1][x + 1];
if (neighbors == 387 || (neighbors == 386 && (local[y][x] == '1')))
new[y][x] = '1';
else
new[y][x] = '0';
}
}
}
int empty(char **local, int width_local, int height_local)
{
int result = 0;
#pragma omp parallel for reduction(+ : result) num_threads(THREADS) schedule(dynamic) firstprivate(local, height_local, width_local)
for (int y = 1; y <= height_local; y++)
{
for (int x = 1; x <= width_local; x++)
{
if (local[y][x] == '1')
{
result++;
break;
}
}
}
return (!result);
}
int empty_all(char **local, int width_local, int height_local, MPI_Comm *new_comm, int comm_sz)
{
int local_flag = empty(local, width_local, height_local),
global_sum;
MPI_Allreduce(&local_flag, &global_sum, 1, MPI_INT, MPI_SUM, *new_comm);
return (global_sum == comm_sz);
}
int similarity(char **local, char **local_old, int width_local, int height_local)
{
int result = 0;
#pragma omp parallel for reduction(+ : result) num_threads(THREADS) schedule(dynamic) firstprivate(local, local_old, height_local, width_local)
for (int y = 1; y <= height_local; y++)
{
for (int x = 1; x <= width_local; x++)
{
if (local_old[y][x] != local[y][x])
{
result++;
break;
}
}
}
return (!result);
}
int similarity_all(char **local, char **local_old, int width_local, int height_local, MPI_Comm *new_comm, int comm_sz)
{
int local_flag = similarity(local, local_old, width_local, height_local),
global_sum;
MPI_Allreduce(&local_flag, &global_sum, 1, MPI_INT, MPI_SUM, *new_comm);
return (global_sum == comm_sz);
}
void game(int width, int height, char *fileArg)
{
int my_rank, comm_sz;
MPI_Init(NULL, NULL);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm old_comm, new_comm;
int ndims, reorder, periods[2], dim_size[2];
old_comm = MPI_COMM_WORLD;
ndims = 2; 
int rows_columns = (int)sqrt(comm_sz);
dim_size[0] = rows_columns; 
dim_size[1] = rows_columns; 
periods[0] = 1;             
periods[1] = 1;             
reorder = 1;                
MPI_Cart_create(old_comm, ndims, dim_size, periods, reorder, &new_comm);
int me, coords[2];
MPI_Comm_rank(new_comm, &me);
MPI_Cart_coords(new_comm, me, ndims, coords);
int width_local, height_local;
width_local = height_local = width / rows_columns;
char **local = malloc((width_local + 2) * sizeof(char *));
char *b = malloc((width_local + 2) * (height_local + 2) * sizeof(char));
if (local == NULL || b == NULL)
perror_exit("malloc: ");
for (int i = 0; i < (width_local + 2); i++)
local[i] = &b[i * (height_local + 2)];
char **new = malloc((width_local + 2) * sizeof(char *));
char *a = malloc((width_local + 2) * (height_local + 2) * sizeof(char));
if (new == NULL || a == NULL)
perror_exit("malloc: ");
for (int i = 0; i < (width_local + 2); i++)
new[i] = &a[i * (height_local + 2)];
char **local_input = malloc(height_local * sizeof(char *));
char *d = malloc(width_local * height_local * sizeof(char));
if (local_input == NULL || d == NULL)
perror_exit("malloc: ");
for (int i = 0; i < height_local; ++i)
local_input[i] = &d[i * width_local];
MPI_File fh;
MPI_Datatype sub_array;
MPI_Status status;
double t_start = MPI_Wtime();
int sub_size[2];
int whole_size[2];
int start_indices[2];
sub_size[0] = height_local; 
sub_size[1] = width_local;
whole_size[0] = height; 
whole_size[1] = width + 1;
start_indices[0] = coords[0] * height_local; 
start_indices[1] = coords[1] * width_local;
MPI_Type_create_subarray(ndims, whole_size, sub_size, start_indices, MPI_ORDER_C, MPI_CHAR, &sub_array);
MPI_Type_commit(&sub_array);
MPI_File_open(new_comm, fileArg, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
MPI_File_set_view(fh, 0, MPI_CHAR, sub_array, "native", MPI_INFO_NULL);
MPI_File_read_all(fh, &local_input[0][0], (height_local * width_local), MPI_CHAR, &status);
MPI_File_close(&fh);
MPI_Type_free(&sub_array);
double msecs = (MPI_Wtime() - t_start) * 1000;
if (me == 0)
printf("Reading file:\t%.2lf msecs\n", msecs);
for (int i = 0; i < height_local; i++)
{
for (int j = 0; j < width_local; j++)
{
local[i + 1][j + 1] = local_input[i][j];
}
}
free(local_input); 
free(d);
local_input = NULL;
d = NULL;
int generation = 1;
#ifdef CHECK_SIMILARITY
int counter = 0;
#endif
int north;
int south;
int east;
int west;
int north_coords[2];
int south_coords[2];
int west_coords[2];
int east_coords[2];
north_coords[0] = coords[0] + 1;
south_coords[0] = coords[0] - 1;
west_coords[0] = coords[0];
east_coords[0] = coords[0];
north_coords[1] = coords[1];
south_coords[1] = coords[1];
west_coords[1] = coords[1] - 1;
east_coords[1] = coords[1] + 1;
int north_west;
int north_east;
int south_west;
int south_east;
int north_west_coords[2];
int north_east_coords[2];
int south_west_coords[2];
int south_east_coords[2];
north_west_coords[0] = coords[0] - 1;
north_east_coords[0] = coords[0] - 1;
south_west_coords[0] = coords[0] + 1;
south_east_coords[0] = coords[0] + 1;
north_west_coords[1] = coords[1] - 1;
north_east_coords[1] = coords[1] + 1;
south_west_coords[1] = coords[1] - 1;
south_east_coords[1] = coords[1] + 1;
MPI_Cart_rank(new_comm, north_coords, &north);
MPI_Cart_rank(new_comm, south_coords, &south);
MPI_Cart_rank(new_comm, west_coords, &west);
MPI_Cart_rank(new_comm, east_coords, &east);
MPI_Cart_rank(new_comm, north_west_coords, &north_west);
MPI_Cart_rank(new_comm, north_east_coords, &north_east);
MPI_Cart_rank(new_comm, south_west_coords, &south_west);
MPI_Cart_rank(new_comm, south_east_coords, &south_east);
MPI_Datatype vertical_type;
MPI_Type_vector(height_local, 1, width_local + 2, MPI_CHAR, &vertical_type);
MPI_Type_commit(&vertical_type);
MPI_Request requests_odd[16];
MPI_Request requests_even[16];
MPI_Recv_init(&local[0][1], width_local, MPI_CHAR, north, 1, new_comm, &requests_odd[0]);
MPI_Send_init(&local[1][1], width_local, MPI_CHAR, north, 2, new_comm, &requests_odd[1]);
MPI_Recv_init(&local[height_local + 1][1], width_local, MPI_CHAR, south, 2, new_comm, &requests_odd[2]);
MPI_Send_init(&local[height_local][1], width_local, MPI_CHAR, south, 1, new_comm, &requests_odd[3]);
MPI_Recv_init(&local[1][width_local + 1], 1, vertical_type, east, 3, new_comm, &requests_odd[4]);
MPI_Send_init(&local[1][width_local], 1, vertical_type, east, 4, new_comm, &requests_odd[5]);
MPI_Recv_init(&local[1][0], 1, vertical_type, west, 4, new_comm, &requests_odd[6]);
MPI_Send_init(&local[1][1], 1, vertical_type, west, 3, new_comm, &requests_odd[7]);
MPI_Recv_init(&local[0][0], 1, MPI_CHAR, north_west, 5, new_comm, &requests_odd[8]);
MPI_Send_init(&local[1][1], 1, MPI_CHAR, north_west, 6, new_comm, &requests_odd[9]);
MPI_Recv_init(&local[0][width_local + 1], 1, MPI_CHAR, north_east, 7, new_comm, &requests_odd[10]);
MPI_Send_init(&local[1][width_local], 1, MPI_CHAR, north_east, 8, new_comm, &requests_odd[11]);
MPI_Recv_init(&local[height_local + 1][0], 1, MPI_CHAR, south_west, 8, new_comm, &requests_odd[12]);
MPI_Send_init(&local[height_local][1], 1, MPI_CHAR, south_west, 7, new_comm, &requests_odd[13]);
MPI_Recv_init(&local[height_local + 1][width_local + 1], 1, MPI_CHAR, south_east, 6, new_comm, &requests_odd[14]);
MPI_Send_init(&local[height_local][width_local], 1, MPI_CHAR, south_east, 5, new_comm, &requests_odd[15]);
MPI_Recv_init(&new[0][1], width_local, MPI_CHAR, north, 1, new_comm, &requests_even[0]);
MPI_Send_init(&new[1][1], width_local, MPI_CHAR, north, 2, new_comm, &requests_even[1]);
MPI_Recv_init(&new[height_local + 1][1], width_local, MPI_CHAR, south, 2, new_comm, &requests_even[2]);
MPI_Send_init(&new[height_local][1], width_local, MPI_CHAR, south, 1, new_comm, &requests_even[3]);
MPI_Recv_init(&new[1][width_local + 1], 1, vertical_type, east, 3, new_comm, &requests_even[4]);
MPI_Send_init(&new[1][width_local], 1, vertical_type, east, 4, new_comm, &requests_even[5]);
MPI_Recv_init(&new[1][0], 1, vertical_type, west, 4, new_comm, &requests_even[6]);
MPI_Send_init(&new[1][1], 1, vertical_type, west, 3, new_comm, &requests_even[7]);
MPI_Recv_init(&new[0][0], 1, MPI_CHAR, north_west, 5, new_comm, &requests_even[8]);
MPI_Send_init(&new[1][1], 1, MPI_CHAR, north_west, 6, new_comm, &requests_even[9]);
MPI_Recv_init(&new[0][width_local + 1], 1, MPI_CHAR, north_east, 7, new_comm, &requests_even[10]);
MPI_Send_init(&new[1][width_local], 1, MPI_CHAR, north_east, 8, new_comm, &requests_even[11]);
MPI_Recv_init(&new[height_local + 1][0], 1, MPI_CHAR, south_west, 8, new_comm, &requests_even[12]);
MPI_Send_init(&new[height_local][1], 1, MPI_CHAR, south_west, 7, new_comm, &requests_even[13]);
MPI_Recv_init(&new[height_local + 1][width_local + 1], 1, MPI_CHAR, south_east, 6, new_comm, &requests_even[14]);
MPI_Send_init(&new[height_local][width_local], 1, MPI_CHAR, south_east, 5, new_comm, &requests_even[15]);
t_start = MPI_Wtime();
while ((!empty_all(local, width_local, height_local, &new_comm, comm_sz)) && (generation <= GEN_LIMIT))
{
if ((generation % 2) == 1)
{
MPI_Startall(16, requests_odd);
MPI_Waitall(16, requests_odd, MPI_STATUSES_IGNORE);
}
else
{
MPI_Startall(16, requests_even);
MPI_Waitall(16, requests_even, MPI_STATUSES_IGNORE);
}
evolve(local, new, width_local, height_local);
char **temp_array = local;
local = new;
new = temp_array;
#ifdef CHECK_SIMILARITY
counter++;
if (counter == SIMILARITY_FREQUENCY)
{
if (similarity_all(local, new, width_local, height_local, &new_comm, comm_sz))
break;
counter = 0;
}
#endif
generation++;
} 
msecs = (MPI_Wtime() - t_start) * 1000;
if (me == 0) 
printf("Generations:\t%d\nExecution time:\t%.2lf msecs\n", generation - 1, msecs);
free(a);
free(new);
a = NULL;
new = NULL;
MPI_Type_free(&vertical_type);
char **local_finished;
char *c;
if (coords[1] == rows_columns - 1) 
{
local_finished = malloc(height_local * sizeof(char *));
c = malloc((width_local + 1) * height_local * sizeof(char)); 
if (local_finished == NULL || c == NULL)
perror_exit("malloc: ");
for (int i = 0; i < height_local; ++i)
local_finished[i] = &c[i * (width_local + 1)];
for (int i = 0; i < height_local; i++)
local_finished[i][width_local] = '\n';
}
else
{
local_finished = malloc(height_local * sizeof(char *));
c = malloc(width_local * height_local * sizeof(char));
if (local_finished == NULL || c == NULL)
perror_exit("malloc: ");
for (int i = 0; i < height_local; ++i)
local_finished[i] = &c[i * width_local];
}
for (int i = 0; i < width_local; i++) 
{
for (int j = 0; j < height_local; j++)
{
local_finished[i][j] = local[i + 1][j + 1];
}
}
whole_size[0] = height; 
whole_size[1] = width + 1;
start_indices[0] = coords[0] * height_local; 
start_indices[1] = coords[1] * width_local;
if (coords[1] == rows_columns - 1) 
width_local++;                 
sub_size[0] = height_local; 
sub_size[1] = width_local;
t_start = MPI_Wtime();
MPI_Type_create_subarray(ndims, whole_size, sub_size, start_indices, MPI_ORDER_C, MPI_CHAR, &sub_array); 
MPI_Type_commit(&sub_array); 
int err = MPI_File_open(new_comm, "./openmp_output.out", MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &fh); 
if (err != MPI_SUCCESS) 
{
if (me == 0) 
MPI_File_delete("./openmp_output.out", MPI_INFO_NULL);
MPI_File_open(new_comm, "./openmp_output.out", MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &fh); 
}
MPI_File_set_view(fh, 0, MPI_CHAR, sub_array, "native", MPI_INFO_NULL);
MPI_File_write_all(fh, &local_finished[0][0], (height_local * width_local), MPI_CHAR, &status);
MPI_File_close(&fh);
MPI_Type_free(&sub_array);
msecs = (MPI_Wtime() - t_start) * 1000;
if (me == 0)
printf("Writing file:\t%.2lf msecs\n", msecs);
free(b);
free(local);
b = NULL;
local = NULL;
free(c);
free(local_finished);
c = NULL;
local_finished = NULL;
MPI_Finalize();
}
int main(int argc, char *argv[])
{
int width = 0, height = 0;
if (argc > 1)
width = atoi(argv[1]);
if (argc > 2)
height = atoi(argv[2]);
height = width;
if (width <= 0)
width = 30;
if (height <= 0)
height = 30;
if (argc > 3)
game(width, height, argv[3]);
return 0;
}
