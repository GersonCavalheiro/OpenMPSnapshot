#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>
#include "utils.h"
int main(void){
MPI_Datatype args_type, filter_type, filter_type1; 
MPI_Status recv_stat; 
Args_type my_args; 
int comm_size, my_rank, error;
int i, j, k, iter, index;
#ifdef CHECK_CONVERGENCE
int print_message = 0, all_finished, equality_flag = 0; 
#endif
MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
int procs_per_line = (int)sqrt(comm_size); 
if(my_rank == 0){
if(procs_per_line * procs_per_line != comm_size){
printf("Invalid number of processes given. Must be a perfect square: 4, 9, 16,...\n");
MPI_Abort(MPI_COMM_WORLD, -1);
}
if(comm_size <= 0 || comm_size > PROCESSES_LIMIT){
printf("Invalid number of processes given. Must be a positive heigher than 0 and less than %d\n",PROCESSES_LIMIT);
MPI_Abort(MPI_COMM_WORLD, -1);
}
}
MPI_Comm old_comm, my_cartesian_comm;
int ndims, reorder, periods[2], dim_size[2];
old_comm = MPI_COMM_WORLD;
ndims = 2;
dim_size[0] = procs_per_line;
dim_size[1] = procs_per_line;
periods[0] = 0;
periods[1] = 0;
reorder = 1;
MPI_Cart_create(old_comm,ndims,dim_size,periods,reorder,&my_cartesian_comm);
MPI_Type_contiguous(FILTER_SIZE, MPI_DOUBLE, &filter_type1); 
MPI_Type_commit(&filter_type1);
MPI_Type_contiguous(FILTER_SIZE, filter_type1, &filter_type); 
MPI_Type_commit(&filter_type);
const int items = 10;
int blocklengths[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1,1};
MPI_Datatype types[10] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, filter_type,
MPI_INT, MPI_INT, MPI_INT, MPI_INT,MPI_INT};
MPI_Aint offsets[10];
offsets[0] = offsetof(Args_type, image_type);
offsets[1] = offsetof(Args_type, image_width);
offsets[2] = offsetof(Args_type, image_height);
offsets[3] = offsetof(Args_type, image_seed);
offsets[4] = offsetof(Args_type, filter);
offsets[5] = offsetof(Args_type, width_per_process);
offsets[6] = offsetof(Args_type, width_remaining);
offsets[7] = offsetof(Args_type, height_per_process);
offsets[8] = offsetof(Args_type, height_remaining);
offsets[9] = offsetof(Args_type, iterations);
MPI_Type_create_struct(items, blocklengths, offsets, types, &args_type);
MPI_Type_commit(&args_type);
if(my_rank == 0){
error = read_user_input(&my_args, procs_per_line);
if(error != 0)
MPI_Abort(my_cartesian_comm, error);
my_args.width_per_process = my_args.image_width / (int)sqrt(comm_size);
my_args.width_remaining = my_args.image_width % (int)sqrt(comm_size);
my_args.height_per_process = my_args.image_height / (int)sqrt(comm_size);
my_args.height_remaining = my_args.image_height % (int)sqrt(comm_size);
for(i = 1; i < comm_size; i++)
MPI_Send(&my_args, 1, args_type, i, 1, my_cartesian_comm);
}
else{
MPI_Recv(&my_args, 1, args_type, 0, 1, my_cartesian_comm, &recv_stat);
}
int neighbours[NUM_NEIGHBOURS]; 
int row_id = my_rank % procs_per_line;
int column_id = my_rank / procs_per_line;
MPI_Cart_shift(my_cartesian_comm, 0, 1, &neighbours[N], &neighbours[S]);
MPI_Cart_shift(my_cartesian_comm, 1, 1, &neighbours[W], &neighbours[E]);
if(column_id != 0 && row_id != procs_per_line - 1) 
neighbours[NE] = my_rank - procs_per_line + 1;
else
neighbours[NE] = MPI_PROC_NULL;
if(column_id != procs_per_line - 1 && row_id != procs_per_line - 1) 
neighbours[SE] = my_rank + procs_per_line + 1;
else
neighbours[SE] = MPI_PROC_NULL;
if(column_id != procs_per_line -1 && row_id != 0) 
neighbours[SW] = my_rank + procs_per_line - 1;
else
neighbours[SW] = MPI_PROC_NULL;
if(row_id != 0 && column_id != 0) 
neighbours[NW] = my_rank - procs_per_line - 1;
else
neighbours[NW] = MPI_PROC_NULL;
int mult; 
int my_width, my_width_incr_1, my_width_decr_1, my_width_incr_2;
int my_height, my_height_incr_1, my_height_decr_1, my_height_incr_2;
int mult_multi_2;
mult = (my_args.image_type == 0) ? 1 : 3; 
mult_multi_2 = mult * 2;
if(row_id < my_args.width_remaining)
my_width = (my_args.width_per_process + 1) * mult; 
else
my_width = my_args.width_per_process * mult;
if(column_id < my_args.height_remaining)
my_height = my_args.height_per_process + 1;
else
my_height = my_args.height_per_process;
my_width_incr_1 = my_width + mult;
my_height_incr_1 = my_height + 1;
my_width_incr_2 = my_width_incr_1 + mult;
my_height_incr_2 = my_height_incr_1 + 1;
my_width_decr_1 = my_width - mult;
my_height_decr_1 = my_height - 1;
srand(my_args.image_seed * ((my_rank + 333) * (my_rank + 333)));
int** my_image_before, **my_image_after; 
my_image_before = malloc((my_height_incr_2) * sizeof(int*));
if(my_image_before == NULL)
MPI_Abort(my_cartesian_comm, error);
my_image_before[0] = malloc((my_height_incr_2) * (my_width_incr_2) * sizeof(int));
if(my_image_before[0] == NULL)
MPI_Abort(my_cartesian_comm, error);
for(i = 1; i < (my_height_incr_2); i++)
my_image_before[i] = &(my_image_before[0][i*(my_width_incr_2)]);
for(i = 1; i <  my_height_incr_1; i++)
for(j = mult; j < my_width_incr_1; j++)
my_image_before[i][j] = rand() % 256;
for(i = 0; i < my_height_incr_2; i++){
for(j = 0; j < mult; j++){
my_image_before[i][j] = my_image_before[i][mult + j];
my_image_before[i][my_width_incr_1 + j] = my_image_before[i][my_width + j];
}
}
for(j = 0; j < my_width_incr_2; j++){
my_image_before[0][j] = my_image_before[1][j];
my_image_before[my_height_incr_1][j] = my_image_before[my_height][j];
}
my_image_after = malloc((my_height_incr_2) * sizeof(int*));
if(my_image_after == NULL)
MPI_Abort(my_cartesian_comm, error);
my_image_after[0] = malloc((my_height_incr_2) * (my_width_incr_2) * sizeof(int));
if(my_image_after[0] == NULL)
MPI_Abort(my_cartesian_comm, error);
for(i = 1; i < (my_height_incr_2); i++)
my_image_after[i] = &(my_image_after[0][i*(my_width_incr_2)]);
for(i = 0; i < my_height_incr_2; i++){
for(j = 0; j < mult; j++){
my_image_after[i][j] = my_image_before[i][mult + j];
my_image_after[i][my_width_incr_1 + j] = my_image_before[i][my_width + j];
}
}
for(j = 0; j < my_width_incr_2; j++){
my_image_after[0][j] = my_image_before[1][j];
my_image_after[my_height_incr_1][j] = my_image_before[my_height][j];
}
MPI_Datatype column_type;
MPI_Type_vector(my_height, mult, my_width_incr_2, MPI_INT, &column_type);
MPI_Type_commit(&column_type);
MPI_Request send_after_requests[NUM_NEIGHBOURS];
MPI_Request send_before_requests[NUM_NEIGHBOURS];
MPI_Request recv_after_requests[NUM_NEIGHBOURS];
MPI_Request recv_before_requests[NUM_NEIGHBOURS];
MPI_Send_init(&my_image_after[1][mult], my_width, MPI_INT, neighbours[N], S, my_cartesian_comm, &send_after_requests[N]);
MPI_Send_init(&my_image_after[1][my_width], mult, MPI_INT, neighbours[NE], SW, my_cartesian_comm, &send_after_requests[NE]);
MPI_Send_init(&my_image_after[1][my_width], 1, column_type, neighbours[E], W, my_cartesian_comm, &send_after_requests[E]);
MPI_Send_init(&my_image_after[my_height][my_width], mult, MPI_INT, neighbours[SE], NW, my_cartesian_comm, &send_after_requests[SE]);
MPI_Send_init(&my_image_after[my_height][mult], my_width, MPI_INT, neighbours[S], N, my_cartesian_comm, &send_after_requests[S]);
MPI_Send_init(&my_image_after[my_height][mult], mult, MPI_INT, neighbours[SW], NE, my_cartesian_comm, &send_after_requests[SW]);
MPI_Send_init(&my_image_after[1][mult], 1, column_type, neighbours[W], E, my_cartesian_comm, &send_after_requests[W]);
MPI_Send_init(&my_image_after[1][mult], mult, MPI_INT, neighbours[NW], SE, my_cartesian_comm, &send_after_requests[NW]);
MPI_Send_init(&my_image_before[1][mult], my_width, MPI_INT, neighbours[N], S, my_cartesian_comm, &send_before_requests[N]);
MPI_Send_init(&my_image_before[1][my_width], mult, MPI_INT, neighbours[NE], SW, my_cartesian_comm, &send_before_requests[NE]);
MPI_Send_init(&my_image_before[1][my_width], 1, column_type, neighbours[E], W, my_cartesian_comm, &send_before_requests[E]);
MPI_Send_init(&my_image_before[my_height][my_width], mult, MPI_INT, neighbours[SE], NW, my_cartesian_comm, &send_before_requests[SE]);
MPI_Send_init(&my_image_before[my_height][mult], my_width, MPI_INT, neighbours[S], N, my_cartesian_comm, &send_before_requests[S]);
MPI_Send_init(&my_image_before[my_height][mult], mult, MPI_INT, neighbours[SW], NE, my_cartesian_comm, &send_before_requests[SW]);
MPI_Send_init(&my_image_before[1][mult], 1, column_type, neighbours[W], E, my_cartesian_comm, &send_before_requests[W]);
MPI_Send_init(&my_image_before[1][mult], mult, MPI_INT, neighbours[NW], SE, my_cartesian_comm, &send_before_requests[NW]);
MPI_Recv_init(&my_image_after[0][mult], my_width, MPI_INT, neighbours[N], N, my_cartesian_comm, &recv_after_requests[N]);
MPI_Recv_init(&my_image_after[0][my_width_incr_1], mult, MPI_INT, neighbours[NE], NE, my_cartesian_comm, &recv_after_requests[NE]);
MPI_Recv_init(&my_image_after[1][my_width_incr_1], 1, column_type, neighbours[E], E, my_cartesian_comm, &recv_after_requests[E]);
MPI_Recv_init(&my_image_after[my_height_incr_1][my_width_incr_1], mult, MPI_INT, neighbours[SE], SE, my_cartesian_comm, &recv_after_requests[SE]);
MPI_Recv_init(&my_image_after[my_height_incr_1][mult], my_width, MPI_INT, neighbours[S], S, my_cartesian_comm, &recv_after_requests[S]);
MPI_Recv_init(&my_image_after[my_height_incr_1][0], mult, MPI_INT, neighbours[SW],SW, my_cartesian_comm, &recv_after_requests[SW]);
MPI_Recv_init(&my_image_after[1][0], 1, column_type, neighbours[W], W, my_cartesian_comm, &recv_after_requests[W]);
MPI_Recv_init(&my_image_after[0][0], mult, MPI_INT, neighbours[NW], NW, my_cartesian_comm, &recv_after_requests[NW]);
MPI_Recv_init(&my_image_before[0][mult], my_width, MPI_INT, neighbours[N], N, my_cartesian_comm, &recv_before_requests[N]);
MPI_Recv_init(&my_image_before[0][my_width_incr_1], mult, MPI_INT, neighbours[NE], NE, my_cartesian_comm, &recv_before_requests[NE]);
MPI_Recv_init(&my_image_before[1][my_width_incr_1], 1, column_type, neighbours[E], E, my_cartesian_comm, &recv_before_requests[E]);
MPI_Recv_init(&my_image_before[my_height_incr_1][my_width_incr_1], mult, MPI_INT, neighbours[SE], SE, my_cartesian_comm, &recv_before_requests[SE]);
MPI_Recv_init(&my_image_before[my_height_incr_1][mult], my_width, MPI_INT, neighbours[S], S, my_cartesian_comm, &recv_before_requests[S]);
MPI_Recv_init(&my_image_before[my_height_incr_1][0], mult, MPI_INT, neighbours[SW],SW, my_cartesian_comm, &recv_before_requests[SW]);
MPI_Recv_init(&my_image_before[1][0], 1, column_type, neighbours[W], W, my_cartesian_comm, &recv_before_requests[W]);
MPI_Recv_init(&my_image_before[0][0], mult, MPI_INT, neighbours[NW], NW, my_cartesian_comm, &recv_before_requests[NW]);
MPI_Request* send_requests[2];
MPI_Request* recv_requests[2];
send_requests[0] = send_before_requests;
send_requests[1] = send_after_requests;
recv_requests[0] = recv_before_requests;
recv_requests[1] = recv_after_requests;
int flag_corner_ul = 0, flag_corner_ur = 0, flag_corner_ll = 0, flag_corner_lr = 0;
MPI_Barrier(my_cartesian_comm);
double start = MPI_Wtime(); 
int r_index = -1; 
int*** im_before = &my_image_before;
int*** im_after = &my_image_after;
int*** tmp = NULL;
for(iter = 0; iter < my_args.iterations; iter++){
r_index = iter % 2;
flag_corner_ul = 0;
flag_corner_ur = 0;
flag_corner_ll = 0;
flag_corner_lr = 0;
MPI_Startall(NUM_NEIGHBOURS, send_requests[r_index]);
#ifdef ENABLE_OPEN_MP
#pragma omp parallel for num_threads(NUM_THREADS) collapse(2) schedule(static, (my_width - 2) * (my_height - 2) / NUM_THREADS)
#endif
for(i = 2; i < my_height; i++){ 
for(j = 2 * mult; j < my_width; j++){ 
(*im_after)[i][j] = (int)((*im_before)[i][j] * my_args.filter[1][1] +
(*im_before)[i - 1][j] * my_args.filter[0][1] +
(*im_before)[i - 1][j + mult] * my_args.filter[0][2] +
(*im_before)[i][j + mult] * my_args.filter[1][2] +
(*im_before)[i + 1][j + mult] * my_args.filter[2][2] +
(*im_before)[i + 1][j] * my_args.filter[2][1] +
(*im_before)[i + 1][j - mult] * my_args.filter[2][0] +
(*im_before)[i][j - mult] * my_args.filter[1][0] +
(*im_before)[i - 1][j - mult] * my_args.filter[0][0]);
if((*im_after)[i][j] < 0)
(*im_after)[i][j] = 0;
else if((*im_after)[i][j] > 255)
(*im_after)[i][j] = 255;
} 
} 
MPI_Startall(NUM_NEIGHBOURS, recv_requests[r_index]);
for(k = 0; k < NUM_NEIGHBOURS; k++){
MPI_Waitany(NUM_NEIGHBOURS, recv_requests[r_index], &index, &recv_stat);
if(index == N){
flag_corner_ul++;
flag_corner_ur++;
for(j = 2 * mult; j < my_width; j++){
(*im_after)[1][j] = (int)((*im_before)[1][j] * my_args.filter[1][1] +
(*im_before)[0][j] * my_args.filter[0][1] +
(*im_before)[0][j + mult] * my_args.filter[0][2] +
(*im_before)[1][j + mult] * my_args.filter[1][2] +
(*im_before)[2][j + mult] * my_args.filter[2][2] +
(*im_before)[2][j] * my_args.filter[2][1] +
(*im_before)[2][j - mult] * my_args.filter[2][0] +
(*im_before)[1][j - mult] * my_args.filter[1][0] +
(*im_before)[0][j - mult] * my_args.filter[0][0]);
if((*im_after)[1][j] < 0)
(*im_after)[1][j] = 0;
else if((*im_after)[1][j] > 255)
(*im_after)[1][j] = 255;
} 
} 
else if(index == NE){
flag_corner_ur++;
} 
else if(index == E){
flag_corner_ur++;
flag_corner_lr++;
for(i = 2; i < my_height; i++){
for(j = 0; j < mult; j++){  
(*im_after)[i][my_width + j] = (int)((*im_before)[i][my_width + j] * my_args.filter[1][1] +
(*im_before)[i - 1][my_width + j] * my_args.filter[0][1] +
(*im_before)[i - 1][my_width_incr_1 + j] * my_args.filter[0][2] +
(*im_before)[i][my_width_incr_1 + j] * my_args.filter[1][2] +
(*im_before)[i + 1][my_width_incr_1 + j] * my_args.filter[2][2] +
(*im_before)[i + 1][my_width + j] * my_args.filter[2][1] +
(*im_before)[i + 1][my_width_decr_1 + j] * my_args.filter[2][0] +
(*im_before)[i][my_width_decr_1 + j] * my_args.filter[1][0] +
(*im_before)[i - 1][my_width_decr_1 + j] * my_args.filter[0][0]);
if((*im_after)[i][my_width + j] < 0)
(*im_after)[i][my_width + j] = 0;
else if((*im_after)[i][my_width + j] > 255)
(*im_after)[i][my_width + j] = 255;
} 
} 
} 
else if(index == SE){
flag_corner_lr++;
} 
else if(index == S){
flag_corner_ll++;
flag_corner_lr++;
for(j = 2 * mult; j < my_width; j++){
(*im_after)[my_height][j] = (int)((*im_before)[my_height][j] * my_args.filter[1][1] +
(*im_before)[my_height_decr_1][j] * my_args.filter[0][1] +
(*im_before)[my_height_decr_1][j + mult] * my_args.filter[0][2] +
(*im_before)[my_height][j + mult] * my_args.filter[1][2] +
(*im_before)[my_height_incr_1][j + mult] * my_args.filter[2][2] +
(*im_before)[my_height_incr_1][j] * my_args.filter[2][1] +
(*im_before)[my_height_incr_1][j - mult] * my_args.filter[2][0] +
(*im_before)[my_height][j - mult] * my_args.filter[1][0] +
(*im_before)[my_height_decr_1][j - mult] * my_args.filter[0][0]);
if((*im_after)[my_height][j] < 0)
(*im_after)[my_height][j] = 0;
else if((*im_after)[my_height][j] > 255)
(*im_after)[my_height][j] = 255;
} 
} 
else if (index == SW){
flag_corner_ll++;
} 
if(index == W){
flag_corner_ul++;
flag_corner_ll++;
for(i = 2; i < my_height; i++){
for(j = 0; j < mult; j++){
(*im_after)[i][mult + j] = (int)((*im_before)[i][mult + j] * my_args.filter[1][1] +
(*im_before)[i - 1][mult + j] * my_args.filter[0][1] +
(*im_before)[i - 1][mult_multi_2 + j] * my_args.filter[0][2] +
(*im_before)[i][mult_multi_2 + j] * my_args.filter[1][2] +
(*im_before)[i + 1][mult_multi_2 + j] * my_args.filter[2][2] +
(*im_before)[i + 1][mult + j] * my_args.filter[2][1] +
(*im_before)[i + 1][j] * my_args.filter[2][0] +
(*im_before)[i][j] * my_args.filter[1][0] +
(*im_before)[i - 1][j] * my_args.filter[0][0]);
if((*im_after)[i][mult + j] < 0)
(*im_after)[i][mult + j] = 0;
else if((*im_after)[i][mult + j] > 255)
(*im_after)[i][mult + j] = 255;
}
} 
} 
if(index == NW){
flag_corner_ul++;
} 
if(flag_corner_ul == 3){
for(j = 0; j < mult; j++){
(*im_after)[1][mult + j] = (int)((*im_before)[1][mult + j] * my_args.filter[1][1] +
(*im_before)[0][mult + j] * my_args.filter[0][1] +
(*im_before)[0][mult_multi_2 + j] * my_args.filter[0][2] +
(*im_before)[1][mult_multi_2 + j] * my_args.filter[1][2] +
(*im_before)[2][mult_multi_2 + j] * my_args.filter[2][2] +
(*im_before)[2][mult + j] * my_args.filter[2][1] +
(*im_before)[2][j] * my_args.filter[2][0] +
(*im_before)[1][j] * my_args.filter[1][0] +
(*im_before)[0][j] * my_args.filter[0][0]);
if((*im_after)[1][mult + j] < 0)
(*im_after)[1][mult + j] = 0;
else if((*im_after)[1][mult + j] > 255)
(*im_after)[1][mult + j] = 255;
} 
} 
if(flag_corner_ur == 3){
for(j = 0; j < mult; j++){
(*im_after)[1][my_width + j] = (int)((*im_before)[1][my_width + j] * my_args.filter[1][1] +
(*im_before)[0][my_width + j] * my_args.filter[0][1] +
(*im_before)[0][my_width_incr_1 + j] * my_args.filter[0][2] +
(*im_before)[1][my_width_incr_1 + j] * my_args.filter[1][2] +
(*im_before)[2][my_width_incr_1 + j] * my_args.filter[2][2] +
(*im_before)[2][my_width + j] * my_args.filter[2][1] +
(*im_before)[2][my_width_decr_1 + j] * my_args.filter[2][0] +
(*im_before)[1][my_width_decr_1 + j] * my_args.filter[1][0] +
(*im_before)[0][my_width_decr_1 + j] * my_args.filter[0][0]);
if((*im_after)[1][my_width + j] < 0)
(*im_after)[1][my_width + j] = 0;
else if((*im_after)[1][my_width + j] > 255)
(*im_after)[1][my_width + j] = 255;
}
} 
if(flag_corner_lr == 3){
for(j = 0; j < mult; j++){
(*im_after)[my_height][my_width + j] = (int)((*im_before)[my_height][my_width + j] * my_args.filter[1][1] +
(*im_before)[my_height_decr_1][my_width + j] * my_args.filter[0][1] +
(*im_before)[my_height_decr_1][my_width_incr_1 + j] * my_args.filter[0][2] +
(*im_before)[my_height][my_width_incr_1 + j] * my_args.filter[1][2] +
(*im_before)[my_height_incr_1][my_width_incr_1 + j] * my_args.filter[2][2] +
(*im_before)[my_height_incr_1][my_width + j] * my_args.filter[2][1] +
(*im_before)[my_height_incr_1][my_width_decr_1 + j] * my_args.filter[2][0] +
(*im_before)[my_height][my_width_decr_1 + j] * my_args.filter[1][0] +
(*im_before)[my_height_decr_1][my_width_decr_1 + j] * my_args.filter[0][0]);
if((*im_after)[my_height][my_width + j] < 0)
(*im_after)[my_height][my_width + j] = 0;
else if((*im_after)[my_height][my_width + j] > 255)
(*im_after)[my_height][my_width + j] = 255;
}
} 
if(flag_corner_ll == 3){
for(j = 0; j < mult; j++){
(*im_after)[my_height][mult + j] = (int)((*im_before)[my_height][mult + j] * my_args.filter[1][1] +
(*im_before)[my_height_decr_1][mult + j] * my_args.filter[0][1] +
(*im_before)[my_height_decr_1][mult_multi_2 + j] * my_args.filter[0][2] +
(*im_before)[my_height][mult_multi_2 + j] * my_args.filter[1][2] +
(*im_before)[my_height_incr_1][mult_multi_2 + j] * my_args.filter[2][2] +
(*im_before)[my_height_incr_1][mult + j] * my_args.filter[2][1] +
(*im_before)[my_height_incr_1][j] * my_args.filter[2][0] +
(*im_before)[my_height][j] * my_args.filter[1][0] +
(*im_before)[my_height_decr_1][j] * my_args.filter[0][0]);
if((*im_after)[my_height][mult + j] < 0)
(*im_after)[my_height][mult + j] = 0;
else if((*im_after)[my_height][mult + j] > 255)
(*im_after)[my_height][mult + j] = 255;
}
} 
} 
MPI_Waitall(NUM_NEIGHBOURS, send_requests[r_index], MPI_STATUS_IGNORE);
#ifdef CHECK_CONVERGENCE
equality_flag = 0;
for(i = 1; (i < my_height_incr_1) && (equality_flag == 0); i++){
for(j = mult; j < my_width_incr_1; j++){
if((*im_after)[i][j] != (*im_before)[i][j]){
equality_flag = 1;
break;
} 
} 
} 
MPI_Allreduce(&equality_flag, &all_finished, 1, MPI_INT, MPI_LOR, my_cartesian_comm);
if(my_rank == 0 && print_message == 0 && all_finished == 0){
printf("Image convergence at %d iteration\n",iter);
print_message = 1;
}
#endif
tmp = im_before;
im_before = im_after;
im_after = tmp;
} 
double end = MPI_Wtime();
double time_elapsed = end - start;
double max_time, min_time;
if(comm_size != 1){
MPI_Reduce(&time_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, my_cartesian_comm);
MPI_Reduce(&time_elapsed, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, my_cartesian_comm);
}
else{
max_time = time_elapsed;
min_time = time_elapsed;
}
if(my_rank == 0)
printf("\n[Parallel Convolution Completed]:\nType of Image: %d\nResolution: %d x %d\nSeed Given: %d\nNumber of Iterations: %d\nNumber of Processes: %d\nRun time: %.5lf seconds\nFastest process completed in: %.5lf seconds\n\n",
my_args.image_type, my_args.image_width, my_args.image_height, my_args.image_seed, my_args.iterations, comm_size, max_time,min_time);
free(my_image_before[0]);
free(my_image_before);
free(my_image_after[0]);
free(my_image_after);
for(i = 0; i < NUM_NEIGHBOURS; i++){
MPI_Request_free(&send_after_requests[i]);
MPI_Request_free(&send_before_requests[i]);
MPI_Request_free(&recv_after_requests[i]);
MPI_Request_free(&recv_before_requests[i]);
} 
MPI_Type_free(&filter_type);
MPI_Type_free(&filter_type1);
MPI_Type_free(&args_type);
MPI_Type_free(&column_type);
MPI_Comm_free(&my_cartesian_comm);
MPI_Finalize();
return 0;
}
