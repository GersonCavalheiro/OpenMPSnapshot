#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ARRAY_SIZE 9600 
#define N_ARRAYS  960 
#define WORKER_ARRAYS 32 
#define MASTER    0    
#define DEFAULT_THREADS 4
#define POISON_PILL -2
#define FIRST_TASK -1
#define ARRAY_MSG 2 
#define DEBUG 0
int cmpfunc (const void * a, const void * b){
return ( *(int*)a - *(int*)b );
}
void bs(int n, int * vetor){
int c=0, d, troca, trocou =1;
while (c < (n-1) & trocou ) {
trocou = 0;
for (d = 0 ; d < n - c - 1; d++)
if (vetor[d] > vetor[d+1]) {
troca = vetor[d];
vetor[d] = vetor[d+1];
vetor[d+1] = troca;
trocou = 1;
}
c++;
}
}
int main(int argc,char **argv){
int threads, n_tasks, task_id, exit_code, i, j, k, index, worker, sent_arrays = 0, received_arrays = 0, sending = 1;
int msg_size = (ARRAY_SIZE+1); 
int index_pos = msg_size -1; 
double t1, t2, t1_send, t2_send, t1_recv, t2_recv;  
MPI_Status mpi_status;
if (argc == 1){
threads = DEFAULT_THREADS; 
}else if (argc == 2){
threads = atoi(argv[1]);
} else {
printf("Usage: %s number_of_threads.\n", argv[0]);
return 0;
}
printf("Run with %d threads...\n", threads);
int buffer_size = msg_size * WORKER_ARRAYS;
exit_code = MPI_Init(&argc,&argv);
exit_code|= MPI_Comm_size(MPI_COMM_WORLD,&n_tasks);
exit_code|= MPI_Comm_rank(MPI_COMM_WORLD,&task_id);
if (exit_code != MPI_SUCCESS) {
printf ("Error initializing MPI and obtaining task ID information\n");
return 1;
}
if (task_id == MASTER){
t1 = MPI_Wtime();
int (*bag_of_tasks)[msg_size] = malloc (N_ARRAYS * sizeof *bag_of_tasks);
int (*buffer) = malloc (buffer_size * sizeof(int));
for (i=0; i < N_ARRAYS; i++){
bag_of_tasks[i][index_pos] = i;
}
for (i = 0; i < N_ARRAYS; i++){
for(j = 0; j < (msg_size-1); j++){ 
bag_of_tasks [i][j] = (ARRAY_SIZE-j)*(i+1); 
}
}
for (worker = 1; worker < n_tasks; worker++){
if (sent_arrays < N_ARRAYS){
k = 0;
for(i=0; i < WORKER_ARRAYS; i++){
for(j=0; j < msg_size; j++){
buffer[k++] = bag_of_tasks[sent_arrays][j];
}
sent_arrays++;
}
MPI_Send(buffer, buffer_size, MPI_INT, worker, ARRAY_MSG, MPI_COMM_WORLD);
}
}
while(sending != 0) {
t1_recv = MPI_Wtime();
MPI_Recv(buffer, buffer_size, MPI_INT, MPI_ANY_SOURCE, ARRAY_MSG, MPI_COMM_WORLD, &mpi_status);
t2_recv = MPI_Wtime();
printf("[Master] Recv time: [%f]\n", (t2_recv - t1_recv));
k = 0;
for (i=0; i < WORKER_ARRAYS; i++){               
int receive_buffer[msg_size];
for (j = 0; j < msg_size; j++){
receive_buffer[j] = buffer[k++];
}
int index = receive_buffer[index_pos];
memcpy(bag_of_tasks[index], receive_buffer, msg_size * sizeof(int));
received_arrays++;  
}
if (received_arrays == N_ARRAYS){
sending = 0;
}else{
if (sent_arrays < N_ARRAYS){
t1_send = MPI_Wtime();
k = 0;
for(i=0; i < WORKER_ARRAYS; i++){
for(j=0; j < msg_size; j++){
buffer[k++] = bag_of_tasks[sent_arrays][j];
}
sent_arrays++;
}
MPI_Send(buffer, buffer_size, MPI_INT, mpi_status.MPI_SOURCE, ARRAY_MSG, MPI_COMM_WORLD);
t2_send = MPI_Wtime();
printf("[Master] Send and build buffer time: [%f]\n", (t2_send - t1_send));
}
}      
}
buffer[index_pos] = POISON_PILL; 
for (worker = 1; worker < n_tasks; worker++){
MPI_Send(buffer, buffer_size, MPI_INT, worker, ARRAY_MSG, MPI_COMM_WORLD);
}
t2 = MPI_Wtime(); 
printf("[Master] Duration [%f]\n", t2-t1);
#if DEBUG == 1
printf("Print enabled...\n");
for (i = 0; i < N_ARRAYS; i++){
printf("Vector %d [", i);
for(j=0; j<ARRAY_SIZE; j++){
printf("%d ", bag_of_tasks[i][j]);
}
printf("]\n");
}
#endif
free(buffer);
free(bag_of_tasks);
} else {
int alive = 1;
int (*worker_buffer) = malloc (buffer_size * sizeof(int));
do {
MPI_Recv(worker_buffer, buffer_size, MPI_INT, MASTER, ARRAY_MSG, MPI_COMM_WORLD, &mpi_status);
index = worker_buffer[index_pos];
if (index == POISON_PILL) {
alive = 0;
} else {
omp_set_num_threads(threads);
#pragma omp parallel private (i)
#pragma omp for schedule (dynamic)
for (i = 0; i < buffer_size; i += msg_size){
bs(msg_size-1, &worker_buffer[i]);
}
MPI_Send(worker_buffer, buffer_size, MPI_INT, MASTER, ARRAY_MSG, MPI_COMM_WORLD);
}
} while (alive != 0);
free(worker_buffer);
}
MPI_Finalize();
}
