#include "worker.h"
#include "global_variables.h"
#include "debug_utils.h"
#include "post_block.h"
#include "process_events.h"
#include "event_generator.h"
#include "event_list.h"
#include "quicksort.h"
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#define MAX_EVENT_KEEPER_SIZE 2048
extern const int MPI_MASTER;
extern const int POST_NUMBER_TAG;
extern const int POST_EXCHANGE_TAG;
extern const int VALUED_EVENT_NUMBER_TAG;
extern const int VALUED_EVENT_TRANSMISSION_TAG;
extern const int TOP_NUMBER;
post_block * receive_post(int worker_id);
post_block * receive_post(int worker_id)
{
MPI_Status ret;
int * post_ts = malloc(sizeof(int));
long * post_id = malloc(sizeof(long));
long * user_id = malloc(sizeof(long));
int * comment_ar_size = malloc(sizeof(int));
int * comment_ts=NULL;
long * comment_user_id=NULL;
MPI_Recv(post_ts,1,MPI_INT,MPI_MASTER,POST_EXCHANGE_TAG*worker_id,MPI_COMM_WORLD, &ret);
MPI_Recv(post_id,1,MPI_LONG,MPI_MASTER,POST_EXCHANGE_TAG*worker_id,MPI_COMM_WORLD, &ret);
MPI_Recv(user_id,1,MPI_LONG,MPI_MASTER,POST_EXCHANGE_TAG*worker_id,MPI_COMM_WORLD, &ret);
MPI_Recv(comment_ar_size,1,MPI_INT,MPI_MASTER,POST_EXCHANGE_TAG*worker_id,MPI_COMM_WORLD,&ret);
if( *comment_ar_size >0)
{
comment_ts = calloc(sizeof(int), *comment_ar_size);
comment_user_id = calloc(sizeof(long), *comment_ar_size);
MPI_Recv(comment_ts,*comment_ar_size,MPI_INT,MPI_MASTER, POST_EXCHANGE_TAG*worker_id,MPI_COMM_WORLD, &ret);
MPI_Recv(comment_user_id,*comment_ar_size,MPI_LONG, MPI_MASTER, POST_EXCHANGE_TAG*worker_id,MPI_COMM_WORLD, &ret);
}
post_block * ret_pb = new_post_block(*post_ts,*post_id,*user_id,*comment_ar_size,comment_ts,comment_user_id);
free(post_ts);
free(post_id);
free(user_id);
free(comment_ar_size);
return ret_pb;
}
int worker_execution(int argc, char * argv[], int worker_id, MPI_Datatype mpi_valued_event)
{
MPI_Status ret;
valued_event *** main_keeper = malloc(sizeof(valued_event **)*MAX_EVENT_KEEPER_SIZE );
int * main_keeper_dim = malloc(sizeof(int)*MAX_EVENT_KEEPER_SIZE );
int main_keeper_size=0;
print_info("Worker %d is waiting for n_posts...", worker_id);
int * n_posts = malloc(sizeof(int));
MPI_Recv(n_posts,1,MPI_INT, MPI_MASTER, POST_NUMBER_TAG*worker_id,MPI_COMM_WORLD, &ret);
while(*n_posts>=0)
{
print_info("Worker %d received n_post: %d", worker_id, *n_posts);
post_block ** pb_ar = malloc(sizeof(post_block * )*(*n_posts) );
for(int i=0; i<*n_posts; i++)
{
pb_ar[i] = receive_post(worker_id);
}
#pragma omp parallel
#pragma omp single nowait
{
#pragma omp task shared(worker_id, main_keeper, main_keeper_dim, main_keeper_size)
{
int v_event_size;
valued_event** v_event_array =  process_events(pb_ar, *n_posts, &v_event_size);
#pragma omp critical(MAIN_KEEPER_UPDATE)
{
main_keeper[main_keeper_size] = v_event_array;
main_keeper_dim[main_keeper_size] = v_event_size;
print_fine("Worker %d processed produced a top_three sequence. put it at position %d in main_keeper", worker_id, main_keeper_size);
main_keeper_size++;
}
for(int i=0; i<*n_posts; i++)
{
del_post_block(pb_ar[i]);
}
free(pb_ar);
}
}
MPI_Recv(n_posts,1,MPI_INT, MPI_MASTER, POST_NUMBER_TAG*worker_id,MPI_COMM_WORLD, &ret);
}
#pragma omp barrier
MPI_Barrier(MPI_COMM_WORLD);
free(n_posts);
print_info("Worker %d received the stop signal for post trasmission. main_keeper_size: %d", worker_id, main_keeper_size);
valued_event * out_ar;
int out_size;
if(main_keeper_size>0)
{
out_ar = merge_valued_event_array_with_ref(main_keeper, main_keeper_dim, main_keeper_size, &out_size);
print_fine("worker %d produced a valued event array %d big.",worker_id, out_size);
if(out_ar==NULL)
{
print_error("worker %d cannot malloc out_ar", worker_id);
}
}
else
{
out_ar=NULL;
out_size=0;
}
if(main_keeper_size>0)
{
for(int i=0; i<main_keeper_size; i++)
{
for(int j=0; j<main_keeper_dim[i]; j++)
{
clear_valued_event(main_keeper[i][j]);
}
free(main_keeper[i]);
}
}
free(main_keeper);
free(main_keeper_dim);
if(out_ar>0)
{
print_fine("Worker %d ts bounds: [%d,%d]",worker_id,out_ar[0].valued_event_ts,out_ar[out_size-1].valued_event_ts);
}
int counter=0;
int master_ts;
while(counter<out_size)
{
MPI_Status ret;
int used_elements=0;
MPI_Recv(&master_ts,1,MPI_INT,MPI_MASTER,VALUED_EVENT_TS_TAG*worker_id,MPI_COMM_WORLD,&ret);
int send_buf_count=0;
while(counter<out_size && out_ar[counter].valued_event_ts<master_ts)
{
counter++;
}
for(int i=counter; i<out_size && out_ar[i].valued_event_ts==master_ts; i++)
{
send_buf_count++;
}
valued_event * send_buf = malloc(sizeof(valued_event)*send_buf_count);
memcpy(send_buf,out_ar+counter,sizeof(valued_event)*send_buf_count);
int send_buf_n_elements;
if(send_buf_count>TOP_NUMBER)
{
send_buf_n_elements=TOP_NUMBER;
}
else
{
send_buf_n_elements=send_buf_count;
}
MPI_Send(&send_buf_n_elements,1,MPI_INT,MPI_MASTER,VALUED_EVENT_NUMBER_TAG*worker_id,MPI_COMM_WORLD);
if(send_buf_count>0)
{
sort_valued_events_on_score_with_array(send_buf, 0, send_buf_count-1);
MPI_Send(send_buf,send_buf_n_elements,mpi_valued_event,MPI_MASTER,VALUED_EVENT_TRANSMISSION_TAG*worker_id, MPI_COMM_WORLD);
}
free(send_buf);
counter=counter+send_buf_count;
}
int stop_worker_request_code=-1;
MPI_Recv(&master_ts,1,MPI_INT,MPI_MASTER,VALUED_EVENT_TS_TAG*worker_id,MPI_COMM_WORLD,&ret);
MPI_Send(&stop_worker_request_code,1,MPI_INT, MPI_MASTER,VALUED_EVENT_NUMBER_TAG*worker_id,MPI_COMM_WORLD);
if(out_size>0)
{
free(out_ar);
}
print_info("Worker %d terminated successfully (:", worker_id);
return 0;
}