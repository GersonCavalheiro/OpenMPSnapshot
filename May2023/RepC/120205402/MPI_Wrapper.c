#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MPI_Dot_Writer.h"
#include "MPI_Task_Graph.h"
#define MPI_WRAPPER_TAG 99999
#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else 
#define _EXTERN_C_
#endif 
#endif 
#ifdef MPICH_HAS_C2F
_EXTERN_C_ void *MPIR_ToPointer(int);
#endif 
#ifdef PIC
#pragma weak pmpi_init
#pragma weak PMPI_INIT
#pragma weak pmpi_init_
#pragma weak pmpi_init__
#endif 
_EXTERN_C_ void pmpi_init(MPI_Fint *ierr);
_EXTERN_C_ void PMPI_INIT(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init__(MPI_Fint *ierr);
typedef struct {
MPI_Request *request;
int mpi_op;
int tag;
int num_of_bytes;
int send_to;
int receive_from;
struct IrecvRequest* next;
} IrecvRequest;
IrecvRequest *request_stack_head;
int num_nodes;
int my_rank;
int start_time, end_time;
void pre_MPI(int mpi_op, int tag, int num_of_bytes, int source, int dest) {
if(mpi_op != MPI_op_Init) {
end_time = (int) MPI_Wtime();
Vertex* vertex = create_vertex(my_rank, mpi_op, start_time, end_time, num_of_bytes, tag, source, dest);
add_vertex_to_list(vertex);
}
if(mpi_op == MPI_op_Finalize) {
MPI_Datatype VertexParcelableType;
MPI_Datatype type[1] = {MPI_INT};
int blocklen[1] = {8};
MPI_Aint disp[1] = {0};
MPI_Type_struct(1, blocklen, disp, type, &VertexParcelableType);
MPI_Type_commit(&VertexParcelableType);
VertexParcelable vertices[get_vertex_count()];
int k=0;
Vertex *v = get_vertex_list_head();
while(v != NULL) {
vertices[k] = create_vertex_parcelable(v);
k++;
v = (Vertex*)v->next;
}
if(my_rank != 0) {
PMPI_Send(vertices, get_vertex_count(), VertexParcelableType, 0, MPI_WRAPPER_TAG, MPI_COMM_WORLD);
} else {
set_num_ranks_to_graph(num_nodes);
add_vertices_to_graph(vertices, get_vertex_count(), my_rank);
int i;
for(i=1;i<num_nodes;i++) {
VertexParcelable *others_vertices;
MPI_Status status;
int  n_elem;
MPI_Probe(i, MPI_WRAPPER_TAG, MPI_COMM_WORLD, &status);
MPI_Get_count(&status, VertexParcelableType, &n_elem); 
if (n_elem != MPI_UNDEFINED) {
others_vertices = (VertexParcelable *)malloc(n_elem*sizeof(VertexParcelable));
PMPI_Recv(others_vertices, n_elem, VertexParcelableType, i, MPI_WRAPPER_TAG, 
MPI_COMM_WORLD, &status);
add_vertices_to_graph(others_vertices, n_elem, i);
}
}
construct_graph();
find_critical_path();
dump_crit_path();
dump_graph_to_dot();
dump_stats();
pop_up_graph();
}
MPI_Type_free(&VertexParcelableType);
free_vertices();
}
}
void post_MPI(int mpi_op, int tag, int num_of_bytes, int source, int dest) {
if(mpi_op == MPI_op_Init) {
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
start_time = end_time = 0;
Vertex* vertex = create_vertex(my_rank, mpi_op, start_time, end_time, num_of_bytes, tag, source, dest);
add_vertex_to_list(vertex);
}
start_time = (int)MPI_Wtime();
}
_EXTERN_C_ int PMPI_Init(int *argc, char ***argv);
_EXTERN_C_ int MPI_Init(int *argc, char ***argv) { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Init, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Init(argc, argv);
post_MPI(MPI_op_Init, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Finalize();
_EXTERN_C_ int MPI_Finalize() { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Finalize, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Finalize();
post_MPI(MPI_op_Finalize, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Barrier(MPI_Comm comm);
_EXTERN_C_ int MPI_Barrier(MPI_Comm comm) { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Barrier, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Barrier(comm);
post_MPI(MPI_op_Barrier, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
_EXTERN_C_ int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Alltoall, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
post_MPI(MPI_op_Alltoall, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Scatter, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
post_MPI(MPI_op_Scatter, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Gather, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
post_MPI(MPI_op_Gather, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Reduce, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
post_MPI(MPI_op_Reduce, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
_EXTERN_C_ int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { 
int _wrap_py_return_val = 0;
pre_MPI(MPI_op_Allreduce, -1, -1, -1, -1);
_wrap_py_return_val = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
post_MPI(MPI_op_Allreduce, -1, -1, -1, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
_EXTERN_C_ int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { 
int _wrap_py_return_val = 0;
int datatype_size;
MPI_Type_size(datatype, &datatype_size);
int num_bytes = count * datatype_size;
pre_MPI(MPI_op_Send, tag, num_bytes, -1, dest);
_wrap_py_return_val = PMPI_Send(buf, count, datatype, dest, tag, comm);
post_MPI(MPI_op_Send, tag, num_bytes, -1, dest);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
_EXTERN_C_ int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
int _wrap_py_return_val = 0;
int datatype_size;
MPI_Type_size(datatype, &datatype_size);
int num_bytes = count * datatype_size;
pre_MPI(MPI_op_Isend, tag, num_bytes, -1, dest);
_wrap_py_return_val = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
post_MPI(MPI_op_Isend, tag, num_bytes, -1, dest);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
_EXTERN_C_ int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) { 
int _wrap_py_return_val = 0;
int datatype_size;
MPI_Type_size(datatype, &datatype_size);
int num_bytes = count * datatype_size;
pre_MPI(MPI_op_Recv, tag, num_bytes, source, -1);
_wrap_py_return_val = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
post_MPI(MPI_op_Recv, tag, num_bytes, source, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
_EXTERN_C_ int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) { 
int _wrap_py_return_val = 0;
int datatype_size;
MPI_Type_size(datatype, &datatype_size);
int num_bytes = count * datatype_size;
pre_MPI(MPI_op_Irecv, tag, num_bytes, source, -1);
IrecvRequest *req = (IrecvRequest *)malloc(1*sizeof(IrecvRequest));
req->request = request;
req->mpi_op = MPI_op_Irecv;
req->tag = tag;
req->num_of_bytes = num_bytes;
req->send_to = -1;
req->receive_from = source;
req->next = NULL;
if(request_stack_head == NULL) {
request_stack_head = req;
} else {
IrecvRequest *tail = request_stack_head;
while(tail->next != NULL) {
tail = (IrecvRequest *)tail->next;
}
tail->next = (struct IrecvRequest *)req;
}
_wrap_py_return_val = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
post_MPI(MPI_op_Irecv, tag, num_bytes, source, -1);
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Wait(MPI_Request *request, MPI_Status *status);
_EXTERN_C_ int MPI_Wait(MPI_Request *request, MPI_Status *status) { 
int _wrap_py_return_val = 0;
int found = 0;
IrecvRequest *prev = NULL;
IrecvRequest *curr = request_stack_head;
if(request_stack_head != NULL) {
do {
if(curr->request == request) {
found = 1;
break;
}
prev = curr;
curr = (IrecvRequest *)curr->next;
} while(curr->next != NULL);
if(found) {
if(prev != NULL) {
prev->next = curr->next;
} else {
request_stack_head = (IrecvRequest *)curr->next;
}
}
}
if(found) {
pre_MPI(MPI_op_Wait, curr->tag, curr->num_of_bytes, curr->receive_from, curr->send_to);
} else {
pre_MPI(MPI_op_Wait, -1, -1, -1, -1);
}
_wrap_py_return_val = PMPI_Wait(request, status);
if(found) {
post_MPI(MPI_op_Wait, curr->tag, curr->num_of_bytes, curr->receive_from, curr->send_to);
} else {
post_MPI(MPI_op_Wait, -1, -1, -1, -1);
}
return _wrap_py_return_val;
}
_EXTERN_C_ int PMPI_Waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses);
_EXTERN_C_ int MPI_Waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses) { 
int _wrap_py_return_val = 0;
int i;
for(i=0;i<count;i++) {
MPI_Wait(&array_of_requests[i], &array_of_statuses[i]);
}
return _wrap_py_return_val;
}
