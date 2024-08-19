#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "knn.h"
#include "matrix.h"
#include "def.h"
#define ROOT 0
#define INIT_MATRIX 1
#define KNN 2
#define PRINTING_DIST 3
#define PRINTING_IDX 4
int read_matrix(FILE *file, elem_t *X, size_t n, size_t d) {
for(size_t i = 0 ; i < n ; i++) {
for(size_t j = 0 ; j < d ; j++) {
int k = fscanf(file, "%f", &MATRIX_ELEM(X, i, j, n, d));
if(!k && feof(file)) {
return 1;
}
}
}
return 0;
}
int main(int argc, char **argv) {
MPI_Init(&argc, &argv);
int n_processes, rank;
MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int error = 0;
FILE *input_file = NULL;
FILE *output_file = NULL;
FILE *log_file = NULL;
size_t N, d, k = 0;
size_t max_mem = 1*GB;
switch(rank) {
case ROOT: ;
char *input_fname = NULL;
char *output_fname = NULL;
char *log_fname = NULL;
error = 0;
int c;
while((c = getopt(argc, argv, "i:o:l:k:m:")) != -1) {
switch(c) {
case 'i':
input_fname = optarg;
break;
case 'o':
output_fname = optarg;
break;
case 'l':
log_fname = optarg;
break;
case 'k':
k = atoi(optarg);
break;
case 'm': ;
size_t size;
char unit; 
int read_count = sscanf(optarg, "%zu%c", &size, &unit);
size_t unit_size = 1;
if(read_count == 0) {
error = EINVAL;
fprintf(stderr, "error parsing parameter '%c': %s\n", 'm', strerror(error));
break;
} else if(read_count == 2) {
switch(unit) {
case 'k':
case 'K':
unit_size = KB;
break;
case 'm':
case 'M':
unit_size = MB;
break;
case 'g':
case 'G':
unit_size = GB;
break;
default:
error = EINVAL;
fprintf(stderr, "error parsing parameter '%c': %s\n", 'm', strerror(error));
break;
}
}
max_mem = size * unit_size;
break;
case '?':
error = EINVAL;
fprintf(stderr, "error: parameter '%c' requires an input argument\n", optopt);
break;
}
}
if(error) break;
if(input_fname == NULL) {
if(optind < argc) {
input_fname = argv[optind++];
} else {
input_fname = "stdin";
input_file = stdin;
}
}
if(output_fname == NULL) {
output_fname = "stdout";
output_file = stdout;
}
if(log_fname == NULL) {
log_fname = "stdout";
log_file = stdout;
}
if((input_file == NULL) && (input_file = fopen(input_fname, "r")) == NULL) {
error = errno;
fprintf(stderr, "%s: %s\n", input_fname, strerror(error));
break;
}
if((output_file == NULL) && (output_file = fopen(output_fname, "w")) == NULL) {
error = errno;
fprintf(stderr, "%s: %s\n", output_fname, strerror(error));
break;
}
if((log_file == NULL) && (log_file = fopen(log_fname, "a")) == NULL) {
error = errno;
fprintf(stderr, "%s: %s\n", log_fname, strerror(error));
break;
}
if(k == 0) {
if(optind >= argc) {
error = EINVAL;
fprintf(stderr, "error: not enough input arguments\n");
break;
} else {
k = atoi(argv[optind++]);
}
}
int count = fscanf(input_file, "%zu %zu\n", &N, &d);
if(count < 2) {
error = errno;
fprintf(stderr, "error reading from %s: %s\n", input_fname, strerror(error));
break;
}
break;
}
MPI_Bcast(&error, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
if(error) {
MPI_Finalize();
exit(error);
}
MPI_Bcast(&N, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&d, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&k, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);
MPI_Bcast(&max_mem, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
size_t n_max = (size_t)ceil((double)N/(double)n_processes);
int r_send = mod(rank - 1, n_processes);
int r_recv = mod(rank + 1, n_processes);
int n_threads = omp_get_max_threads();
omp_set_num_threads(n_threads);
elem_t *X = create_matrix(n_max, d);
elem_t *Y = create_matrix(n_max, d);
elem_t *Z = create_matrix(n_max, d);
int i_begin = rank * N / n_processes;
int i_end = min((rank + 1) * N / n_processes, N);
int n = i_end - i_begin;
intmax_t t = (intmax_t)(max_mem/n_max - 3*d*sizeof(elem_t) - 2*k*(sizeof(elem_t) + sizeof(size_t))) 
/ (intmax_t)(sizeof(elem_t) + sizeof(size_t));
t = min(t, n);
MPI_Barrier(MPI_COMM_WORLD);
if(t < 1) {
if(rank == ROOT) fprintf(stderr, "not enough memory.\n");
MPI_Finalize();
exit(ENOMEM);
}
MPI_Barrier(MPI_COMM_WORLD);
int recv_size = 0;
MPI_Request send_req = MPI_REQUEST_NULL, recv_req;
MPI_Status send_status, recv_status;
switch(rank) {
case ROOT: ;
read_matrix(input_file, X, n, d);
for(int r = 1 ; r < n_processes ; r++) {
int r_begin = r * N / n_processes;
int r_end = min((r + 1) * N / n_processes, N);
int r_n = r_end - r_begin;
read_matrix(input_file, Y, r_n, d);
MPI_Wait(&send_req, &send_status);
MPI_Isend(Y, r_n * d, MPI_ELEM_T, r, INIT_MATRIX, MPI_COMM_WORLD, &send_req);
SWAP(Y, Z);
}
if(input_file != stdin) fclose(input_file);
break;
default: ;
MPI_Recv(X, n_max * d, MPI_ELEM_T, ROOT, INIT_MATRIX, MPI_COMM_WORLD, &recv_status);
break;
}
int m = n;
#pragma omp parallel for simd
for(size_t i = 0 ; i < m * d ; i++) {
Y[i] = X[i];
}
if(rank == ROOT) MPI_Wait(&send_req, &send_status);
MPI_Barrier(MPI_COMM_WORLD);
struct timespec t_begin, t_end;
clock_gettime(CLOCK_MONOTONIC, &t_begin);
knn_result *res = NULL;
for(int q = 0 ; q < n_processes ; q++) {
MPI_Isend(Y, m * d, MPI_ELEM_T, r_send, KNN, MPI_COMM_WORLD, &send_req);
MPI_Irecv(Z, n_max * d, MPI_ELEM_T, r_recv, KNN, MPI_COMM_WORLD, &recv_req);
int r = mod(rank + q, n_processes);
size_t Y_idx = r * N / n_processes;
res = knn(X, n, Y, Y_idx, m, d, k, t, &res);
MPI_Wait(&send_req, &send_status);
MPI_Wait(&recv_req, &recv_status);
MPI_Get_count(&recv_status, MPI_ELEM_T, &recv_size);
m = recv_size / d;
SWAP(Y, Z);
}
MPI_Barrier(MPI_COMM_WORLD);
clock_gettime(CLOCK_MONOTONIC, &t_end);
delete_matrix(X);
delete_matrix(Y);
delete_matrix(Z);
switch(rank) {
case ROOT: ;
double time_elapsed = (t_end.tv_sec - t_begin.tv_sec) + (t_end.tv_nsec - t_begin.tv_nsec) / 1e9f;
fprintf(log_file, "%zu %zu %zu %zu %d %d %zu %zd %lf\n", N, d, k, max_mem, n_processes, n_threads, n_max, t, time_elapsed);
elem_t *dists= (elem_t *) malloc(n_max * k * sizeof(elem_t));
elem_t *dists_swp = (elem_t *) malloc(n_max * k * sizeof(elem_t));
size_t *idxs= (size_t *) malloc(n_max * k * sizeof(size_t));
size_t *idxs_swp = (size_t *) malloc(n_max * k * sizeof(size_t));
#pragma omp parallel for simd
for(int i = 0 ; i < n * k ; i++) {
dists[i] = res->n_dist[i];
idxs[i] = res->n_idx[i];
}
MPI_Request recv_dist, recv_idx;
MPI_Status dist_status, idx_status;
for(int r = 1 ; r < n_processes ; r++) {
MPI_Irecv(dists_swp, n_max * k, MPI_ELEM_T, r, PRINTING_DIST, MPI_COMM_WORLD, &recv_dist);
MPI_Irecv(idxs_swp, n_max * k, MPI_SIZE_T, r, PRINTING_IDX, MPI_COMM_WORLD, &recv_idx);
for(size_t i = 0 ; i < n ; i++) {
for(size_t j = 0 ; j < k ; j++) {
fprintf(output_file, "%zu:%0.2f ", 
MATRIX_ELEM(idxs, i, j, n, k), MATRIX_ELEM(dists, i, j, n, k));
}
fprintf(output_file, "\n");
}
MPI_Wait(&recv_dist, &dist_status);
MPI_Wait(&recv_idx, &idx_status);
MPI_Get_count(&dist_status, MPI_ELEM_T, &recv_size);
n = recv_size / k;
SWAP(dists, dists_swp);
SWAP(idxs, idxs_swp);
} for(size_t i = 0 ; i < n ; i++) {
for(size_t j = 0 ; j < k ; j++) {
fprintf(output_file, "%zu:%0.2f ", 
MATRIX_ELEM(idxs, i, j, n, k), MATRIX_ELEM(dists, i, j, n, k));
}
fprintf(output_file, "\n");
}
if(output_file != stdout) fclose(output_file);
if(log_file != stdout) fclose(log_file);
free(dists_swp);
free(idxs_swp);
break;
default: ;
MPI_Isend(res->n_dist, n * k, MPI_ELEM_T, ROOT, PRINTING_DIST, MPI_COMM_WORLD, &send_req);
MPI_Send(res->n_idx, n * k, MPI_SIZE_T, ROOT, PRINTING_IDX, MPI_COMM_WORLD);
MPI_Wait(&send_req, &send_status);
break;
}
MPI_Barrier(MPI_COMM_WORLD);
delete_knn(res);
MPI_Finalize();
}
