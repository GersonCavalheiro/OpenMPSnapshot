

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <float.h>

#define BC_HOT  1.0
#define BC_COLD 0.0
#define INITIAL_GRID 0.5
#define MAX_ITERATIONS 10000
#define TOL 1.0e-4

struct timeval tv;
double get_clock() {
struct timeval tv; int ok;
ok = gettimeofday(&tv, (void *) 0);
if (ok<0) { printf("gettimeofday error");  }
return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

double **create_matrix(int subprob_size) {
int i;
double **a;
double *rows;

a = (double**) malloc(sizeof(double*)*subprob_size);
rows = (double*) malloc(sizeof(double)*subprob_size*subprob_size);

#pragma omp parallel for
for (i=0;i<subprob_size;i++) {
a[i] = &rows[i*subprob_size];
}

return a;
}

void init_matrix(double **a, double *rfrbuff, double *rfcbuff, double *rlrbuff, double *rlcbuff, int n_subprobs, int subprob_size, int column_num, int row_num) {
int i, j;

#pragma omp parallel for collapse(2)
for(i=0; i<subprob_size; i++) {
for(j=0; j<subprob_size; j++)
a[i][j] = INITIAL_GRID;
}

if (column_num == 0){
if (row_num == 0){
#pragma omp parallel for
for(i=0; i<subprob_size; i++){
rlcbuff[i] = INITIAL_GRID;
rfcbuff[i] = BC_HOT;
rlrbuff[i] = INITIAL_GRID;
rfrbuff[i] = BC_HOT;
}
}
else if(row_num == ((int) sqrt(n_subprobs))-1){
#pragma omp parallel for
for(i=0; i<subprob_size; i++){
rlcbuff[i] = INITIAL_GRID;
rfcbuff[i] = BC_HOT;
rlrbuff[i] = BC_COLD;
rfrbuff[i] = INITIAL_GRID;
}
}

}
else if(column_num == ((int) sqrt(n_subprobs))-1){
if (row_num == 0){
#pragma omp parallel for
for(i=0; i<subprob_size; i++){
rlcbuff[i] = BC_HOT;
rfcbuff[i] = INITIAL_GRID;
rlrbuff[i] = INITIAL_GRID;
rfrbuff[i] = BC_HOT;
}
}
else if(row_num == ((int) sqrt(n_subprobs))-1){
#pragma omp parallel for
for(i=0; i<subprob_size; i++){
rlcbuff[i] = BC_HOT;
rfcbuff[i] = INITIAL_GRID;
rlrbuff[i] = BC_COLD;
rfrbuff[i] = INITIAL_GRID;
}
}
}
}

void swap_matrix(double ***a, double ***b) {
double **temp;

temp = *a;
*a = *b;
*b = temp;
}

void print_grid(double **a, int nstart, int nend) {
int i, j;

for(i=nstart; i<nend; i++) {
for(j=nstart; j<nend; j++) {
printf("%6.4lf ", a[i][j]);
}
printf("\n");
}
}

void free_matrix(double **a) {
free(a[0]);
free(a);
}

void wait_req(MPI_Request *req){
int req_flag = 0;
while(!req_flag) MPI_Test(req, &req_flag, MPI_STATUS_IGNORE);
}

int main(int argc, char* argv[]) {
int i, j , i_aux = 0, j_aux = 0, generic_tag = 0, iteration;
int n_dim, n_subprobs, subprob_size;
int column_num, row_num;
double **a, **b, maxdiff, maxdiff_aux;
MPI_Datatype double_strided_vect;
double **res;
int root_rank = 0, res_offset;

double tstart, tend, computetotal, gathertotal;

int my_rank;
double *sfrbuff, *sfcbuff, *slrbuff, *slcbuff;
double *rfrbuff, *rfcbuff, *rlrbuff, *rlcbuff;
MPI_Request *sfrreq, *sfcreq, *slrreq, *slcreq;
MPI_Request *rfrreq, *rfcreq, *rlrreq, *rlcreq;

if (argc != 4) return -1;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

omp_set_num_threads(atoi(argv[1]));

n_subprobs = atoi(argv[2]);
n_dim = atoi(argv[3]);

subprob_size = (int) sqrt((n_dim*n_dim)/n_subprobs);

column_num = my_rank%((int) sqrt(n_subprobs));
row_num = (int) (my_rank/(int) sqrt(n_subprobs));

sfrbuff = malloc(subprob_size*sizeof(double)); sfcbuff = malloc(subprob_size*sizeof(double)); slrbuff = malloc(subprob_size*sizeof(double)); slcbuff = malloc(subprob_size*sizeof(double));
rfrbuff = malloc(subprob_size*sizeof(double)); rfcbuff = malloc(subprob_size*sizeof(double)); rlrbuff = malloc(subprob_size*sizeof(double)); rlcbuff = malloc(subprob_size*sizeof(double));

sfrreq = malloc(sizeof(MPI_Request)); sfcreq = malloc(sizeof(MPI_Request)); slrreq = malloc(sizeof(MPI_Request)); slcreq = malloc(sizeof(MPI_Request));
rfrreq = malloc(sizeof(MPI_Request)); rfcreq = malloc(sizeof(MPI_Request)); rlrreq = malloc(sizeof(MPI_Request)); rlcreq = malloc(sizeof(MPI_Request));

a = create_matrix(subprob_size);
b = create_matrix(subprob_size);
if (my_rank == root_rank) res = create_matrix(n_dim);

MPI_Type_vector(subprob_size, subprob_size, n_dim, MPI_DOUBLE, &double_strided_vect);
MPI_Type_commit(&double_strided_vect);

iteration=0;
printf("[%d] Running simulation with tolerance=%lf and max iterations=%d\n",
my_rank, TOL, MAX_ITERATIONS);
tstart = MPI_Wtime();

init_matrix(a, rfrbuff, rfcbuff, rlrbuff, rlcbuff, n_subprobs, subprob_size, column_num, row_num);

maxdiff = DBL_MAX;
while(maxdiff > TOL && iteration<MAX_ITERATIONS) {
maxdiff = 0.0;

if (column_num != ((int) sqrt(n_subprobs))-1){
#pragma omp parallel for
for(i=0; i<subprob_size; i++) slcbuff[i] = a[i][subprob_size-1];
MPI_Isend(slcbuff, subprob_size, MPI_DOUBLE, my_rank+1, iteration, MPI_COMM_WORLD, slcreq);

MPI_Irecv(rlcbuff, subprob_size, MPI_DOUBLE, my_rank+1, iteration, MPI_COMM_WORLD, rlcreq);
}
if (column_num != 0){
#pragma omp parallel for
for(i=0; i<subprob_size; i++) sfcbuff[i] = a[i][0];
MPI_Isend(sfcbuff, subprob_size, MPI_DOUBLE, my_rank-1, iteration, MPI_COMM_WORLD, sfcreq);

MPI_Irecv(rfcbuff, subprob_size, MPI_DOUBLE, my_rank-1, iteration, MPI_COMM_WORLD, rfcreq);
}

if (row_num != ((int) sqrt(n_subprobs))-1){
memcpy(slrbuff, a[subprob_size-1], subprob_size*sizeof(double));
MPI_Isend(slrbuff, subprob_size, MPI_DOUBLE, (int) (my_rank+sqrt(n_subprobs)), iteration, MPI_COMM_WORLD, slrreq);

MPI_Irecv(rlrbuff, subprob_size, MPI_DOUBLE, (int) (my_rank+sqrt(n_subprobs)), iteration, MPI_COMM_WORLD, rlrreq);
}
if (row_num != 0){
memcpy(sfrbuff, a[0], subprob_size*sizeof(double));
MPI_Isend(sfrbuff, subprob_size, MPI_DOUBLE, (int) (my_rank-sqrt(n_subprobs)), iteration, MPI_COMM_WORLD, sfrreq);

MPI_Irecv(rfrbuff, subprob_size, MPI_DOUBLE, (int) (my_rank-sqrt(n_subprobs)), iteration, MPI_COMM_WORLD, rfrreq);
}


#pragma omp parallel for reduction(max:maxdiff) collapse(2)
for(i=1;i<subprob_size-1;i++) {
for(j=1;j<subprob_size-1;j++) {
b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);
}
}


if (column_num != ((int) sqrt(n_subprobs))-1){
wait_req(rlcreq);
}
if (column_num != 0){
wait_req(rfcreq);
}

if (row_num != ((int) sqrt(n_subprobs))-1){
wait_req(rlrreq);
}
if (row_num != 0){
wait_req(rfrreq);
}


i=0;
j=0;
b[i][j] = 0.2*(a[i][j]+rfrbuff[j]+a[i+1][j]+rfcbuff[i]+a[i][j+1]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);
#pragma omp parallel for reduction(max:maxdiff)
for(j=1;j<subprob_size-1;j++){
b[i][j] = 0.2*(a[i][j]+rfrbuff[j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);
}
j=subprob_size-1;
b[i][j] = 0.2*(a[i][j]+rfrbuff[j]+a[i+1][j]+a[i][j-1]+rlcbuff[i]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);

#pragma omp parallel for reduction(max:maxdiff)
for(i=1;i<subprob_size-1;i++) {
j=0;
b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+rfcbuff[i]+a[i][j+1]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);
}

#pragma omp parallel for reduction(max:maxdiff)
for(i=1;i<subprob_size-1;i++){
j=subprob_size-1;
b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+rlcbuff[i]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);
}

i=subprob_size-1;
j=0;
b[i][j] = 0.2*(a[i][j]+a[i-1][j]+rlrbuff[j]+rfcbuff[i]+a[i][j+1]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);
#pragma omp parallel for reduction(max:maxdiff)
for(j=1;j<subprob_size-1;j++){
b[i][j] = 0.2*(a[i][j]+a[i-1][j]+rlrbuff[j]+a[i][j-1]+a[i][j+1]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);
}
j=subprob_size-1;
b[i][j] = 0.2*(a[i][j]+a[i-1][j]+rlrbuff[j]+a[i][j-1]+rlcbuff[i]);
if (fabs(b[i][j]-a[i][j]) > maxdiff) maxdiff = fabs(b[i][j]-a[i][j]);


MPI_Allreduce(&maxdiff, &maxdiff_aux, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
maxdiff = maxdiff_aux;

swap_matrix(&a,&b);


iteration+=1;
}

tend = MPI_Wtime();
computetotal = tend-tstart;
tstart = MPI_Wtime();


if (my_rank == root_rank){
MPI_Isend(a[0], subprob_size*subprob_size, MPI_DOUBLE, root_rank, generic_tag, MPI_COMM_WORLD, sfrreq);
MPI_Recv(res[0], 1, double_strided_vect, root_rank, generic_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

for(i=1;i<n_subprobs;i++){
res_offset = ((int) (i/(int) sqrt(n_subprobs))) *subprob_size*subprob_size*(int) sqrt(n_subprobs);
res_offset += i%((int) sqrt(n_subprobs)) * subprob_size;
MPI_Recv(res[0] + res_offset, 1, double_strided_vect, i, generic_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
}
else{
MPI_Send(a[0], subprob_size*subprob_size, MPI_DOUBLE, root_rank, generic_tag, MPI_COMM_WORLD);
}

tend = MPI_Wtime();
gathertotal = tend-tstart;

if (my_rank == root_rank){

printf("Results:\n");
printf("Iterations=%d\n", iteration);
printf("Tolerance=%12.10lf\n", maxdiff);
printf("Problem dimmensions=%dx%d\n", n_dim, n_dim);
printf("Number of subproblems=%d\n", n_subprobs);
printf("Compute time=%12.10lf\n", computetotal);
printf("Gather time=%12.10lf\n", gathertotal);
printf("Total time=%12.10lf\n", gathertotal+computetotal);

free_matrix(res);
}
free_matrix(a);
free_matrix(b);

free(sfrbuff); free(sfcbuff); free(slrbuff); free(slcbuff);
free(rfrbuff); free(rfcbuff); free(rlrbuff); free(rlcbuff);
free(sfrreq); free(sfcreq); free(slrreq); free(slcreq);
free(rfrreq); free(rfcreq); free(rlrreq); free(rlcreq);

MPI_Type_free(&double_strided_vect);

MPI_Finalize();

return 0;
}
