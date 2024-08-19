

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "chol.h"
#include <sys/time.h>
#include <pthread.h>

#define NUM_THREADS 16


typedef struct barrier_struct {
pthread_mutex_t mutex; 
pthread_cond_t condition; 
int counter; 
} barrier_t;

barrier_t barrier = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
barrier_t barrier2 = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
barrier_t barrier3 = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};

typedef struct args_for_thread_s {
int thread_id; 
int num_elements; 
float *matrixA; 
float *matrixU; 
int num_rows; 
} ARGS_FOR_THREAD;

Matrix allocate_matrix(int num_rows, int num_columns, int init);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern int chol_gold(const Matrix, Matrix);
extern int check_chol(const Matrix, const Matrix);
void chol_using_pthreads(float *, float*, unsigned int);
int chol_using_openmp(const Matrix, Matrix);
void * chol_pthread(void *);
void barrier_sync(barrier_t *);


int main(int argc, char** argv) 
{	
if(argc > 1){
printf("Error. This program accepts no arguments. \n");
exit(0);
}		
struct timeval start, stop; 
Matrix A; 
Matrix reference; 
Matrix U_pthreads; 
Matrix U_openmp; 

srand(time(NULL));

int success = 0;
while(!success){
A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
if(A.elements != NULL)
success = 1;
}



reference  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 
U_pthreads =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 
U_openmp =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 


printf("Performing Cholesky decomposition on the CPU using the single-threaded version. \n");
gettimeofday(&start, NULL);
int status = chol_gold(A, reference);
gettimeofday(&stop, NULL);
if(status == 0){
printf("Cholesky decomposition failed. The input matrix is not positive definite. \n");
exit(0);
}
float serialTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
printf("Serial program run time = %.4f s. \n", serialTime);



printf("Cholesky decomposition on the CPU was successful. \n");



gettimeofday(&start, NULL);
chol_using_pthreads(A.elements, U_pthreads.elements, A.num_rows);
gettimeofday(&stop, NULL);
float pthreadTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
printf("Pthread time = %.4f\n", pthreadTime);
printf("PThreads Speedup = %.4f \n", serialTime/pthreadTime);	



gettimeofday(&start, NULL);
chol_using_openmp(A, U_openmp);
gettimeofday(&stop, NULL);
float openmpTime = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
printf("OpenMP run time = %.4f s. \n", openmpTime);
printf("OpenMP Speedup = %.4f \n", serialTime/openmpTime);


if(check_chol(A, U_pthreads) == 0) 
printf("Error performing Cholesky decomposition using pthreads. \n");
else
printf("Cholesky decomposition using pthreads was successful. \n");

if(check_chol(A, U_openmp) == 0) 
printf("Error performing Cholesky decomposition using openmp. \n");
else	
printf("Cholesky decomposition using openmp was successful. \n");









free(A.elements); 	
free(U_pthreads.elements);	
free(U_openmp.elements);
free(reference.elements); 
return 1;
}


void chol_using_pthreads(float* A, float* U, unsigned int num_elements)
{
unsigned int size = num_elements * num_elements;
unsigned int i, k;
for (i = 0; i < size; i ++)
U[i] = A[i];

pthread_t thread_id[NUM_THREADS]; 
pthread_attr_t attributes; 
pthread_attr_init(&attributes); 
ARGS_FOR_THREAD * args[NUM_THREADS];

for(i = 0; i < NUM_THREADS; i++) {
args[i] = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
args[i]->thread_id = i; 
args[i]-> num_elements = num_elements;
args[i]->matrixA = A;
args[i]->matrixU = U;
}


for (i = 0; i < NUM_THREADS; i++) {
pthread_create(&thread_id[i], &attributes, chol_pthread, (void*) args[i]);
}

for (i = 0; i < NUM_THREADS; i++) {
pthread_join(thread_id[i], NULL);
}



for (i = 0; i < NUM_THREADS; i++) 
free((void*)args[i]);
}

void * chol_pthread(void * args) {

ARGS_FOR_THREAD * my_args = (ARGS_FOR_THREAD*)args;
unsigned int i, j, k; 
int num_elements = my_args->num_elements;
unsigned int firstIndex, lastIndex, chunk, offset;
float * U = my_args->matrixU;


for (k = 0; k < num_elements; k++) {
chunk = (int)floor((float)(num_elements - k) / (float) NUM_THREADS); 
offset = my_args->thread_id*chunk;
firstIndex = k + 1 + my_args->thread_id*chunk;
lastIndex= firstIndex + chunk;

if (my_args->thread_id == (NUM_THREADS-1)) {
lastIndex = num_elements;
}
if (my_args->thread_id == 0)
U[k * num_elements + k] = sqrt(U[k * num_elements + k]);

if(U[k * num_elements + k] <= 0) {
printf("Cholesky decomposition failed. \n");
exit(0); }


barrier_sync(&barrier);

for(j = firstIndex; j < lastIndex; j++)
U[k * num_elements + j] /= U[k * num_elements + k]; 


barrier_sync(&barrier2);

for(i = firstIndex; i < lastIndex; i++)
for(j = i; j < num_elements; j++)
U[i * num_elements + j] -= U[k * num_elements + i] * U[k * num_elements + j];

barrier_sync(&barrier3);
}

for(i = 0; i < num_elements; i++)
for(j = 0; j < i; j++)
U[i * num_elements + j] = 0.0;



}




int chol_using_openmp(const Matrix A, Matrix U)
{

unsigned int i, j, k; 
unsigned int size = A.num_rows * A.num_columns;

#pragma omp parallel for	num_threads(NUM_THREADS) default(none) private(i) shared(U, size)
for (i = 0; i < size; i ++)
U.elements[i] = A.elements[i];

for(k = 0; k < U.num_rows; k++){
U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);
if(U.elements[k * U.num_rows + k] <= 0){
printf("Cholesky decomposition failed. \n");
return 0;
}

#pragma omp parallel for num_threads(NUM_THREADS)
for(j = (k + 1); j < U.num_rows; j++)
U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k]; 

#pragma omp parallel for num_threads(NUM_THREADS) default(none) private(i,j) shared(U,k)
for(i = (k + 1); i < U.num_rows; i++)
for(j = i; j < U.num_rows; j++)
U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j];

}

for(i = 0; i < U.num_rows; i++)
for(j = 0; j < i; j++)
U.elements[i * U.num_rows + j] = 0.0;

return 1;
}


Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
Matrix M;
M.num_columns = M.pitch = num_columns;
M.num_rows = num_rows;
int size = M.num_rows * M.num_columns;

M.elements = (float *) malloc(size * sizeof(float));
for(unsigned int i = 0; i < size; i++){
if(init == 0) M.elements[i] = 0; 
else
M.elements[i] = (float)rand()/(float)RAND_MAX;
}
return M;
}	


void 
barrier_sync(barrier_t *barrier)
{
pthread_mutex_lock(&(barrier->mutex));
barrier->counter++;

if(barrier->counter == NUM_THREADS){
barrier->counter = 0; 
pthread_cond_broadcast(&(barrier->condition)); 
} 
else{

while((pthread_cond_wait(&(barrier->condition), &(barrier->mutex))) != 0); 		  
}

pthread_mutex_unlock(&(barrier->mutex));
}


