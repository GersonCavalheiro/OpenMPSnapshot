#include <getopt.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <mpi.h>
#include <assert.h>
#include <omp.h>
#include "benchmark_mpi.h"
#define DEFAULT_ITERATIONS	10
#define DEFAULT_NTHREADS 	4
#define DEFAULT_CENTROIDS   	3
#define DEFAULT_DIMENSIONS  	3
#define DEFAULT_INPUT   	"./Data/input.csv"
#define DEFAULT_OUTPUT 		"./Data/output.csv"
#define MIN_NORM		0.000000001
void print_args(void)
{
printf("Kmeans benchmark:\n");
printf("options:\n");
printf("  -h: print help message\n");
printf("  -n: number of threads per machine (default %d)\n", DEFAULT_NTHREADS);
printf("  -c: number of centroids (default %d)\n", DEFAULT_CENTROIDS);
printf("  -d: number of dimensions (default %d)\n", DEFAULT_DIMENSIONS);
printf("  -i: path to input .csv file (default path %s)\n", DEFAULT_INPUT);
printf("  -o: path to output .csv file (default path %s)\n", DEFAULT_OUTPUT);
}
void print_point(double* point, int nr_dimensions){
int i;
for(i = 0; i < nr_dimensions - 1; i++){
printf("%lf ", point[i]);
}
printf("%lf\n", point[nr_dimensions - 1]);
}
long build_data_points(FILE* fp, double** dataset_ptr, const int dimensions, const int rank, const int group_size){
long size = 0, points_per_machine;
if ( !fscanf(fp, "# %ld\n", &size) ){
printf("Error while parsing the input file\n");
return -1;
}
if( size <= 0){
printf("Size value must be a positive integer! \n");
return -1;
}
points_per_machine = (rank == 0) ? size : size / group_size;
*dataset_ptr = (double*) malloc(sizeof(double) * points_per_machine * dimensions);
double* dataset = *dataset_ptr;
if( dataset == NULL){
printf("Memory allocation error\n");
return -1;
}
long index = 0;
long offset = points_per_machine * rank;
char buffer[180];
for(int i = 0; i < offset; i++)
fgets(buffer, 180, fp);
while( !feof(fp) && index < (points_per_machine * dimensions) ){
int i;
for(i = 0; i < dimensions - 1; i++){
if( !fscanf(fp, "%lf, ", &dataset[index + i]))
return -1;
}
if( !fscanf(fp, "%lf\n", &dataset[index + dimensions - 1]) ) 
return -1;
#ifdef DEBUG
printf("[%d] - ", rank);
print_point(&dataset[index], dimensions);
#endif
index+=dimensions;
}
return size;
}
int allocate_centroids(double** centroids_ptr, int nr_centroids, int nr_dimensions){
*centroids_ptr = (double*)(malloc(sizeof(double) * nr_centroids * nr_dimensions));
if(centroids_ptr == NULL){
printf("Memory allocation error!\n");
return -1;
}
double * centroids = *centroids_ptr;
for (int i = 0; i < nr_centroids * nr_dimensions; i++)
centroids[i] = 0;
return 0;
}
void init_centroids(double* centroids, int k, int nr_dimensions, double* dataset, long size, int rank){
int i, j;
srand(time(NULL));
for(i = 0; i < k*nr_dimensions; i+=nr_dimensions){
j = (rand() % size) * nr_dimensions;
for (int p = 0; p < nr_dimensions; p++)
centroids[i + p] = dataset[j + p];
#ifdef DEBUG
printf("[%d] - picked: %d\n", rank, j / nr_dimensions);
print_point(&centroids[i], nr_dimensions);
#endif
}
}
void copy_int_long_vector(long* src, long* dst, int nr_points, int nr_dimensions){
for(int i = 0; i < nr_points * nr_dimensions; i++){
dst[i] = src[i];
}
}
void copy_vector(double* src, double* dst, int nr_points, int nr_dimensions){
for(int i = 0; i < nr_points * nr_dimensions; i++){
dst[i] = src[i];
}
}
void set_accumulators_to_zero(long* points_accumulator, double* coordinates_accumulator, int nr_centroids, int dimensions){
for(int i = 0; i < nr_centroids; i++){
points_accumulator[i] = 0;
}
for(int i = 0; i < nr_centroids * dimensions; i++){
coordinates_accumulator[i] = 0.0;
}
}
double distance(double* v1, double* v2, int nr_dimensions){
double norm = 0.0;
for(int i = 0; i < nr_dimensions; i++){
norm += (v1[i] - v2[i]) * (v1[i] - v2[i]);
}
return norm;
}
void assign_cluster(double* dataset, double* centroids, long* points_accumulator, double* coordinates_accumulator, long nr_points, int nr_centroids, int nr_dimensions, int nr_threads){
double d_min, d_temp;
int closest_centroid = 0, j;
long i;
long points_accumulator_thread [nr_centroids];
double coordinates_accumulator_thread [nr_centroids * nr_dimensions];
for (int i = 0; i < nr_centroids; i++){
for(int j = 0; j < nr_dimensions; j++){
coordinates_accumulator_thread[i*nr_dimensions + j] = 0.0;
}
points_accumulator_thread[i] = 0;
}
#ifdef USE_OMP	
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+: points_accumulator_thread[:nr_centroids], coordinates_accumulator_thread[:nr_centroids*nr_dimensions]) private(d_min, d_temp, closest_centroid, j, i) shared(dataset, centroids, nr_points, nr_centroids, nr_dimensions)
for(i = 0; i < nr_points; i++){
closest_centroid = 0;
d_min = distance(&dataset[i * nr_dimensions], &centroids[0], nr_dimensions);
for( j = 1; j < nr_centroids; j++){
d_temp = distance(&dataset[i * nr_dimensions], &centroids[j*nr_dimensions], nr_dimensions);
if(d_temp < d_min){
closest_centroid = j;
d_min = d_temp;
}
}
#ifdef DEBUG
#endif		
points_accumulator_thread[closest_centroid]++;
for(j = 0; j < nr_dimensions; j++){
coordinates_accumulator_thread[closest_centroid * nr_dimensions + j] += dataset[i * nr_dimensions + j];
}
}
#ifdef DEBUG
for(int j = 0; j < nr_centroids; j++){
printf("accumulator centroid %d: ", j);
print_point(&coordinates_accumulator_thread[j * nr_dimensions], nr_dimensions);
printf("weight: %ld\n", points_accumulator_thread[j]);
}
#endif
copy_int_long_vector(points_accumulator_thread, points_accumulator, nr_centroids, 1);
copy_vector(coordinates_accumulator_thread, coordinates_accumulator, nr_centroids, nr_dimensions);
#else	
for(long i = 0; i < nr_points; i++){
closest_centroid = 0;
d_min = distance(&dataset[i * nr_dimensions], &centroids[0], nr_dimensions);
for(int j = 1; j < nr_centroids; j++){
d_temp = distance(&dataset[i * nr_dimensions], &centroids[j*nr_dimensions], nr_dimensions);
if(d_temp < d_min){
closest_centroid = j;
d_min = d_temp;
}
}
#ifdef DEBUG
printf("%d has ", closest_centroid);
print_point(&dataset[i * nr_dimensions], nr_dimensions);
#endif		
points_accumulator[closest_centroid]++;
for(int j = 0; j < nr_dimensions; j++){
coordinates_accumulator[closest_centroid * nr_dimensions + j] += dataset[i * nr_dimensions + j];
}
}
#endif	
}
void update_centroids(double* new_centroids, long* counter, int nr_centroids, int nr_dimensions){
for(int i = 0; i < nr_centroids; i++){
for(int j = 0; j < nr_dimensions; j++){
if(new_centroids[i * nr_dimensions + j] != 0)
new_centroids[i * nr_dimensions + j] /= counter[i];
}
}  
}
void save_to_file(FILE* f_out, double* centroids, int nr_centroids, int dimensions){
int i, j;
for (i = 0; i < nr_centroids; i++){
for ( j = 0; j < dimensions - 1; j++){
fprintf(f_out, "%lf, ", centroids[i*dimensions + j]); 			
}
fprintf(f_out, "%lf\n", centroids[i*dimensions + dimensions - 1]);
}
}
int main(int argc, char** argv) {
struct option bench_options[] = {
{"help",           		no_argument,       NULL, 'h'},
{"num-of-threads", 		required_argument, NULL, 'n'},
{"num-of-centroids",		required_argument, NULL, 'c'},
{"num-of-dimensions",		required_argument, NULL, 'd'},
{"input-file",     		required_argument, NULL, 'i'},
{"output-file",    		required_argument, NULL, 'o'},
{0,                		0,                 0,    0  }
};
int c, i;
long size = 0, iterations = 0;
int duration;
int nr_threads = 				DEFAULT_NTHREADS;
int nr_centroids = 				DEFAULT_CENTROIDS;
int nr_dimensions =			 	DEFAULT_DIMENSIONS;
char* input_file = 				DEFAULT_INPUT;
char* output_file = 				DEFAULT_OUTPUT;
int nr_machines = 0;
int rank = 0;
FILE* fp, *f_out;
double* dataset = 					NULL;
double* centroids = 					NULL;
long* points_per_centroid_accumulator = 		NULL;
long* points_per_centroid_accumulator_master = 		NULL;
double* centroids_coordinates_accumulator = 		NULL;
double* centroids_coordinates_accumulator_master = 	NULL;
struct timeval start, end;
while (1) {
c = getopt_long(argc, argv, "hn:c:d:i:o:", bench_options, &i);
if (c == -1)
break;
if (c == 0 && bench_options[i].flag == 0)
c = bench_options[i].val;
switch(c) {
case 'h':
print_args();
goto out;
case 'n':
nr_threads = atoi(optarg);
break;
case 'c':
nr_centroids = atoi(optarg);
break;
case 'd':
nr_dimensions = atoi(optarg);
break;
case 'i':
input_file = optarg;
break;
case 'o':
output_file = optarg;
break;
default:
printf("Error while processing options.\n");
goto out;
}
}
if (nr_threads <= 0) {
printf("invalid thread number\n");
goto out;
}
if (nr_centroids <= 0) {
printf("invalid number of centroids\n");
goto out;
}
if (nr_dimensions <= 0) {
printf("invalid number of dimensions\n");
goto out;
}
fp = fopen(input_file, "r");
if ( fp == NULL ) {
printf("unable to open the input file %s error number: %d\n", input_file, errno);
goto out;
}
f_out = fopen(output_file, "w");
if ( f_out == NULL ) {
printf("unable to open the output file %s error number: %d\n", output_file, errno);
goto out;
}
MPI_Init(NULL, NULL);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nr_machines);
if (rank == 0){
printf("Kmeans benchmark\n");
printf("Machine_nr:		%d\n", nr_machines);
printf("Thread_nr:		%d\n", nr_threads);
printf("Centroids_nr:		%d\n", nr_centroids);
printf("Dimensions_nr:		%d\n", nr_dimensions);
printf("Input:			%s\n", input_file);
printf("Output:			%s\n", output_file);
}
printf("rank: %d\n", rank);
#ifdef DEBUG
printf("[%d] - building the datapoints\n", rank);
#endif
size = build_data_points(fp, &dataset, nr_dimensions, rank, nr_machines);
if(size < 0) 
goto out;
fclose(fp);
if(allocate_centroids(&centroids, nr_centroids, nr_dimensions) < 0)
goto out;
points_per_centroid_accumulator = (long *)malloc(sizeof(long) * nr_centroids);
centroids_coordinates_accumulator = (double * )malloc(sizeof(double) * nr_centroids * nr_dimensions);
if (rank == 0){
#ifdef DEBUG
printf("[%d] - building the centroids\n", rank);
printf("[0] - testing norm: %lf\n", distance(&dataset[0], &dataset[0], nr_dimensions));
#endif
init_centroids(centroids, nr_centroids, nr_dimensions, dataset, size, rank);
points_per_centroid_accumulator_master = (long* )malloc(sizeof(long) * nr_centroids);
centroids_coordinates_accumulator_master = (double* )malloc(sizeof(double) * nr_centroids * nr_dimensions);
}
#ifdef DEBUG
char name[80];
int name_len;
MPI_Get_processor_name(name, &name_len);
printf("[%d] - processor name %s\n", rank, name);
#endif 
if(rank == 0){
gettimeofday(&start, NULL);
}
double norm = 1.0;
while( norm > MIN_NORM ){
MPI_Bcast(centroids, nr_centroids*nr_dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
set_accumulators_to_zero(points_per_centroid_accumulator, centroids_coordinates_accumulator, nr_centroids, nr_dimensions);
assign_cluster(dataset, centroids, points_per_centroid_accumulator, centroids_coordinates_accumulator, size / nr_machines, nr_centroids, nr_dimensions, nr_threads);
MPI_Reduce(points_per_centroid_accumulator, points_per_centroid_accumulator_master, nr_centroids, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(centroids_coordinates_accumulator, centroids_coordinates_accumulator_master, nr_centroids * nr_dimensions, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
if(rank == 0){
#ifdef DEBUG
for(int j = 0; j < nr_centroids; j++){
printf("accumulator centroid %d: ", j);
print_point(&centroids_coordinates_accumulator_master[j * nr_dimensions], nr_dimensions);				  
}
#endif
update_centroids(centroids_coordinates_accumulator_master, points_per_centroid_accumulator_master, nr_centroids, nr_dimensions);
norm = distance(centroids, centroids_coordinates_accumulator_master, nr_centroids * nr_dimensions);
#ifdef DEBUG
printf("iteration %ld norm: %lf\n", iterations, norm);
for(int j = 0; j < nr_centroids; j++){
printf("old centroid %d: ", j);
print_point(&centroids[j * nr_dimensions], nr_dimensions);
printf("new centroid %d: ", j);
print_point(&centroids_coordinates_accumulator_master[j * nr_dimensions], nr_dimensions);
printf("weight: %ld\n", points_per_centroid_accumulator_master[j]);  
}
#endif
copy_vector(centroids_coordinates_accumulator_master, centroids, nr_centroids, nr_dimensions);
}
MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
iterations++;
}
if (rank == 0){
gettimeofday(&end, NULL);
duration = (end.tv_sec * 1000 + end.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
printf("Duration: %d ms\n", duration);		
printf("Iterations: %ld\n", iterations);
printf("Centroids: \n");
for( i = 0; i < nr_centroids * nr_dimensions; i+=nr_dimensions){
printf("%d: ", i / nr_dimensions);
print_point(&centroids[i], nr_dimensions);
}
save_to_file(f_out, centroids, nr_centroids, nr_dimensions);
}
out:
#ifdef DEBUG
printf("[%d] - free memory\n", rank);
#endif
if (rank == 0){
free(points_per_centroid_accumulator_master);
free(centroids_coordinates_accumulator_master);
}
free(centroids);
free(points_per_centroid_accumulator);
free(centroids_coordinates_accumulator);
free(dataset);
MPI_Finalize();
return 0;
}
