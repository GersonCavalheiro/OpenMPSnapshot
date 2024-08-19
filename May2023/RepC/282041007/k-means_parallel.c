#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define MAX_ITER 100
#define THRESHOLD 1e-6

int number_of_points_global;
int number_of_threads_global;
int number_of_iterations_global;
int K_global;
int *data_points_global;
float *iter_centroids_global;
int *data_point_cluster_global;
int **iter_cluster_count_global;

double delta_global = THRESHOLD + 1;

void kmeans_openmp_thread(int *tid)
{
int *id = (int *)tid;

int data_length_per_thread = number_of_points_global / number_of_threads_global;
int start = (*id) * data_length_per_thread;
int end = start + data_length_per_thread;
if (end + data_length_per_thread > number_of_points_global)
{
end = number_of_points_global;
data_length_per_thread = number_of_points_global - start;
}

printf("Thread ID:%d, start:%d, end:%d\n", *id, start, end);

int i = 0, j = 0;
double min_dist, current_dist;

int *point_to_cluster_id = (int *)malloc(data_length_per_thread * sizeof(int));

float *cluster_points_sum = (float *)malloc(K_global * 3 * sizeof(float));

int *points_inside_cluster_count = (int *)malloc(K_global * sizeof(int));

int iter_counter = 0;
while ((delta_global > THRESHOLD) && (iter_counter < MAX_ITER))
{
for (i = 0; i < K_global * 3; i++)
cluster_points_sum[i] = 0.0;

for (i = 0; i < K_global; i++)
points_inside_cluster_count[i] = 0;

for (i = start; i < end; i++)
{
min_dist = DBL_MAX;
for (j = 0; j < K_global; j++)
{
current_dist = pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3] - (float)data_points_global[i * 3]), 2.0) +
pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 1] - (float)data_points_global[i * 3 + 1]), 2.0) +
pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 2] - (float)data_points_global[i * 3 + 2]), 2.0);
if (current_dist < min_dist)
{
min_dist = current_dist;
point_to_cluster_id[i - start] = j;
}
}

points_inside_cluster_count[point_to_cluster_id[i - start]] += 1;

cluster_points_sum[point_to_cluster_id[i - start] * 3] += (float)data_points_global[i * 3];
cluster_points_sum[point_to_cluster_id[i - start] * 3 + 1] += (float)data_points_global[i * 3 + 1];
cluster_points_sum[point_to_cluster_id[i - start] * 3 + 2] += (float)data_points_global[i * 3 + 2];
}


#pragma omp critical
{
for (i = 0; i < K_global; i++)
{
if (points_inside_cluster_count[i] == 0)
{
printf("Unlikely situation!\n");
continue;
}
iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 1]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 2]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);

iter_cluster_count_global[iter_counter][i] += points_inside_cluster_count[i];
}
}


#pragma omp barrier
if (*id == 0)
{
double temp_delta = 0.0;
for (i = 0; i < K_global; i++)
{
temp_delta += (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]);
}
delta_global = temp_delta;
number_of_iterations_global++;
}

#pragma omp barrier
iter_counter++;
}

for (i = start; i < end; i++)
{
data_point_cluster_global[i * 4] = data_points_global[i * 3];
data_point_cluster_global[i * 4 + 1] = data_points_global[i * 3 + 1];
data_point_cluster_global[i * 4 + 2] = data_points_global[i * 3 + 2];
data_point_cluster_global[i * 4 + 3] = point_to_cluster_id[i - start];
assert(point_to_cluster_id[i - start] >= 0 && point_to_cluster_id[i - start] < K_global);
}
}

void kmeans_omp(int num_threads,
int N,
int K,
int *data_points,
int **data_points_cluster_id,
float **iter_centroids,
int *number_of_iterations)
{


number_of_points_global = N;
number_of_threads_global = num_threads;
number_of_iterations_global = 0;
K_global = K;
data_points_global = data_points;

*data_points_cluster_id = (int *)malloc(N * 4 * sizeof(int));   
data_point_cluster_global = *data_points_cluster_id;


iter_centroids_global = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

int i = 0;
for (i = 0; i < K; i++)
{
iter_centroids_global[i * 3] = data_points[i * 3];
iter_centroids_global[i * 3 + 1] = data_points[i * 3 + 1];
iter_centroids_global[i * 3 + 2] = data_points[i * 3 + 2];
}

for (i = 0; i < K; i++)
{
printf("initial centroid #%d: %f,%f,%f\n", i + 1, iter_centroids_global[i * 3], iter_centroids_global[i * 3 + 1], iter_centroids_global[i * 3 + 2]);
}


iter_cluster_count_global = (int **)malloc(MAX_ITER * sizeof(int *));
for (i = 0; i < MAX_ITER; i++)
{
iter_cluster_count_global[i] = (int *)calloc(K, sizeof(int));
}

omp_set_num_threads(num_threads);

#pragma omp parallel
{
int ID = omp_get_thread_num();
printf("Thread: %d created!\n", ID);
kmeans_openmp_thread(&ID);
}

*number_of_iterations = number_of_iterations_global;

int iter_centroids_size = (*number_of_iterations + 1) * K * 3;
printf("Number of iterations :%d\n", *number_of_iterations);
*iter_centroids = (float *)calloc(iter_centroids_size, sizeof(float));
for (i = 0; i < iter_centroids_size; i++)
{
(*iter_centroids)[i] = iter_centroids_global[i];
}

for (i = 0; i < K; i++)
{
printf("centroid #%d: %f,%f,%f\n", i + 1, (*iter_centroids)[((*number_of_iterations) * K + i) * 3], (*iter_centroids)[((*number_of_iterations) * K + i) * 3 + 1], (*iter_centroids)[((*number_of_iterations) * K + i) * 3 + 2]);
}

}

void dataset_in(const char *dataset_filename, int *N, int **data_points)
{
FILE *fin = fopen(dataset_filename, "r");
fscanf(fin, "%d", N);
*data_points = (int *)malloc(sizeof(int) * ((*N) * 3));
int i = 0;
for (i = 0; i < (*N) * 3; i++)
{
fscanf(fin, "%d", (*data_points + i));
}
fclose(fin);
}

void clusters_out(const char *cluster_filename, int N, int *cluster_points)
{
FILE *fout = fopen(cluster_filename, "w");
int i = 0;
for (i = 0; i < N; i++)
{
fprintf(fout, "%d %d %d %d\n",
*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
*(cluster_points + (i * 4) + 2), *(cluster_points + (i * 4) + 3));
}
fclose(fout);
}

void centroids_out(const char *centroid_filename, int K, int number_of_iterations, float *iter_centroids)
{
FILE *fout = fopen(centroid_filename, "w");
int i = 0;
for (i = 0; i < number_of_iterations + 1; i++)
{
int j = 0;
for (j = 0; j < K; j++)
{
fprintf(fout, "%f %f %f, ",
*(iter_centroids + (i * K + j) * 3),		 
*(iter_centroids + (i * K + j) * 3 + 1),  
*(iter_centroids + (i * K + j) * 3 + 2)); 
}
fprintf(fout, "\n");
}
fclose(fout);
}

void main()
{

int N;					
int K;					
int num_threads;		
int* data_points;		
int* cluster_points;	
float* iter_centroids;			
int number_of_iterations;    

char *dataset_filename = "dataset-10000.txt";

printf("Enter No. of Clusters: ");
scanf("%d", &K);
printf("Enter No. of threads to be used: ");
scanf("%d",&num_threads);

printf("\nFollowing files should be in the same directory as of program\n");
printf("1 for 10000 datapoints\n");
printf("2 for 50000 datapoints\n");
printf("3 for 100000 datapoints\n");
printf("4 for 200000 datapoints\n");
printf("5 for 400000 datapoints\n");
printf("6 for 500000 datapoints\n");
printf("7 for 600000 datapoints\n");
printf("8 for 800000 datapoints\n");
printf("9 for 1000000 datapoints\n");
printf("\nEnter the number of dataset file to input: ");

int x;
scanf("%d",&x);

switch (x)
{
case 1:
dataset_filename = "dataset-10000.txt";
break;
case 2:
dataset_filename = "dataset-50000.txt";
break;
case 3:
dataset_filename = "dataset-100000.txt";
break;
case 4:
dataset_filename = "dataset-200000.txt";
break;
case 5:
dataset_filename = "dataset-400000.txt";
break;
case 6:
dataset_filename = "dataset-500000.txt";
break;
case 7:
dataset_filename = "dataset-600000.txt";
break;
case 8:
dataset_filename = "dataset-800000.txt";
break;
case 9:
dataset_filename = "dataset-1000000.txt";
break;
default:
dataset_filename = "dataset-10000.txt";
break;
}


double start_time, end_time;
double computation_time;



dataset_in (dataset_filename, &N, &data_points);



start_time = omp_get_wtime();
kmeans_omp(num_threads, N, K, data_points, &cluster_points, &iter_centroids, &number_of_iterations);
end_time = omp_get_wtime();

char num_threads_char[3];
snprintf(num_threads_char,10,"%d", num_threads);

char file_index_char[2];
snprintf(file_index_char,10,"%d", x);

char cluster_filename[105] = "cluster_output_threads";
strcat(cluster_filename,num_threads_char);
strcat(cluster_filename,"_dataset");
strcat(cluster_filename,file_index_char);
strcat(cluster_filename,".txt");

char centroid_filename[105] = "centroid_output_threads";
strcat(centroid_filename,num_threads_char);
strcat(centroid_filename,"_dataset");
strcat(centroid_filename,file_index_char);
strcat(centroid_filename,".txt");


clusters_out (cluster_filename, N, cluster_points);


centroids_out (centroid_filename, K, number_of_iterations, iter_centroids);


computation_time = end_time - start_time;
printf("Time Taken: %lf \n", computation_time);

char time_file_omp[100] = "compute_time_openmp_threads";
strcat(time_file_omp,num_threads_char);
strcat(time_file_omp,"_dataset");
strcat(time_file_omp,file_index_char);
strcat(time_file_omp,".txt");

FILE *fout = fopen(time_file_omp, "a");
fprintf(fout, "%f\n", computation_time);
fclose(fout);

printf("Cluster Centroid point output file '%s' saved\n", centroid_filename);
printf("Clustered points output file '%s' saved\n", cluster_filename);
printf("Computation time output file '%s' saved\n", time_file_omp);

}