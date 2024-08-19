#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <omp.h>
void run_test(int);
void compute_gold(int *, int *, int, int);
void compute_using_openmp(int *, int *, int, int);
void check_histogram(int *, int, int);
#define HISTOGRAM_SIZE 500
#define NUM_THREADS 16
int 
main( int argc, char** argv) 
{
if(argc != 2){
printf("Usage: histogram <num elements> \n");
exit(0);	
}
int num_elements = atoi(argv[1]);
run_test(num_elements);
return 0;
}
void run_test(int num_elements) 
{
double diff;
int i; 
int *reference_histogram = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); 
int *histogram_using_openmp = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); 
int size = sizeof(int) * num_elements;
int *input_data = (int *)malloc(size);
for(i = 0; i < num_elements; i++)
input_data[i] = floorf((HISTOGRAM_SIZE - 1) * (rand()/(float)RAND_MAX));
printf("Creating the reference histogram. \n"); 
struct timeval start, stop;	
gettimeofday(&start, NULL);
compute_gold(input_data, reference_histogram, num_elements, HISTOGRAM_SIZE);
gettimeofday(&stop, NULL);
printf("CPU run time = %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
printf("Creating histogram using OpenMP. \n");
struct timeval start_2, stop_2;
gettimeofday(&start_2, NULL);
compute_using_openmp(input_data, histogram_using_openmp, num_elements, HISTOGRAM_SIZE);
gettimeofday(&stop_2, NULL);
printf("OpenMP CPU run time = %0.04f s. \n", (float)(stop_2.tv_sec - start_2.tv_sec + (stop_2.tv_usec - start_2.tv_usec)/(float)1000000));
check_histogram(histogram_using_openmp, num_elements, HISTOGRAM_SIZE);
diff = 0.0;
for(i = 0; i < HISTOGRAM_SIZE; i++)
diff = diff + abs(reference_histogram[i] - histogram_using_openmp[i]);
printf("Difference between the reference and OpenMP results: %f. \n", diff);
free(input_data);
free(reference_histogram);
free(histogram_using_openmp);
}
void compute_gold(int *input_data, int *histogram, int num_elements, int histogram_size)
{
int i;
for(i = 0; i < histogram_size; i++) 
histogram[i] = 0; 
for(i = 0; i < num_elements; i++){
histogram[input_data[i]]++;
}
}
void compute_using_openmp(int *input_data, int *histogram, int num_elements, int histogram_size)
{	
#pragma omp parallel num_threads(NUM_THREADS)
{	
int i;
int hist_temp[NUM_THREADS+1][HISTOGRAM_SIZE]; 
const int current_thread = omp_get_thread_num();
for (i = 0; i < histogram_size; i++){
histogram[i] = 0;
hist_temp[current_thread][i] = 0;}
#pragma omp barrier
#pragma omp for nowait
for (i = 0; i < num_elements; i++) hist_temp[current_thread][input_data[i]]++;
#pragma omp critical
for (i = 0; i < histogram_size; i++) histogram[i] += hist_temp[current_thread][i];
} 
}
void check_histogram(int *histogram, int num_elements, int histogram_size)
{
int sum = 0;
for(int i = 0; i < histogram_size; i++)
sum += histogram[i];
printf("Number of histogram entries = %d. \n", sum);
if(sum == num_elements)
printf("Histogram generated successfully. \n");
else
printf("Error generating histogram. \n");
}
