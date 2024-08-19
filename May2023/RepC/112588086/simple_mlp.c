#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include "load_parameters.h"
#include "read_image.h"
static void softmax(float *input, size_t input_len) {
float * Z = input;
int K = input_len;
float sum = 0;
for(int j = 0; j < K; j++){
sum += exp(Z[j]);
}
for(int i = 0; i < K; i++){
input[i] = exp(input[i])/sum;
} 
}
static void relu(float *input, int input_len){
for(int i = 0; i < input_len; i++){
input[i] = fmax((float)0, input[i]);
}
}
void calculate_layer(int layer_num, int*layer_size, float* input_matrix, float **weights,
float* biases, float* out_matrix, int activation_function)
{
int i, j;
int X = layer_size[layer_num-1];
int Y = layer_size[layer_num];
int x = 20, y=10;
#pragma omp parallel private(i,j) shared(Y, X, out_matrix, input_matrix, weights, biases)
{
#pragma omp for schedule(static)
for(i = 0; i < Y; i++)
{
for(j = 0; j < X; j++)
{
out_matrix[i] += input_matrix[j]* (weights[i][j]);
}
out_matrix[i] += biases[i];
}
}
if (activation_function == 1) {
relu(out_matrix, Y);
} else if(activation_function == 2) {
softmax(out_matrix, Y);
}   
}
void forward_propagation(int layer_num, int *layer_size,float ***weights, 
float **biases, float *input, float **output){
float **L = (float**) malloc(layer_num*sizeof(float*));
for (size_t i = 0; i < layer_num; i++)
{
L[i] = (float*) calloc(layer_size[i+1], sizeof(float));
}
calculate_layer(1, layer_size, input, weights[0], biases[0], L[0], 1); 
for (size_t i = 1; i < layer_num; i++)
{
int act_fn = 1;
if(i==layer_num-1) act_fn = 2;
calculate_layer(i+1, layer_size, L[i-1], 
weights[i], biases[i], L[i], act_fn);
}
(*output) = L[layer_num-1];
}
void main(int argc, char **argv)
{
char *network_name = argv[1];
int *layer_sizes;
float ***weights;
float **biases;
int layer_num = load_parameters(network_name, &layer_sizes, 
&weights, &biases);
float *flatten_image;
char tmp[50];
strcpy(tmp, argv[2]);
read_png_file(tmp, &flatten_image);
struct timeval  tv1, tv2;
gettimeofday(&tv1, NULL);
float *output;
forward_propagation(layer_num, layer_sizes, 
weights, biases, flatten_image, &output);
gettimeofday(&tv2, NULL);
printf ("\n\t\t*\tTotal time   CPU = %f microseconds \n",
(float) (tv2.tv_usec - tv1.tv_usec) +
(float) 1000000*(tv2.tv_sec - tv1.tv_sec));
for(int i = 0; i < layer_sizes[layer_num]; i++)
{
printf("  %d\t", i);
}
printf("\n");
for(int i = 0; i < layer_sizes[layer_num]; i++)
{
printf("%.2f\t", output[i]);
}
printf("\n");
}
