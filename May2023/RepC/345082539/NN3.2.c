#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


#define ALPHA 0.1
#define NUM_TRAIN 3000

#define L1_NEURONS 100
#define L2_NEURONS 10

#define X_DIM 12
#define L1_DIM (X_DIM+1)
#define L2_DIM (L1_NEURONS+1)


double X[X_DIM] = {1,0,1,0,1,0,1,0,1,0,1,0};
double B[L2_NEURONS] = {0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};

double WL1[L1_NEURONS][L1_DIM];
double WL2[L2_NEURONS][L2_DIM];

double DL1[L1_NEURONS];
double DL2[L2_NEURONS];

double OL1[L1_NEURONS];
double OL2[L2_NEURONS];

double DELTA_L1[L1_NEURONS];
double DELTA_L2[L2_NEURONS];


void weights_init(void);

double sigmoid(double y);
double sigmoid_derivative(double y);

void activate_NN(double *vector_in);

void update_deltas(double *desired);
void update_weights(double *vector_in);
void train_NN(double *vector_in, double *desired);

void print_results(void);
void print_weights(void);
void print_array(double *arr, int length);



int main() {
double err;

weights_init();

for(int i=0;i<NUM_TRAIN;i++){
train_NN(X, B);

err = 0.0;
for(int j=0;j<L2_NEURONS;j++){
err += fabs(OL2[j] - B[j]);
}
printf("TRAIN: %3d ERROR: %.4f\n", i+1, err);
}

return 0;
}

void weights_init(void) {
for(int neuron=0;neuron<L1_NEURONS;neuron++){
for(int input=0;input<L1_DIM;input++){
WL1[neuron][input] = (2.0 * rand() / RAND_MAX) - 1;	
}
}

for(int neuron=0;neuron<L2_NEURONS;neuron++){
for(int input=0;input<L2_DIM;input++){
WL2[neuron][input] = (2.0 * rand() / RAND_MAX) - 1;
}
}
}

double sigmoid(double y){
return 1/(1+exp(-y));
}

double sigmoid_derivative(double y){
double temp = exp(-y);
return temp/((1+temp)*(1+temp));
}

void activate_NN(double *vector_in) {
double y;

for(int i=0;i<L1_NEURONS;i++){
y = WL1[i][0];	
#pragma omp simd
for(int j=1;j<L1_DIM;j++){
y += X[j-1] * WL1[i][j];
}
DL1[i] = y;
OL1[i] = sigmoid(y);
}

for(int i=0;i<L2_NEURONS;i++){
y = WL2[i][0];	
#pragma omp simd
for(int j=1;j<L2_DIM;j++){
y += OL1[j-1] * WL2[i][j];
}
DL2[i] = y;
OL2[i] = sigmoid(y);
}
}

void update_deltas(double *desired){
for(int i=0;i<L2_NEURONS;i++){
DELTA_L2[i] = sigmoid_derivative(DL2[i]) * (desired[i] - OL2[i]); 
}

double acc;
for(int i=0;i<L1_NEURONS;i++){
acc = 0.0;
for(int k=0;k<L2_NEURONS;k++){
acc += DELTA_L2[k] * WL2[k][i+1];	
}
DELTA_L1[i] = sigmoid_derivative(DL1[i]) * acc;	
}
}

void update_weights(double *vector_in){
double delta_i;

for(int i=0;i<L1_NEURONS;i++){
delta_i = DELTA_L1[i];
WL1[i][0] += ALPHA * delta_i;
#pragma omp simd
for(int j=1;j<L1_DIM;j++){
WL1[i][j] += ALPHA * delta_i * vector_in[j-1];	
}
}

for(int i=0;i<L2_NEURONS;i++){
delta_i = DELTA_L2[i];
WL2[i][0] += ALPHA * delta_i;
#pragma omp simd
for(int j=0;j<L2_DIM;j++){
WL2[i][j] += ALPHA * delta_i * OL1[j-1];	
}
}
}

void train_NN(double *vector_in, double *desired){
activate_NN(vector_in);
update_deltas(desired);
update_weights(vector_in);
}

void print_results(void){
printf("\n----------DL1----------\n");
print_array(DL1, L1_NEURONS);
printf("\n----------OL1----------\n");
print_array(OL1, L1_NEURONS);
printf("\n----------DL2----------\n");
print_array(DL2, L2_NEURONS);
printf("\n----------OL2----------\n");
print_array(OL2, L2_NEURONS);
printf("\n----------ERROR----------\n");
double sum = 0;
for(int i=0;i<L2_NEURONS;i++){
sum += fabs(OL2[i]-B[i]);
}
printf("%.3f\n", sum);
}

void print_weights(void) {
printf("----------WL1----------\n");
for(int neuron=0;neuron<L1_NEURONS;neuron++){
for(int input=0;input<L1_DIM;input++){
printf("%.3f ", WL1[neuron][input]);
}
printf("\n");
}

printf("\n----------WL2----------\n");
for(int neuron=0;neuron<L2_NEURONS;neuron++){
for(int input=0;input<L2_DIM;input++){
printf("%.3f ", WL2[neuron][input]);
}
printf("\n");
}
}

void print_array(double *arr, int length) {
for(int i=0;i<length;i++){
printf("%.3f ", arr[i]);
}
printf("\n");
}
