#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "omp.h"
#include "string.h"
#define LAYER1_NEURONS 100    
#define LAYER2_NEURONS 10    
#define DEBUG 0            
#define LEARNING_RATE 0.2   
#define UNITY_NEURON 1      
#define TRAIN_FILE_PATH "fashion-mnist_train.csv"
#define TEST_FILE_PATH  "fashion-mnist_test.csv"
#define TRAIN_DATA_NUMBER 60000
#define TEST_DATA_NUMBER  10000
#define PIXELS 784
#define EPOCHS 15
#define BATCH_SIZE 100
double TRAIN_DATA[TRAIN_DATA_NUMBER][PIXELS+1]; 
double TEST_DATA[TEST_DATA_NUMBER][PIXELS+1];   
double TRAIN_GOLDEN_OUTPUTS[TRAIN_DATA_NUMBER][LAYER2_NEURONS];
double TEST_GOLDEN_OUTPUTS[TEST_DATA_NUMBER][LAYER2_NEURONS];
double WL1[LAYER1_NEURONS][PIXELS+UNITY_NEURON];       
double WL2[LAYER2_NEURONS][LAYER1_NEURONS+UNITY_NEURON];   
double WL1delta[LAYER1_NEURONS][PIXELS+UNITY_NEURON];       
double WL2delta[LAYER2_NEURONS][LAYER1_NEURONS+UNITY_NEURON];   
double DL1[LAYER1_NEURONS]; 
double DL2[LAYER2_NEURONS]; 
double OL1[LAYER1_NEURONS]; 
double OL2[LAYER2_NEURONS]; 
double EL1[LAYER1_NEURONS]; 
double EL2[LAYER2_NEURONS]; 
void InitializeWeights() {
for (int i=0; i<LAYER1_NEURONS; i++) {
for (int j=0; j<PIXELS+UNITY_NEURON; j++) {
WL1[i][j] = 2 * ((double)rand())/((double)RAND_MAX) - 1;
}
}
for (int i=0; i<LAYER2_NEURONS; i++) {
for (int j=0; j<LAYER1_NEURONS+UNITY_NEURON; j++) {
WL2[i][j] = 2 * ((double)rand())/((double)RAND_MAX) - 1;
}
}
}
void InitializeDeltas() {
for (int i=0; i<LAYER1_NEURONS; i++) 
for (int j=0; j<PIXELS+UNITY_NEURON; j++)
WL1delta[i][j] = 0;
for (int i=0; i<LAYER2_NEURONS; i++) 
for (int j=0; j<LAYER1_NEURONS+UNITY_NEURON; j++)
WL2delta[i][j] = 0;
}
void AcquireData() {
FILE *fp1 = fopen(TRAIN_FILE_PATH, "r");
char *token1;
if(fp1 != NULL) {
char line[PIXELS*6];
int picture = 0;
while(fgets(line, sizeof line, fp1) != NULL) {
token1 = strtok(line, ",");
int element = 0;
while(token1 != NULL) {
TRAIN_DATA[picture][element++] = atoi(token1);
token1 = strtok(NULL, ",");
}
picture++;  
}
fclose(fp1);
}
FILE *fp2 = fopen(TEST_FILE_PATH, "r");
char *token2;
if(fp2 != NULL) {
char line[PIXELS*6];
int picture = 0;
while(fgets(line, sizeof line, fp2) != NULL) {
token2 = strtok(line, ",");
int element = 0;
while(token2 != NULL) {
TEST_DATA[picture][element++] = atoi(token2);
token2 = strtok(NULL, ",");
}
picture++;
}
fclose(fp2);
}
}
void AcquireGoldenOutputs() {
for (int p=0; p<TRAIN_DATA_NUMBER; p++) {
for (int i=0; i<LAYER2_NEURONS; i++) {
if (i == (int) TRAIN_DATA[p][0]) {
TRAIN_GOLDEN_OUTPUTS[p][i] = 0.9;
}
else 
TRAIN_GOLDEN_OUTPUTS[p][i] = 0.1;
}
}
for (int p=0; p<TEST_DATA_NUMBER; p++)
for (int i=0; i<LAYER2_NEURONS; i++) {
if (i == (int) TEST_DATA[p][0]) 
TEST_GOLDEN_OUTPUTS[p][i] = 0.9;
else 
TEST_GOLDEN_OUTPUTS[p][i] = 0.1;
}
}
void ActivateNetwork(double input[LAYER1_NEURONS]) {
for (int n=0; n<LAYER1_NEURONS; n++) {
double innerState = (UNITY_NEURON==1)?WL1[n][PIXELS]:0;
for (int i=0; i<PIXELS; i++) {
innerState += input[i] * WL1[n][i];
}
OL1[n] = 1 / (1+exp(-innerState));
}
for (int n=0; n<LAYER2_NEURONS; n++) {
double innerState = (UNITY_NEURON==1)?WL2[n][LAYER1_NEURONS]:0;
for (int i=0; i<LAYER1_NEURONS; i++) {
innerState += OL1[i] * WL2[n][i];
}
OL2[n] = 1 / (1+exp(-innerState));
}
}
double NeuronOutputDerivative(double output) {
return output * (1.0 - output);
}
void ErrorBackPropagation(double expectedOutput[LAYER2_NEURONS]) {
for (int c=0; c<LAYER2_NEURONS; c++) {
EL2[c] = (expectedOutput[c] - OL2[c]) * NeuronOutputDerivative(OL2[c]);
}
for (int c=0; c<LAYER1_NEURONS; c++) {
double error = 0.0;
for (int n=0; n<LAYER2_NEURONS; n++) {
error += WL1[n][c] * EL2[n];
}
EL1[c] = error * NeuronOutputDerivative(OL1[c]);
}
}
void UpdateWeightsDeltas(double inputs[PIXELS]) {
for (int n=0; n<LAYER1_NEURONS; n++) {
for (int i=0; i<PIXELS; i++) {
WL1delta[n][i] += EL1[n] * inputs[i] / ((double) BATCH_SIZE);
}
if (UNITY_NEURON==1) WL1delta[n][PIXELS] += EL1[n] / ((double) BATCH_SIZE);
}
for (int n=0; n<LAYER2_NEURONS; n++) {
for (int i=0; i<LAYER1_NEURONS; i++) {
WL2delta[n][i] += EL2[n] * OL1[i] / ((double) BATCH_SIZE);
}
if (UNITY_NEURON==1) WL2delta[n][LAYER1_NEURONS] += EL2[n] / ((double) BATCH_SIZE);
}
}
void UpdateWeights() {
for (int n=0; n<LAYER1_NEURONS; n++) {
for (int i=0; i<PIXELS+UNITY_NEURON; i++) {
WL1[n][i] += LEARNING_RATE * WL1delta[n][i];
}
}
for (int n=0; n<LAYER2_NEURONS; n++) {
for (int i=0; i<LAYER1_NEURONS+UNITY_NEURON; i++) {
WL2[n][i] += LEARNING_RATE * WL2delta[n][i];
}
}
}
int estimatedClass(double *array) {
double mx = 0;
int mx_index = -1;
for (int i=0; i<LAYER2_NEURONS; i++) {
if (array[i] > mx) {
mx = array[i];
mx_index = i;
}
}
return mx_index;
}
int main() {
printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("This program implements a Neural Network of %d Layers.\n", 2);
printf("Inputs: %d, Hidden layer neurons: %d, Output layer neurons: %d\n", PIXELS, LAYER1_NEURONS, LAYER2_NEURONS);
printf("Epochs: %d, Batches per epoch: %d, Train data: %d\n", EPOCHS, TRAIN_DATA_NUMBER/BATCH_SIZE, TRAIN_DATA_NUMBER);
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
__time_t seconds;
srand(seconds);
InitializeWeights();
AcquireData();
AcquireGoldenOutputs();
int step=0, batch=0, epoch=0;
printf("--- Now training the Neural Network using the train data... ---\n");
for (int epoch=0; epoch<EPOCHS; epoch++) {
printf("> Epoch: %2d", epoch+1);
int hits = 0;
for (int batch=0; batch<TRAIN_DATA_NUMBER/BATCH_SIZE; batch++) {
InitializeDeltas();
for (int step=0; step<BATCH_SIZE; step++) {
int picture = step % TRAIN_DATA_NUMBER;
ActivateNetwork(&TRAIN_DATA[picture][1]);
ErrorBackPropagation(TRAIN_GOLDEN_OUTPUTS[picture]);
UpdateWeightsDeltas(&TRAIN_DATA[picture][1]);
if (estimatedClass(OL2) == (int) TRAIN_DATA[picture][0]) hits++;
}
UpdateWeights();
}
printf("  -->  Accuracy: %.2lf%%\n", ((double)100*hits)/((double)TRAIN_DATA_NUMBER));
}
int hits = 0;
printf("\n--- Now evaluating the Neural Network using the test data... --\n");
for (int step=0; step<TEST_DATA_NUMBER; step++) {
int picture = step ;
ActivateNetwork(&TEST_DATA[picture][1]);
if (estimatedClass(OL2) == (int) TEST_DATA[picture][0]) hits++;
printf("\r> Test data: %d  Total Accuracy: %.2lf%%", step+1, ((double)100*hits)/((double)TEST_DATA_NUMBER));
} 
}
