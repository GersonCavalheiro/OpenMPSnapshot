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
#define LEARNING_RATE 0.0001   
#define UNITY_NEURON 1      
#define TRAIN_FILE_PATH "fashion-mnist_train.csv"
#define TEST_FILE_PATH  "fashion-mnist_test.csv"
#define TRAIN_DATA_NUMBER 60000
#define TEST_DATA_NUMBER  10000
#define PIXELS 784
#define EPOCHS 50
double TRAIN_DATA[TRAIN_DATA_NUMBER][PIXELS+1]; 
double TEST_DATA[TEST_DATA_NUMBER][PIXELS+1];   
double TRAIN_GOLDEN_OUTPUTS[TRAIN_DATA_NUMBER][LAYER2_NEURONS];
double TEST_GOLDEN_OUTPUTS[TEST_DATA_NUMBER][LAYER2_NEURONS];
double WL1[LAYER1_NEURONS][PIXELS+UNITY_NEURON];       
double WL2[LAYER2_NEURONS][LAYER1_NEURONS+UNITY_NEURON];   
double DL1[LAYER1_NEURONS]; 
double DL2[LAYER2_NEURONS]; 
double OL1[LAYER1_NEURONS]; 
double OL2[LAYER2_NEURONS]; 
double EL1[LAYER1_NEURONS]; 
double EL2[LAYER2_NEURONS]; 
double NeuronOutputDerivative(double output) {
return output * (1.0 - output);
}
void CalculateInnerStates(double *Inputs, double *InnerStates, double *Weights, int InputSize, int neurons) {
#pragma omp parallel for schedule(static, 10)
for (int i=0; i<neurons; i++) {
InnerStates[i] = 1.0 * UNITY_NEURON ? Weights[i*InputSize + InputSize - UNITY_NEURON] : 0.0; 
for (int j=0; j<InputSize-UNITY_NEURON; j++) {
InnerStates[i] += Inputs[j] * Weights[i*(InputSize) + j];
}
}
}
void CalculateOuterStates(double *InnerStates, double *OuterStates, int neurons) {
for (int i=0; i<neurons; i++) {
OuterStates[i] = 1 / (1+exp(-InnerStates[i]));
}
}
void ActivateLayer(double *Inputs, double *InnerStates, double *Outputs, double *Weights, int inputSize, int neurons) {
CalculateInnerStates(Inputs, InnerStates, Weights, inputSize, neurons);
CalculateOuterStates(InnerStates, Outputs, neurons);
}
void ActivateNeuralNetwork(double *Inputs) {
ActivateLayer(Inputs, DL1, OL1, &WL1[0][0], PIXELS+UNITY_NEURON, LAYER1_NEURONS);
ActivateLayer(Inputs, DL2, OL2, &WL2[0][0], LAYER1_NEURONS+UNITY_NEURON, LAYER2_NEURONS);
}
void InitializeLayerWeights(double *Weights, int neurons, int inps) {
for (int i=0; i<neurons; i++) {
for (int j=0; j<inps; j++) {
Weights[i*(inps) + j] = -1 + 2 * ((double)rand()) / ((double)RAND_MAX);
}
}
}
void InitializeAllWeights() {
InitializeLayerWeights(&WL1[0][0], LAYER1_NEURONS, PIXELS+UNITY_NEURON);
InitializeLayerWeights(&WL2[0][0], LAYER2_NEURONS, LAYER1_NEURONS+UNITY_NEURON);
}
void OutputLayerErrors(double *outputs, double *expected, double *errors, int neurons) {
for (int i=0; i<neurons; i++) {
errors[i] = (expected[i] - outputs[i]) * NeuronOutputDerivative(outputs[i]);
}
}
void InnerLayerErrors(double *curOutputs, double *nextWeights, double *curErrors, double *nextErrors, int curNeurons, int nextNeurons) {
for (int c=0; c<curNeurons; c++) {
double myError = 0.0;
for (int n=0; n<nextNeurons; n++) {
myError += nextWeights[n*curNeurons + c] * nextErrors[n];
}
curErrors[c] = myError * NeuronOutputDerivative(curOutputs[c]);
}
}
void UpdateLayer(double *weights, double *errors, double *inputs, int neurons, int inputsNum) {
#pragma omp parallel for schedule(static, 10)
for (int i=0; i<neurons; i++) {
for (int j=0; j<inputsNum; j++) {
weights[i*inputsNum + j] += LEARNING_RATE * errors[i] * inputs[j];
}
}
}
void UpdateLayers(double *inputs) {
UpdateLayer(&WL1[0][0], &EL1[0], inputs, LAYER1_NEURONS, PIXELS+UNITY_NEURON);
UpdateLayer(&WL2[0][0], &EL2[0], &OL1[0], LAYER2_NEURONS, LAYER1_NEURONS+UNITY_NEURON);
}
void ErrorBackPropagation(double *GoldenOutputs) {
OutputLayerErrors(&OL2[0], GoldenOutputs, &EL2[0], LAYER2_NEURONS);
InnerLayerErrors(&OL1[0], &WL2[0][0], &EL1[0], &EL2[0], LAYER1_NEURONS, LAYER2_NEURONS);
}
void TrainNeuralNetwork(double *inputs, double *GoldenOutputs) {
ErrorBackPropagation(GoldenOutputs);
UpdateLayers(inputs);
}
void AcquireTrainData() {
FILE *fp = fopen(TRAIN_FILE_PATH, "r");
char *token;
if(fp != NULL) {
char initLine[10000];
if(fgets(initLine, sizeof initLine, fp) != NULL) {
char line[PIXELS*5];
int picture = 0;
while(fgets(line, sizeof line, fp) != NULL) {
token = strtok(line, ",");
int element = 0;
while(token != NULL) {
TRAIN_DATA[picture][element++] = atoi(token);
token = strtok(NULL, ",");
}
picture++;
}
}
fclose(fp);
}
}
void AcquireTestData() {
FILE *fp = fopen(TEST_FILE_PATH, "r");
char *token;
if(fp != NULL) {
char initLine[10000];
if(fgets(initLine, sizeof initLine, fp) != NULL) {
char line[PIXELS*5];
int picture = 0;
while(fgets(line, sizeof line, fp) != NULL) {
token = strtok(line, ",");
int element = 0;
while(token != NULL) {
TEST_DATA[picture][element++] = atoi(token);
token = strtok(NULL, ",");
}
picture++;
}
}
fclose(fp);
}
}
void AcquireTrainGoldenOutputs() {
for (int p=0; p<TRAIN_DATA_NUMBER; p++) {
for (int i=0; i<LAYER2_NEURONS; i++) {
if (i == (int) TRAIN_DATA[p][0]) {
TRAIN_GOLDEN_OUTPUTS[p][i] = 0.9;
}
else 
TRAIN_GOLDEN_OUTPUTS[p][i] = 0.1;
}
}
}
void AcquireTestGoldenOutputs() {
for (int p=0; p<TEST_DATA_NUMBER; p++)
for (int i=0; i<LAYER2_NEURONS; i++) {
if (i == (int) TEST_DATA[p][0]) 
TEST_GOLDEN_OUTPUTS[p][i] = 0.9;
else 
TEST_GOLDEN_OUTPUTS[p][i] = 0.1;
}
}
double MeanSquareError(double *RealOutputs, double *GoldenOutputs, int outputSize) {
double error = 0.0;
for (int i=0; i<outputSize; i++)
error += (RealOutputs[i]-GoldenOutputs[i]) * (RealOutputs[i]-GoldenOutputs[i]);
return sqrt(error);
}
int arrayMax(double *array, int size) {
double mx = -INFINITY;
int mx_index = -1;
for (int i=0; i<size; i++) {
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
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
__time_t seconds;
srand(seconds);
int step  = 0;
int epoch = 1;
int hits = 0;
InitializeAllWeights();
AcquireTrainData();
AcquireTestData();
AcquireTrainGoldenOutputs();
AcquireTestGoldenOutputs();
printf("~~~~~ NOW TRAINING THE NETWORK... ~~~~~\n");
do {
int picture = step % TRAIN_DATA_NUMBER;
double *DataIn = &TRAIN_DATA[picture][1];
double *GoldenData = &TRAIN_GOLDEN_OUTPUTS[picture][0]; 
ActivateNeuralNetwork(DataIn);
TrainNeuralNetwork(DataIn, GoldenData);
step ++;
if ((int) TRAIN_DATA[picture][0] == arrayMax(OL2, LAYER2_NEURONS)) hits++;
if (step == TRAIN_DATA_NUMBER) {
printf("TRAINING EPOCH: %3d,  STEP: %6d", epoch, step);
printf(" ==> ACCURACY: %.2lf%%\n", 100*hits/((double) step)); 
epoch++;
hits = 0;
step = 0;
}
} while (epoch<=EPOCHS);
printf("\n~~~~~ NOW EVALUATING THE NETWORK... ~~~~~\n");
step = 0;
hits = 0;
do {
int picture = step;
double *DataIn = &TEST_DATA[picture][1];
double *GoldenData = &TEST_GOLDEN_OUTPUTS[picture][0]; 
ActivateNeuralNetwork(DataIn);
step ++;
if ((int) TEST_DATA[picture][0] == arrayMax(OL2, LAYER2_NEURONS)) hits++;
} while (step<TEST_DATA_NUMBER);
printf("EVALUATION STEP: %6d", step);
printf(" ==> ACCURACY: %.2lf%%\n", 100*hits/((double) step));   
return 0;
}
