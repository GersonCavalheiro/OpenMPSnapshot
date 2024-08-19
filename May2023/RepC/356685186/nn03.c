#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#define INPUT_SIZE 800        
#define LAYER1_NEURONS 100    
#define LAYER2_NEURONS 10    
#define DEBUG 0            
#define LEARNING_RATE 0.1   
#define UNITY_NEURON 1      
double WL1[LAYER1_NEURONS][INPUT_SIZE+UNITY_NEURON];       
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
ActivateLayer(Inputs, DL1, OL1, &WL1[0][0], INPUT_SIZE+UNITY_NEURON, LAYER1_NEURONS);
ActivateLayer(Inputs, DL2, OL2, &WL2[0][0], LAYER1_NEURONS+UNITY_NEURON, LAYER2_NEURONS);
}
void InitializeLayerWeights(double *Weights, int neurons, int inps) {
for (int i=0; i<neurons; i++) {
for (int j=0; j<inps; j++) {
Weights[i*(inps) + j] = ((double)rand()) / ((double)RAND_MAX);
}
}
}
void InitializeAllWeights() {
InitializeLayerWeights(&WL1[0][0], LAYER1_NEURONS, INPUT_SIZE+UNITY_NEURON);
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
for (int i=0; i<neurons; i++) {
for (int j=0; j<inputsNum; j++) {
weights[i*inputsNum + j] += LEARNING_RATE * errors[i] * inputs[j];
}
}
}
void UpdateLayers(double *inputs) {
UpdateLayer(&WL1[0][0], &EL1[0], inputs, LAYER1_NEURONS, INPUT_SIZE+UNITY_NEURON);
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
void AcquireInputs(double *array, int size) {
for (int i=0; i<size; i++)
array[i] = -1 + 2 * ((double) rand()) / ((double) RAND_MAX);
}
void AcquireGoldenOutputs(double *array, int size) {
for (int i=0; i<size; i++)
array[i] = ((double) rand()) / ((double) RAND_MAX);
}
double MeanSquareError(double *RealOutputs, double *GoldenOutputs, int outputSize) {
double error = 0.0;
for (int i=0; i<outputSize; i++)
error += (RealOutputs[i]-GoldenOutputs[i]) * (RealOutputs[i]-GoldenOutputs[i]);
return sqrt(error);
}
int main() {
printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("This program implements a Neural Network of %d Layers.\n", 2);
printf("Inputs: %d, Hidden layer neurons: %d, Output layer neurons: %d\n", INPUT_SIZE, LAYER1_NEURONS, LAYER2_NEURONS);
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
__time_t seconds;
srand(seconds);
double *DataIn = (double*) calloc(INPUT_SIZE, sizeof(double)); 
double *GoldenData = (double*) calloc(LAYER2_NEURONS, sizeof(double)); 
int steps = 0;
int max_steps = 0;
InitializeAllWeights();
AcquireInputs(DataIn, INPUT_SIZE);
AcquireGoldenOutputs(GoldenData, LAYER2_NEURONS);
printf("Set number of steps: ");
scanf("%d", &max_steps);
if (DEBUG==6) {
printf("\nThe input was:\n");
for (int i=0; i<INPUT_SIZE; i++) 
printf("%.10lf\n", DataIn[i]);
printf("\n");
}
ActivateNeuralNetwork(DataIn); 
do {
steps++;
TrainNeuralNetwork(DataIn, GoldenData);
ActivateNeuralNetwork(DataIn);
} while (steps<max_steps);
printf("\nSteps completed: %d", steps);
printf("\nThe final output compared to the golden output is:\n");
for (int i=0; i<LAYER2_NEURONS; i++) 
printf("  Golden: %.13lf  <o>  Real: %.13lf  \n", GoldenData[i], OL2[i]);
printf("\n");
printf("The final Mean Square Error is: %.15lf\n", MeanSquareError(OL2, GoldenData, LAYER2_NEURONS));
free(DataIn);
free(GoldenData);
return 0;
}
