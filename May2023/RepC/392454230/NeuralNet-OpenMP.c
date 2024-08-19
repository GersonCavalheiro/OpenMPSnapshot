#define NL1 100            
#define NL2 10             
#define NINPUT 784         
#define NTRAIN 60000       
#define NTEST 10000        
#define ITERATIONS 500     
#define ALPHA (double)0.05 
#include "extra_functions.c"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
double WL1[NL1][NINPUT + 1];
double WL2[NL2][NL1 + 1];
double DL1[NL1];
double DL2[NL2];
double OL1[NL1];
double OL2[NL2];
double delta2[NL2];
double delta1[NL1];
double data_train[NTRAIN][NINPUT];
double data_test[NTEST][NINPUT];
int class_train[NTRAIN];
int class_test[NTEST];
double input[NINPUT];
void activateNN(double *in){
#pragma omp parallel for
for (int i = 0; i < NL1; i++)
{
double register sum = 0;
for (int j = 0; j < NINPUT; j++)
{
sum += WL1[i][j] * in[j];
}
sum += WL1[i][NINPUT]; 
DL1[i] = sum;
OL1[i] = logistic(sum);
}
#pragma omp parallel for
for (int i = 0; i < NL2; i++)
{
double register sum = 0;
for (int j = 0; j < NL1; j++)
{
sum += WL2[i][j] * OL1[j];
}
sum += WL2[i][NL1]; 
DL2[i] = sum;
OL2[i] = logistic(sum);
}
}
void trainNN(double *in,double *desired){
activateNN(in);
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < NL2; i++)
{
delta2[i] = (OL2[i]-desired[i])*OL2[i]*(1-OL2[i]);
}
#pragma omp for
for (int i = 0; i < NL1; i++)
{
double register sum = 0;
for (int j = 0; j < NL2; j++)
{
sum += WL2[j][i] * delta2[j];
}
double register Oi = OL1[i];
delta1[i] = sum * Oi * (1-Oi);
}
}
#pragma omp parallel
{
#pragma omp for nowait
for (int i = 0; i < NL2; i++)
{
for (int j = 0; j < NL1; j++)
{
WL2[i][j] -= ALPHA * OL1[j] * delta2[i];
}
WL2[i][NL1] -= ALPHA * delta2[i];
}
#pragma omp for
for (int i = 0; i < NL1; i++)
{
for (int j = 0; j < NINPUT; j++)
{
WL1[i][j] -= ALPHA * in[j] * delta1[i];
}
WL1[i][NINPUT] -= ALPHA * delta1[i];
}
}
}
void evaluate(int inputClass,double confMatrix[NL2][NL2]){
int maxIndex = 0;
double maxVal = 0;
for (int i = 0; i < NL2; i++)
{
if (maxVal<OL2[i])
{
maxVal = OL2[i];
maxIndex = i;
}
}
if (maxVal==OL2[inputClass])
{
maxIndex = inputClass;
}
confMatrix[maxIndex][inputClass]++;
}
void normalizeData(double in[][NINPUT],int inSize){
double average[NINPUT] = {0};
double var[NINPUT] = {0};
#pragma omp parallel for
for (int i = 0; i < NINPUT; i++)
{
for (int j = 0; j < inSize; j++)
{
average[i] += in[j][i];
}
average[i] /= inSize;
double register mean = average[i];
for (int j = 0; j < inSize; j++)
{
var[i] += (in[j][i] - mean)*(in[j][i] - mean);
}
var[i] /= inSize-1;
}
#pragma omp parallel for
for (int i = 0; i < NINPUT; i++)
{
double register mean = average[i];
double register stddev = sqrt(var[i]);
for (int j = 0; j < inSize; j++)
{
in[j][i] -= mean;
in[j][i] /= stddev;
}
}
}
int main() {
double desiredOut[NL2]={0};
double confusionMatrixTrain[NL2][NL2]= {0};
double confusionMatrixTest[NL2][NL2]= {0};
readfile("./DATA/fashion-mnist_train.csv",class_train,data_train,NTRAIN);
readfile("./DATA/fashion-mnist_test.csv",class_test,data_test,NTEST);
normalizeData(data_test,NTEST);
normalizeData(data_train,NTRAIN);
initVecs();
for (int i = 0; i < NL2; i++)
{
desiredOut[i] = 0.1;
}
int register tmp = 0;
for (int i = 0; i < NTRAIN*ITERATIONS; i++)
{
tmp = rand()%NTRAIN;
desiredOut[class_train[tmp]] = 0.9;
trainNN(data_train[tmp],desiredOut);
desiredOut[class_train[tmp]] = 0.1;
}
printf("TRAINING FINISHED!\n\n");
for (int i = 0; i < NTRAIN; i++)
{
activateNN(data_train[i]);
evaluate(class_train[i],confusionMatrixTrain);
}
for (int i = 0; i < NTEST; i++)
{
activateNN(data_test[i]);
evaluate(class_test[i],confusionMatrixTest);
}
double register testCorrect = 0;
double register trainCorrect = 0;
for (int i = 0; i < NL2; i++)
{
testCorrect += confusionMatrixTest[i][i];
trainCorrect += confusionMatrixTrain[i][i];
}
double totalCorrect = testCorrect + trainCorrect;
testCorrect /= (double)NTEST;
trainCorrect /= (double)NTRAIN;
totalCorrect /= ((double)NTEST+(double)NTRAIN);
printf("TRAINING SAMPLES CONFUSION MATRIX:\n");
printTable(confusionMatrixTrain);
printf("TESTING SAMPLES CONFUSION MATRIX:\n");
printTable(confusionMatrixTest);
printf("Correct rate in training samples: %0.3f\n",trainCorrect);
printf("Correct rate in testing samples: %0.3f\n",testCorrect);
printf("Overall hit rate: %0.3f\n",totalCorrect);
printf("Learning rate = %0.4f\n",ALPHA);
printf("EPOCHS = %d\n",(int)ITERATIONS);
return 0;
}