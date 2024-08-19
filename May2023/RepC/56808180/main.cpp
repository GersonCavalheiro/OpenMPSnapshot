#include <stdio.h>
#include <fstream>
#include <limits.h>
#include <omp.h>
#include "../../Libraries/Timer.h"
#define FLT_MIN 0.000000000001
#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define ALLOC_ALIGN 64
#pragma offload_attribute (push, target(mic))
void MergeArray(int array[], int startLeftIndex, int endLeftIndex, int startRightIndex, int endRightIndex)
{
int leftArraySize = endLeftIndex - startLeftIndex + 1;
int rightArraySize = endRightIndex - startRightIndex + 1;
int* leftArray = (int*)_mm_malloc(sizeof(int) * leftArraySize, ALLOC_ALIGN);
int* rightArray = (int*)_mm_malloc(sizeof(int) * rightArraySize, ALLOC_ALIGN);
for (int i = 0; i < leftArraySize - 1; ++i)
{
leftArray[i] = array[startLeftIndex + i];
}
leftArray[leftArraySize - 1] = INT_MAX;
for (int i = 0; i < rightArraySize - 1; ++i)
{
rightArray[i] = array[startRightIndex + i];
}
rightArray[rightArraySize - 1] = INT_MAX;
int k = startLeftIndex, i = 0, j = 0;
for (; i < leftArraySize - 1 && j < rightArraySize - 1; ++k)
{
array[k] = leftArray[i] <= rightArray[j] ? leftArray[i++] : array[k] = rightArray[j++];
}
for (; i < leftArraySize - 1; ++k)
{
array[k] = leftArray[i++];
}
for (; j < rightArraySize - 1; ++k)
{
array[k] = rightArray[j++];
}
_mm_free(leftArray);
_mm_free(rightArray);
}
#pragma offload_attribute (pop)
#pragma offload_attribute (push, target(mic))
void MergeSort(int array[], int length)
{
if (length < 2)
{
return;
}
int step = 1;
while (step < length)
{
int startLeftIndex = 0;
int startRightIndex = step;
while (startRightIndex + step <= length)
{
MergeArray(array, startLeftIndex, startLeftIndex + step, startRightIndex, startRightIndex + step);
startLeftIndex = startRightIndex + step;
startRightIndex = startLeftIndex + step;
}
if (startRightIndex < length)
{
MergeArray(array, startLeftIndex, startLeftIndex + step, startRightIndex, length);
}
step *= 2;
}
}
#pragma offload_attribute (pop)
void MergeOnMic(int array[], int arrayCounter, int arraySize, int numOfTransferedArrays, int numberOfArrays)
{
#pragma offload target(mic:0) in(array[arrayCounter*arraySize:arraySize*numOfTransferedArrays] : into(array[0:arraySize*numOfTransferedArrays]) REUSE) out(array[0:arraySize*numOfTransferedArrays] : into(array[arrayCounter*arraySize:arraySize*numOfTransferedArrays]) REUSE)
{
#pragma omp parallel for
for(int i=0; i<numOfTransferedArrays; i++)
{
MergeSort(array + (i*arraySize), arraySize);
}
}
}
void GenerateArray(int array[], int size)
{
for (int i = size - 1; i >= 0; --i)
{
array[i] = rand();
}
}
int main(int argc, char** argv) {
Timer mainTimer(Timer::Mode::Single);
Timer allCalculation(Timer::Mode::Single);
Timer medianTimer(Timer::Mode::Median);
allCalculation.Start();
int arraySize, numberOfArrays, arrayCounter;
int numOfTransferedArrays;
if (argc != 4) {
printf("%d", argc);
printf("Wrong number of arguments, correct number is: 1- sizeOfArray 2- numberOfArrays\n");
return 0;
}
else {
arraySize = atoi(argv[1]);
numberOfArrays = atoi(argv[2]);
numOfTransferedArrays = atoi(argv[3]);
}
int* array = (int*)_mm_malloc(sizeof(int*) * numberOfArrays * arraySize, ALLOC_ALIGN);
for(arrayCounter = 0; arrayCounter < numberOfArrays; ++arrayCounter)
{
GenerateArray(array + (arrayCounter * arraySize), arraySize);
}
#pragma offload_transfer target(mic) in(array[0:0]:alloc(array[0:arraySize * numOfTransferedArrays]) ALLOC)
for(arrayCounter = 0; arrayCounter < numberOfArrays; arrayCounter = arrayCounter + numOfTransferedArrays)
{
mainTimer.Start();
MergeOnMic(array, arrayCounter, arraySize, numOfTransferedArrays, numberOfArrays);
mainTimer.Stop();
medianTimer.PushTime(mainTimer.Get());
}
#pragma offload_transfer target(mic) in(array[0:0]:alloc(array[0:arraySize * numOfTransferedArrays]) FREE)
_mm_free(array);
allCalculation.Stop();
printf("OpenMP,%d,%d,%lu,%lu,%lu,%lu,", numOfTransferedArrays, arraySize, medianTimer.Get(), medianTimer.GetAvg(), allCalculation.Get(), allCalculation.Get());
return 0;
}
