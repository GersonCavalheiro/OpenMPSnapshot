#pragma once
using namespace std; 
#include <iostream> 
#include <stdlib.h> 
#include <stdio.h> 
#include <omp.h> 


void printDescriptorListC(float* m, int matrixWidth, int matrixHeight) {
printf("\n");
for (unsigned i = 0; i < matrixHeight; i++) {
for (unsigned j = 0; j < matrixWidth; j++) {
}
}
}

void initDescriptorFloatsTwodigitsC(float* v, int vec_size) {

for (int i = 0; i < vec_size; i++) {
v[i] = (float)(rand() % 100) / 10;
}
}

void printDescriptorC(float* v, int n_elems) {
for (int i = 0; i < n_elems; i++) {
}
}



float euclDistBetweenTwoPointsC(float point1, float point2) {

int ax = floor(point1);
int ay = (int)(point1 * 10) % 10;

int bx = floor(point2);
int by = (int)(point2 * 10) % 10;

float euclideanDist = (float)sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));


return euclideanDist;
}

void getEuclideanDistancesVectorC(float* descriptor, float* descriptor_list, float* distances, int n_elems_descriptor, int n_descriptors) {

float sum;
for (int i = 0; i < n_descriptors; i++)
{
sum = 0.0;
for (int j = 0; j < n_elems_descriptor; j++)
{
float p1 = descriptor_list[j + i * n_elems_descriptor];
float p2 = descriptor[j];
sum += euclDistBetweenTwoPointsC(p1,p2);
}
distances[i] = sum;
}
}


void docSearchC(int n_elements_descriptor, int n_descriptors, int trials) {
printf("docSearch C");

int N_ELEMENTS_VECTOR = n_elements_descriptor;
int N_BYTES_VECTOR = N_ELEMENTS_VECTOR * sizeof(float);

int N_DESCRIPTORS_INDEX = n_descriptors;
int N_BYTES_INDEX = N_ELEMENTS_VECTOR * N_DESCRIPTORS_INDEX * sizeof(float);

int N_ELEM_OUTPUT = n_descriptors;
int N_BYTES_OUTPUT = N_ELEM_OUTPUT * sizeof(float);


float* descriptorToCompare = (float*)malloc(N_BYTES_VECTOR);
float* listOfDescriptors = (float*)malloc(N_BYTES_INDEX);
float* vectorOfDistances = (float*)malloc(N_BYTES_OUTPUT);


initDescriptorFloatsTwodigitsC(descriptorToCompare, N_ELEMENTS_VECTOR); 
initDescriptorFloatsTwodigitsC(listOfDescriptors, N_DESCRIPTORS_INDEX * N_ELEMENTS_VECTOR); 


double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
getEuclideanDistancesVectorC(descriptorToCompare, listOfDescriptors, vectorOfDistances, N_ELEMENTS_VECTOR, N_DESCRIPTORS_INDEX);
}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de docSearchC() con vector de %d elementos e indice de %d descriptores: % lf seconds.\n\n", trials, N_ELEMENTS_VECTOR, N_DESCRIPTORS_INDEX, (t2 - t1) / (float)trials);


printf("descriptor to compare: ");
printDescriptorC(descriptorToCompare, N_ELEMENTS_VECTOR);

printf("descriptor list: ");
printDescriptorListC(listOfDescriptors, N_ELEMENTS_VECTOR, N_DESCRIPTORS_INDEX);

printf("vector of euclidean distances: ");
printDescriptorC(vectorOfDistances, N_DESCRIPTORS_INDEX);

float minDist = vectorOfDistances[0];
int minVecPos = 0;
for (int i = 0; i < N_DESCRIPTORS_INDEX; i++)
{
if (vectorOfDistances[i] < minDist) minDist = vectorOfDistances[i];
minVecPos = i;
}
printf("Vector from data with min euclidean distance: vector %d from data with distance %f\n", minVecPos, minDist);


free(descriptorToCompare);
free(listOfDescriptors);
free(vectorOfDistances);
}
