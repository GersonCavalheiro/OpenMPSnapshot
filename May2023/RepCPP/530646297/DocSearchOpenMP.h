#pragma once
using namespace std; 
#include <iostream> 
#include <stdlib.h> 
#include <stdio.h> 
#include <omp.h> 



void printDescriptorList(float* m, int matrixWidth, int matrixHeight) {

for (unsigned i = 0; i < matrixHeight; i++) {
for (unsigned j = 0; j < matrixWidth; j++) {
}
}
}

void initDescriptorFloatsTwodigits(float* v, int vec_size) {

for (int i = 0; i < vec_size; i++) {
v[i] = (float)(rand() % 100) / 10;
}
}

void printDescriptor(float* v, int n_elems) {

for (int i = 0; i < n_elems; i++) {
}
}




float euclDistBetweenTwoPoints(float point1, float point2) {

int ax = floor(point1);
int ay = (int)(point1 * 10) % 10;

int bx = floor(point2);
int by = (int)(point2 * 10) % 10;

float euclideanDist = (float)sqrt( (ax - bx)*(ax - bx) + (ay - by)*(ay - by) );


return euclideanDist;
}


void getEuclideanDistancesVector(float* descriptor, float* descriptor_list, float* distances, int n_elems_descriptor, int n_descriptors) {

#pragma omp parallel for
for (int i = 0; i < n_descriptors; i++)
{
float sum = 0.0;
#pragma omp parallel for reduction (+:sum)
for (int j = 0; j < n_elems_descriptor; j++)
{
float euclideanDist = euclDistBetweenTwoPoints(descriptor_list[j + i * n_elems_descriptor], descriptor[j]);
sum += euclideanDist;
}
distances[i] = sum;
}
}


void docSearchOpenMP(int n_elements_descriptor, int n_descriptors, int trials, int n_threads) {
printf("docSearch OpenMP");

int N_ELEMENTS_DESCRIPTOR = n_elements_descriptor;
int N_BYTES_DESCRIPTOR = N_ELEMENTS_DESCRIPTOR * sizeof(float);

int N_DESCRIPTORS = n_descriptors;
int N_BYTES_LIST_OF_DESCRIPTORS = N_ELEMENTS_DESCRIPTOR * N_DESCRIPTORS * sizeof(float);


float* descriptorToCompare = (float*)malloc(N_BYTES_DESCRIPTOR);
float* listOfDescriptors = (float*)malloc(N_BYTES_LIST_OF_DESCRIPTORS);
float* vectorOfDistances = (float*)malloc(N_DESCRIPTORS * sizeof(float)); 

initDescriptorFloatsTwodigits(descriptorToCompare, N_ELEMENTS_DESCRIPTOR); 
initDescriptorFloatsTwodigits(listOfDescriptors, N_ELEMENTS_DESCRIPTOR*N_DESCRIPTORS); 

omp_set_num_threads(n_threads);
double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
getEuclideanDistancesVector(descriptorToCompare, listOfDescriptors, vectorOfDistances, N_ELEMENTS_DESCRIPTOR, N_DESCRIPTORS);
}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de docSearchOpenMP() con %d threads, con vector de %d elementos e indice de %d descriptores: % lf seconds.\n\n", trials, n_threads, N_ELEMENTS_DESCRIPTOR, N_DESCRIPTORS, (t2 - t1) / (float)trials);


printf("descriptor to compare: ");
printDescriptor(descriptorToCompare, N_ELEMENTS_DESCRIPTOR);

printf("descriptor list: ");
printDescriptorList(listOfDescriptors, N_ELEMENTS_DESCRIPTOR, N_DESCRIPTORS);

printf("vector of euclidean distances: ");
printDescriptor(vectorOfDistances, N_DESCRIPTORS);

free(descriptorToCompare);
free(listOfDescriptors);
free(vectorOfDistances);
}