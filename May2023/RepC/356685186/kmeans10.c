#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N  100000
#define Nv 1000
#define Nc 100
#define THRESHOLD 0.000001
#define MAX_REPETITIONS 16
float Vectors[N][Nv]; 
float Centers[Nc][Nv]; 
int   Class_of_Vec[N]; 
void printVectors(void) {
int i, j;
printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
for (i = 0; i < N; i++) {
printf("--------------------\n");
printf(" Vector #%d is:\n", i);
for (j = 0; j < Nv; j++)
printf("  %f\n", Vectors[i][j]);
}
printf("--------------------\n");
printf("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n");
}
void printCenters(void) {
int i, j;
printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
for (i = 0; i < Nc; i++) {
printf("--------------------\n");
printf(" Center #%d is:\n", i);
for (j = 0; j < Nv; j++)
printf("  %f\n", Centers[i][j]);
}
printf("--------------------\n");
printf("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n");
}
void printClasses(void) {
int i, j;
printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
for (i = 0; i < N; i++) {
printf("--------------------\n");
printf(" Class of Vector #%d is:\n", i);
printf("  %d\n", Class_of_Vec[i]);
}
printf("--------------------\n");
printf("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n");
}
int notVectorInCenters(float Vec[Nv], int maxIndex) {
int flag;
for (int c=0; c<maxIndex; c++) {
flag = 1; 
for (int i=0; i<Nv; i++) {
if (Vec[i] != Centers[c][i]) {
flag=0;
break;
}
}
if (flag)     
return 0; 
}
return 1;
}
void pickSubstituteCenter(int indexOfCenterToChange){
int currentVec = 0;
do {
if (notVectorInCenters(Vectors[currentVec], Nc)) {
for (int i=0; i<Nv; i++) 
Centers[indexOfCenterToChange][i] = Vectors[currentVec][i];  
return;    
}
currentVec ++; 
} while (currentVec<N);
printf("\n");
return;
}
void initCenters2() {
int currentCenter=0, currentVec=0;
do {
if (notVectorInCenters(Vectors[currentVec], currentCenter)) {
for (int i=0; i<Nv; i++) 
Centers[currentCenter][i] = Vectors[currentVec][i];
currentCenter ++;                
}
currentVec++;
} while (currentCenter<Nc);
}
float estimateClasses() {
float min_dist, dist, tot_min_distances = 0;
int temp_class;
int i, j, w;
for (w=0; w<N; w++) {
min_dist = 1e6;
temp_class = -1;
for (i=0; i<Nc; i++) {
dist = 0;
for (j=0; j<Nv; j++) 
dist += (Vectors[w][j]-Centers[i][j]) * (Vectors[w][j]-Centers[i][j]); 
if (dist < min_dist) {
temp_class = i;
min_dist = dist;
}
}
Class_of_Vec[w] = temp_class; 
tot_min_distances += sqrt(min_dist); 
}
return tot_min_distances;
}
void estimateCenters() {
int Centers_matchings[Nc] = {0};    
int i, j, w;
int needToRecalculateCenters = 0;
for (i = 0; i < Nc; i++)
for (j = 0; j < Nv; j++)
Centers[i][j] = 0;
for (w = 0; w < N; w ++) {
Centers_matchings[Class_of_Vec[w]] ++;
for (j = 0; j<Nv; j++)
Centers[Class_of_Vec[w]][j] += Vectors[w][j];
}
for (i = 0; i < Nc; i++) {
if (Centers_matchings[i] != 0)
for (j = 0; j < Nv; j++)
Centers[i][j] /= Centers_matchings[i];
else {
printf("\nWARNING: Center %d has no members.\n", i);
pickSubstituteCenter(i);
needToRecalculateCenters = 1;
break;
}
}
if (needToRecalculateCenters == 1) estimateCenters();
}
void SetVec( void ) {
int i, j;
float *Vec = &Vectors[0][0];
for( i = 0 ; i< N*Nv ; i++ )
*Vec++ =  (1.0*rand())/RAND_MAX ;
}
int main( int argc, const char* argv[] ) {
int repetitions = 0;
float totDist, prevDist, diff;
printf("--------------------------------------------------------------------------------------------------\n");
printf("This program executes the K-Means algorithm for random vectors of arbitrary number and dimensions.\n");
printf("Current configuration has %d Vectors, %d Classes and %d Elements per vector.\n", N, Nc, Nv);
printf("--------------------------------------------------------------------------------------------------\n");
printf("Now initializing vectors...\n");
SetVec() ;
printf("Now initializing centers...\n");
initCenters2() ;
totDist = 1.0e30;
printf("Now running the main algorithm...\n\n");
do {
repetitions++; 
prevDist = totDist ;
totDist = estimateClasses() ;
estimateCenters() ;
diff = (prevDist-totDist)/totDist ;
printf(">> REPETITION: %3d  ||  ", repetitions);
printf("DISTANCE IMPROVEMENT: %.8f \n", diff);
} while( (diff > THRESHOLD) && (repetitions < MAX_REPETITIONS) ) ;
printf("\n\nProcess finished!\n");
printf("Total repetitions were: %d\n", repetitions);
return 0 ;
}
