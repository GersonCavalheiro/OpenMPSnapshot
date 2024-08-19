#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
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
for (int c=0; c<maxIndex; c++) {
int flag = 1; 
for (int i=0; i<Nv; i++) {
if (Vec[i] != Centers[c][i]) {
flag = 0;
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
printf("> Now searching for a substitute center...\n");
do {
printf(">> Now examining vec:%d\n", currentVec);
if (notVectorInCenters(Vectors[currentVec], Nc)) {
printf(">>> Current vec is not in existing centers\n");
for (int i=0; i<Nv; i++) 
Centers[indexOfCenterToChange][i] = Vectors[currentVec][i];  
printf(">>> Substituted old center with current vector\n");
return;    
}
printf(">>> WARNING: If the center was substituted, this line must not be present\n");
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
float tot_min_distances = 0;
for (int w=0; w<N; w++) {
float min_dist = 1e30;
int temp_class = -1;
for (int i=0; i<Nc; i++) {
float dist = 0;
for (int j=0; j<Nv; j++) 
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
int needToRecalculateCenters = 0;
for (int i = 0; i < Nc; i++)
for (int j = 0; j < Nv; j++)
Centers[i][j] = 0;
for (int w = 0; w < N; w ++) {
Centers_matchings[Class_of_Vec[w]] ++;
for (int j = 0; j<Nv; j++)
Centers[Class_of_Vec[w]][j] += Vectors[w][j];
}
for (int i = 0; i < Nc; i++) {
if (Centers_matchings[i] != 0)
for (int j = 0; j < Nv; j++)
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
for(int i = 0 ; i< N ; i++ )
for(int j = 0 ; j< Nv ; j++ )
Vectors[i][j] =  (1.0*rand())/RAND_MAX ;
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
