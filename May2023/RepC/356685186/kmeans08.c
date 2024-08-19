#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N  100000
#define Nv 1000
#define Nc 100
#define THRESHOLD 0.000001
float Vectors[N][Nv]; 
float Centers[Nc][Nv]; 
int   Class_of_Vec[N]; 
void printVectors(void) {
int i, j;
float *Vec = &Vectors[0][0];
for (i = 0; i < N; i++) {
printf("--------------------\n");
printf(" Vector #%d is:\n", i);
for (j = 0; j < Nv; j++)
printf("  %f\n", *Vec++);
}
}
void printCenters(void) {
int i, j;
float *Cent = &Centers[0][0];
for (i = 0; i < Nc; i++) {
printf("--------------------\n");
printf(" Center #%d is:\n", i);
for (j = 0; j < Nv; j++)
printf("  %f\n", *Cent++);
}
}
void printClasses(void) {
int i, j;
int *Cll = &Class_of_Vec[0];
for (i = 0; i < N; i++) {
printf("--------------------\n");
printf(" Class of #%d is:\n", i);
printf("  %d\n", *Cll++);
}
}
int notIn(int Num, int Vec[Nc], int max_index){
int j;
for (j=0; j<max_index; j++)
if (*Vec++ == Num) return 0;
return 1;
}
void initCenters( void ) {
float *Cent = &Centers[0][0], *Vec = &Vectors[0][0];
for (int i=0; i<Nc*Nv; i++) *Cent++ = *Vec++;
}
float estimateClasses(void) {
float tot_min_distances = 0;
int *CofVec = &Class_of_Vec[0];
float *Vec, *Cent;
for (int w=0; w<N; w++) {
float min_dist = 0;
int temp_class = 0;
Vec  = &Vectors[w][0];
Cent = &Centers[0][0];
for (int j=0; j<Nv; j++) {
min_dist += ((*Vec) - (*Cent)) * ((*Vec++) - (*Cent++)); 
}
for (int i=1; i<Nc; i++) {
float dist = 0;
Vec  = &Vectors[w][0];
Cent = &Centers[i][0];
for (int j=0; j<Nv; j++) {
dist += ((*Vec) - (*Cent)) * ((*Vec++) - (*Cent++)); 
}
if (dist < min_dist) {
temp_class = i;
min_dist = dist;
}
}
*CofVec++ = temp_class;
tot_min_distances += sqrt(min_dist);
}
return tot_min_distances;
}
void estimateCenters( void ) {
int Centers_matchings[Nc] = {0};    
int i, j, w;
float *Cent = &Centers[0][0], *Vec = &Vectors[0][0];
for (i = 0; i < Nc*Nv; i++) {
*Cent++ = 0;
}
for (w = 0; w < N; w ++) {
Centers_matchings[Class_of_Vec[w]] ++;
for (j = 0; j<Nv; j++) {
Centers[Class_of_Vec[w]][j] += Vectors[w][j];
}
}
for (i = 0; i < Nc; i++) {
if (Centers_matchings[i] != 0) {
Cent = &Centers[i][0];
for (j = 0; j < Nv; j++)
(*Cent++) /= Centers_matchings[i];
}
else
printf("\nERROR: CENTER %d HAS NO NEIGHBOURS...\n", i);
}
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
initCenters() ;
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
} while( diff > THRESHOLD ) ;
printf("\nProcess finished!\n");
printf("\nTotal repetitions were: %d\n", repetitions);
return 0 ;
}
