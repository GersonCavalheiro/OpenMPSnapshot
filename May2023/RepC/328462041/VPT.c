#include "VPT.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "supplementary.h"

struct VPtree *buildVPT_recursively(double *X, int n, int d, int *x_index);
double *calculate_distances(double * X , int n, int d);

struct VPtree *createVPT(double *X, int n, int d, int index_offset){

int *x_indx = (int *)malloc(n*sizeof(int));
if (x_indx == NULL){
exit(1);
}

for(int i=0;i<n;i++){
x_indx[i] = i + index_offset;
}
vptree *T = buildVPT_recursively(X, n, d, x_indx);
free(x_indx);
return T;
}

struct VPtree *buildVPT_recursively(double *X, int n, int d, int *x_index){

double *dist;
double *distances=(double *)malloc(n*sizeof(double));
if (distances == NULL){
exit(1);
}

vptree *T = NULL;
int innerSize= 0;
int outerSize = 0;
double *innerX = (double *)malloc(innerSize*d*sizeof(double));
if (innerX == NULL){
exit(1);
}
int *innerID = (int *)malloc(innerSize*sizeof(int));
if (innerID == NULL){
exit(1);
}
double *outerX = (double *)malloc(outerSize*d*sizeof(double));
if (outerX == NULL){
exit(1);
}
int *outerID = (int *)malloc(outerSize*sizeof(int));
if (outerID == NULL){
exit(1);
}

if(n == 0){
free(innerX);
free(outerX);
free(innerID);
free(outerID);
free(dist);
free(distances);
return T;
}

T = (struct VPtree *)calloc(1, sizeof(struct VPtree));
T->vp = (double *)malloc(d*sizeof(double));
T->idx = x_index[n-1];
for (int i=0; i<d; i++){
T->vp[i] = X[(n-1)*d + i];
}

if(n == 1){
free(innerX);
free(outerX);
free(innerID);
free(outerID);
free(dist);
free(distances);
return T;
}

dist = calculate_distances(X, n, d);
for(int i=0; i<n; i++){
distances[i]=dist[i];
}

if ((n-1)%2 != 0){
T->md = quickselect(dist, 0, n-1, n/2);
}
else{
T->md = (quickselect(dist, 0, n-1, n/2+1) + quickselect(dist,0,  n-1, n/2)) / 2;
}

for(int i=0 ; i<n-1 ; i++){
if(distances[i] <= T->md){
innerSize++;
innerX = realloc(innerX, innerSize*d*sizeof(double));
innerID = realloc(innerID, innerSize*sizeof(int));
innerX[innerSize*d - 1] = i;
innerID[innerSize-1] = x_index[i];
}
else{
outerSize++;
outerX = realloc(outerX, outerSize*d*sizeof(double));
outerID = realloc(outerID, outerSize*sizeof(int));
outerX[outerSize*d - 1] = i;
outerID[outerSize-1] = x_index[i];
}
}

for(int i=0 ; i<innerSize; i++){
for(int j=0; j<d; j++)
innerX[i*d+j]=  X[(int)(innerX[i*d + d -1])*d+j];
}

for(int i=0 ; i<outerSize; i++){
for(int j=0; j<d; j++)
outerX[i*d+j]=X[(int)(outerX[i*d + d - 1])*d+j];
}

#pragma omp task
T->inner = buildVPT_recursively(innerX, innerSize, d, innerID);
T->outer = buildVPT_recursively(outerX, outerSize, d, outerID);
#pragma omp taskwait
free(innerX); free(outerX);
free(innerID); free(outerID);
free(distances); free(dist);

return T;
}

double *calculate_distances(double * X , int n, int d){

double *distances = (double *)calloc(n, sizeof(double));
if (distances == NULL){
exit(1);
}
for(int point_iter=0; point_iter<(n-1); point_iter++){
for (int coord_iter=0; coord_iter<d; coord_iter++){
double coord_diff = X[point_iter*d + coord_iter] - X[(n-1)*d + coord_iter];
distances[point_iter] += coord_diff*coord_diff;
}
distances[point_iter] = sqrt(fabs(distances[point_iter]));
}

return distances;
}

void destroy(struct VPtree *T){
if(T==NULL){
return;
}
destroy(T->inner);
destroy(T->outer);
free(T->vp);
free(T);
}


struct VPtree *getInner(struct VPtree * T){
return T->inner;
}

struct VPtree *getOuter(struct VPtree * T){
return T->outer;
}

double getMD(struct VPtree * T){
return T->md;
}

double *getVP(struct VPtree *T){
return T->vp;
}

int getIDX(struct VPtree *T){
return T->idx;
}

