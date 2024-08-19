#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "correlation.h"
static size_t getIndex(const size_t x,const size_t y,const size_t n) {
size_t k = ( n * ( n - 1 ) / 2 ) - ( ( n - x ) * ( n - x - 1 ) / 2 ) + y - x - 1;
return k;
}
extern float* getCorrelation(const float* vectors,const size_t cols,const size_t rows) {
float preSum[rows]; 
float preS23[rows]; 
float fCols= cols;
fprintf(stderr,"start\n");
fprintf(stderr,"precomputing sums\n");
#pragma omp parallel for
for(size_t i=0;i<rows;i++) { 
size_t base=i*cols;
float sum=0.0;
float sum2=0.0;
for(size_t j=0;j<cols;j++) {
float x=vectors[base+j];
sum += x;
sum2 += x*x;
}
preSum[i]=sum;
preS23[i]=(sum2*fCols) - (sum*sum);
}
fprintf(stderr, "done\n");
fprintf(stderr, "calculate correlation values\n");
size_t totalSize=(rows)*(rows-1)/2; 
fprintf(stderr,"%zu (%zu)\n",totalSize,totalSize*sizeof(float));
float *output=malloc(sizeof(float)*totalSize);
assert(output!=NULL);
if(output==NULL) {
fprintf(stderr,"not enough memory\n");
abort();
}
#pragma omp parallel for
for(size_t i=0;i<rows;i++) {
size_t baseX=i*cols;
float sumX = preSum[i];
float s2=preS23[i];
#pragma omp parallel for
for(size_t j=i+1;j<rows;j++) {
size_t baseY=j*cols;
float sumXY = 0.0;
for(size_t k=0;k<cols;k++) {
float a=vectors[baseX+k];
float b=vectors[baseY+k];
sumXY += a*b;
}
float sumY = preSum[j];
float s3 = preS23[j];
float s1 = (sumXY*fCols) - (sumX*sumY);
float s4 = sqrtf(s2*s3);
float correlation = s1/s4;
size_t index=getIndex(i,j,rows);
output[index] = correlation;
}
}
fprintf(stderr,"\nend\n");
return(output);
}
