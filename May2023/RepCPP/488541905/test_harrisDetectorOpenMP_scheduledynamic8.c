

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <sys/time.h>

#include <helper_cuda.h>
#include <helper_image.h>
#include <omp.h>

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define MAX_BRIGHTNESS 255

typedef int pixel_t;


void harrisDetectorHost(const pixel_t *h_idata, const int w, const int h, 
const int ws,               
const int threshold,        
pixel_t * reference)
{
int i,j,k,l;  
int Ix, Iy;   
int R;        
int sumIx2, sumIy2, sumIxIy;

for(i=0; i<h; i++) 
{
for(j=0; j<w; j++) 
{
reference[i*w+j]=h_idata[i*w+j]/4; 
}
}

for(i=ws+1; i<h-ws-1; i++) 
{
for(j=ws+1; j<w-ws-1; j++) 
{
sumIx2=0;sumIy2=0;sumIxIy=0;
for(k=-ws; k<=ws; k++) 
{
for(l=-ws; l<=ws; l++) 
{
Ix = ((int)h_idata[(i+k-1)*w + j+l] - (int)h_idata[(i+k+1)*w + j+l])/32;         
Iy = ((int)h_idata[(i+k)*w + j+l-1] - (int)h_idata[(i+k)*w + j+l+1])/32;         
sumIx2 += Ix*Ix;
sumIy2 += Iy*Iy;
sumIxIy += Ix*Iy;
}
}

R = sumIx2*sumIy2-sumIxIy*sumIxIy-0.05*(sumIx2+sumIy2)*(sumIx2+sumIy2);
if(R > threshold) {
reference[i*w+j]=MAX_BRIGHTNESS; 
}
}
}
}   

void harrisDetectorOpenMP(const pixel_t *h_idata, const int w, const int h, 
const int ws, const int threshold, 
pixel_t * h_odata)
{
int n_threads = 4;

omp_set_num_threads(n_threads);

int i,j,k,l;  
int Ix, Iy;   
int R;        
int sumIx2, sumIy2, sumIxIy;

#pragma omp parallel for shared(h_odata, h_idata) private(i, j) firstprivate(w, h) schedule(dynamic,8)
for(i=0; i<h; i++) 
{
for(j=0; j<w; j++) 
{   
h_odata[i*w+j]=h_idata[i*w+j]/4; 
}
}

#pragma omp parallel for shared(h_odata, h_idata) private(j, sumIx2, sumIy2, sumIxIy, Ix, Iy) firstprivate(w, h, ws, threshold) schedule(dynamic,8)
for(i=ws+1; i<h-ws-1; i++) 
{
for(j=ws+1; j<w-ws-1; j++) 
{
sumIx2=0;sumIy2=0;sumIxIy=0;
for(k=-ws; k<=ws; k++) 
{
for(l=-ws; l<=ws; l++) 
{
Ix = ((int)h_idata[(i+k-1)*w + j+l] - (int)h_idata[(i+k+1)*w + j+l])/32;         
Iy = ((int)h_idata[(i+k)*w + j+l-1] - (int)h_idata[(i+k)*w + j+l+1])/32;         
sumIx2 += Ix*Ix;
sumIy2 += Iy*Iy;
sumIxIy += Ix*Iy;
}
}

R = sumIx2*sumIy2-sumIxIy*sumIxIy-0.05*(sumIx2+sumIy2)*(sumIx2+sumIy2);
if(R > threshold) {
h_odata[i*w+j]=MAX_BRIGHTNESS; 
}
}
}

}

void usage(char *command) 
{
printf("Usage: %s [-h] [-i inputfile] [-o outputfile] [-r referenceFile] [-w windowsize] [-t threshold]\n",command);
}

int main( int argc, char** argv) 
{

int deviceId = 0;
char *fileIn        = (char *)"../chessBig.pgm",
*fileOut       = (char *)"../resultChessBigOpenMP.pgm",
*referenceOut  = (char *)"../referenceChessBigOpenMP.pgm";
unsigned int ws = 1, threshold = 500;

int opt;
while( (opt = getopt(argc,argv,"i:o:r:w:t:h")) !=-1)
{
switch(opt)
{

case 'i':
if(strlen(optarg)==0)
{
usage(argv[0]);
exit(1);
}

fileIn = strdup(optarg);
break;
case 'o':
if(strlen(optarg)==0)
{
usage(argv[0]);
exit(1);
}
fileOut = strdup(optarg);
break;
case 'r':
if(strlen(optarg)==0)
{
usage(argv[0]);
exit(1);
}
referenceOut = strdup(optarg);
break;
case 'w':
if(strlen(optarg)==0 || sscanf(optarg,"%d",&ws)!=1)
{
usage(argv[0]);
exit(1);
}
break;
case 't':
if(strlen(optarg)==0 || sscanf(optarg,"%d",&threshold)!=1)
{
usage(argv[0]);
exit(1);
}
break;
case 'h':
usage(argv[0]);
exit(0);
break;

}
}

pixel_t * h_idata=NULL;
unsigned int h,w;

if (sdkLoadPGM<pixel_t>(fileIn, &h_idata, &w, &h) != true) {
printf("Failed to load image file: %s\n", fileIn);
exit(1);
}

pixel_t * h_odata   = (pixel_t *) malloc( h*w*sizeof(pixel_t));
pixel_t * reference = (pixel_t *) malloc( h*w*sizeof(pixel_t));

struct timeval start, end;
gettimeofday(&start, NULL);

harrisDetectorHost(h_idata, w, h, ws, threshold, reference);   

gettimeofday(&end, NULL);

struct timeval startMP, endMP;
gettimeofday(&startMP, NULL);

harrisDetectorOpenMP(h_idata, w, h, ws, threshold, h_odata);   

gettimeofday(&endMP, NULL);

printf( "Host processing time: %f (ms)\n", (end.tv_sec-start.tv_sec)*1000.0 + ((double)(end.tv_usec - start.tv_usec))/1000.0);
printf( "OpenMP processing time: %f (ms)\n", (endMP.tv_sec-startMP.tv_sec)*1000.0 + ((double)(endMP.tv_usec - startMP.tv_usec))/1000.0);

if (sdkSavePGM<pixel_t>(referenceOut, reference, w, h) != true) {
printf("Failed to save image file: %s\n", referenceOut);
exit(1);
}
if (sdkSavePGM<pixel_t>(fileOut, h_odata, w, h) != true) {
printf("Failed to save image file: %s\n", fileOut);
exit(1);
}

free( h_idata );
free( h_odata );
free( reference );
}
