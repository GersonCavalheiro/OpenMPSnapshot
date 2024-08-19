#include <stdio.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <algorithm>
#include <sys/time.h>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define MATCH(s) (!strcmp(argv[ac], (s)))


using std::vector;
using std::unordered_set;

static const double kMicro = 1.0e-6;
double getTime()
{
struct timeval TV;
struct timezone TZ;

const int RC = gettimeofday(&TV, &TZ);
if(RC == -1) {
printf("ERROR: Bad call to gettimeofday\n");
return(-1);
}

return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}

void imageSegmentation(int *labels, unsigned char *data, int width, int height, int pixelWidth, int Threshold)
{
int maxN = std::max(width,height);
int phases = (int) ceil(log(maxN)/log(2)) + 1;

for(int pp = 0; pp <= phases; pp++)
{
bool change = false;
for (int i = height - 1; i >= 0; i--) {
for (int j = width - 1; j >= 0; j--) {

int idx = i*width + j;
int idx3 = idx*pixelWidth;
if (idx3 > width * height * pixelWidth) {
printf("idx3 bigger: %d\n", idx3);
}
if (idx3 < 0) {
printf("idx3 negative: %d\n", idx3);
}
if (idx >= width * height) {
printf("idx bigger: %d\n", idx);
}
if (idx3 < 0) {
printf("idx negative: %d\n", idx);
}
if (labels[idx] == 0)
continue;

int ll = labels[idx]; 


if (j != 0 && abs((int)data[(i*width + j - 1)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[i*width + j - 1]);
change = true;
}

if (j != width-1 && abs((int)data[(i*width + j + 1)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[i*width + j + 1]);
change = true;
}


if(i != 0 && j != 0 && abs((int)data[((i-1)*width + j - 1)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[(i-1) * width + j - 1]);
change = true;
}
if(i != 0 && abs((int)data[((i-1)*width + j)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[(i-1)*width + j]);
change = true;
}

if(i != 0 && j != width-1 && abs((int)data[((i-1)*width + j + 1)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[(i-1) * width + j + 1]);
change = true;
}

if(i != height-1 && j!= 0 && abs((int)data[((i+1)*width + j - 1)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[(i+1) * width + j - 1]);
change = true;
}

if(i != height-1 && abs((int)data[((i+1)*width + j)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[(i+1)*width + j]);
change = true;
}

if(i != height-1 && j != width-1 && abs((int)data[((i+1)*width + j + 1)*pixelWidth] - (int)data[idx3]) < Threshold) {
labels[idx] = std::max(labels[idx], labels[(i+1) * width + j + 1]);
change = true;
}





if (ll < labels[idx]) {
labels[ll - 1] = std::max(labels[idx],labels[ll - 1]);
}
}
}
if (!change) break;
for (int i = 0; i < height; i++) {
for (int j = 0; j < width; j++) {

int idx = i*width + j;

if (labels[idx] != 0) {
labels[idx] = std::max(labels[idx], labels[labels[idx] - 1]);
}
}
}

}

}

int main(int argc,char **argv)
{
int width,height;
int pixelWidth;
int Threshold = 3;
int numThreads = 1;
int seed =1 ;
const char *filename = "input.png";
const char *outputname = "output.png";

if(argc<2)
{
printf("Usage: %s [-i < filename>] [-s <threshold>] [-t <numThreads>] [-o outputfilename]\n",argv[0]);
return(-1);
}
for(int ac=1;ac<argc;ac++)
{
if(MATCH("-s")) {Threshold = atoi(argv[++ac]);}
else if(MATCH("-t")) {numThreads = atoi(argv[++ac]);}
else if(MATCH("-i"))  {filename = argv[++ac];}
else if(MATCH("-o"))  {outputname = argv[++ac];}
else {
printf("Usage: %s [-i < filename>] [-s <threshold>] [-t <numThreads>] [-o outputfilename]\n",argv[0]);
return(-1);
}
}

printf("Reading image...\n");
unsigned char *data = stbi_load(filename, &width, &height, &pixelWidth, 0);
if (!data) {
fprintf(stderr, "Couldn't load image.\n");
return (-1);
}

printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

int *labels = (int *)malloc(sizeof(int)*width*height);
unsigned char *seg_data = (unsigned char *)malloc(sizeof(unsigned char)*width*height*3);

printf("Applying segmentation...\n");

double start_time = getTime();

for(int i = 0; i < height; i++)
{
for(int j = 0; j < width; j++)
{
int idx = (i*width+j);
int idx3 = idx*pixelWidth;

labels[idx] = 0;

if((int)data[idx3] == 0)
continue;
labels[idx] = idx + 1;
}
}
#if defined(_OPENMP)
omp_set_dynamic(0);
#endif

int localDataSize = (width * pixelWidth) * (height / numThreads);
int localLabelSize = width * (height / numThreads);
int remainingSize = width * (height % numThreads);

for(int i=0; i<numThreads; i++)
printf("%d %d\n", localLabelSize, localDataSize);

#pragma omp parallel for num_threads(numThreads)
for(int i=0; i<numThreads + (remainingSize==0?0:1); i++){
if (i < numThreads) { 
for (int j = 0; j < localLabelSize; j++) {
int newLabel = ((int)data[(i*localLabelSize+j)*pixelWidth]) == 0 ? 0 : j+1;
labels[i*localLabelSize+j] = newLabel;
}
imageSegmentation(labels+i*localLabelSize,data+i*localDataSize,width, height/numThreads, pixelWidth, Threshold);
} else { 
for (int j = 0; j < remainingSize; j++) {
int newLabel = ((int)data[(i*localLabelSize+j)*pixelWidth]) == 0 ? 0 : j+1;
labels[i*localLabelSize+j] = newLabel;
}
imageSegmentation(labels+i*localLabelSize,data+i*localDataSize,width, height%numThreads, pixelWidth, Threshold);

}
}

printf("In the memory of \"here\"\n" );
if(numThreads > 1) {


for (int index = 0; index < width*height; index++) {
if (labels[index] == 0) continue;
labels[index] = (index / localLabelSize) * localLabelSize + labels[index];
}

int decrement = (remainingSize == 0 ? localLabelSize : remainingSize);

for (int border = ((width*height)) - decrement;  0 < border; border-=localLabelSize) {
std::unordered_map<int, int> changes;
for(int index = border; index<border+width; index++){

int upIndex = index-width;
int maxVal = labels[index];
int oldUp = -1, oldLeft=-1, oldRight=-1;

if (abs(data[index*pixelWidth] - data[upIndex*pixelWidth]) < Threshold) {
maxVal = std::max(maxVal, labels[upIndex]);
oldUp = labels[upIndex];
if (maxVal != oldUp) {
if (changes.find(oldUp) == changes.end()) {
changes[oldUp] = maxVal;
} else {

int lastUp = oldUp;
while (changes.find(lastUp) != changes.end()) {
lastUp = changes[lastUp];
}
int lastDown = maxVal;
while (changes.find(lastDown) != changes.end()) {
lastDown = changes[lastDown];
}
if (lastUp != lastDown)
changes[std::min(lastUp, lastDown)] = std::max(lastUp, lastDown);

}
}

}
if (upIndex % width != 0 && abs(data[index*pixelWidth] - data[(upIndex-1)*pixelWidth]) < Threshold) {
oldLeft = labels[upIndex-1];
maxVal = std::max(maxVal, labels[upIndex-1]);
if (oldLeft != maxVal) {
int lastUp = oldLeft;
while (changes.find(lastUp) != changes.end()) {
lastUp = changes[lastUp];
}
int lastDown = maxVal;
while (changes.find(lastDown) != changes.end()) {
lastDown = changes[lastDown];
}
if (lastUp != lastDown)
changes[std::min(lastUp, lastDown)] = std::max(lastUp, lastDown);
}

}
if ((upIndex + 1) % width != 0 && abs(data[index*pixelWidth] - data[(upIndex+1)*pixelWidth]) < Threshold) {
maxVal = std::max(maxVal, labels[upIndex+1]);
oldRight = labels[upIndex+1];
if (maxVal != oldRight) {
if(!(changes.find(oldRight) != changes.end() && changes[oldRight] > maxVal)) {
int lastUp = oldRight;
while (changes.find(lastUp) != changes.end()) {
lastUp = changes[lastUp];
}
int lastDown = maxVal;
while (changes.find(lastDown) != changes.end()) {
lastDown = changes[lastDown];
}
if (lastUp != lastDown)
changes[std::min(lastUp, lastDown)] = std::max(lastUp, lastDown);
}
}

}

}

#pragma omp parallel for num_threads(numThreads)
for (int i = border-localLabelSize; i < border+decrement; i++) {
if (changes.find(labels[i]) != changes.end()) {
int newVal = changes[labels[i]];
while (changes.find(newVal) != changes.end()) {

newVal = changes[newVal];
}
labels[i] = newVal;
}
}
decrement = localLabelSize;
}



}
double stop_time = getTime();
double segTime = stop_time - start_time;

std::unordered_map<int, int> red;
std::unordered_map<int, int> green;
std::unordered_map<int, int> blue;
std::unordered_map<int, int> count;

srand(seed);
start_time = getTime();
int clusters = 0;
int min_cluster = height*width;
int max_cluster = -1;
double avg_cluster = 0.0;

for (int i = 0; i < height; i++) {
for (int j = 0; j < width; j++) {
int label = labels[i*width+j];

if(label == 0) 
{
red[0] = 0;
green[0] = 0;
blue[0] = 0;

}
else {
if(red.find(label) == red.end())
{
clusters++;
count[label] = 1;

red[label] = (int)random()*255;
green[label] = (int)random()*255;
blue[label] = (int)random()*255;
}
else
count[label]++;
}
}
}

#pragma omp parallel for num_threads(numThreads) collapse(2)
for (int i = 0; i < height; i++) {
for (int j = 0; j < width; j++) {

int label = labels[i*width+j];
seg_data[(i*width+j)*3+0] = (char)red[label];
seg_data[(i*width+j)*3+1] = (char)blue[label];
seg_data[(i*width+j)*3+2] = (char)green[label];
}
}

for ( auto it = count.begin(); it != count.end(); ++it )
{
min_cluster = std::min( min_cluster, it->second);
max_cluster = std::max( max_cluster, it->second);
avg_cluster += it->second;
}

stop_time = getTime();
double colorTime = stop_time - start_time;

printf("Segmentation Time (sec): %f\n", segTime);
printf("Coloring Time     (sec): %f\n", colorTime);
printf("Total Time        (sec): %f\n", colorTime + segTime);
printf("-----------Statisctics---------------\n");
printf("Number of Clusters   : %d\n", clusters);
printf("Min Cluster Size     : %d\n", min_cluster);
printf("Max Cluster Size     : %d\n", max_cluster);
printf("Average Cluster Size : %f\n", avg_cluster/clusters);

printf("Writing Segmented Image...\n");
stbi_write_png(outputname, width, height, 3, seg_data, 0);
stbi_image_free(data);
free(seg_data);
free(labels);

printf("Done...\n");
return 0;
}
