#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <utility>
#include <vector>

struct BGRColor {
static const int TOTAL_POSSIBLE_COLORS = (1 << 24); 	
int colorValue;						
BGRColor(int color) { colorValue = color; }
BGRColor(int blue, int green, int red) { colorValue = (blue << 16) ^ (green << 8) ^ red; }
~BGRColor() {}									
int getValue() { return colorValue; }						
int getBlue() { return (colorValue >> 16); }					
int getGreen() { return ((colorValue >> 8) ^ (getBlue() << 8)); }		
int getRed() { return (colorValue ^ (getBlue() << 16) ^ (getGreen() << 8)); }	
};

struct BGRImage {
long long nRows; 	
long long nCols; 	
BGRColor *colors; 	
BGRImage(int *bgrColors, long long cols, long long rows) {
nRows = rows;
nCols = cols;
long long size = nRows * nCols;
colors = (BGRColor *) malloc (size * sizeof(BGRColor));
for(long long i = 0; i < size; i++) { colors[i] = BGRColor(bgrColors[3 * i], bgrColors[3 * i + 1], bgrColors[3 * i + 2]); }
}
~BGRImage() { free(colors); }			
static const int INVALID_BLOCK_SIZE = -1; 	
static const int INVALID_ROW_NUMBER = -2; 	
static const int INVALID_COL_NUMBER = -3;	
static const int INVALID_BLOCK_NUMBER = -4;	
void print() { printf("Number of rows = %lld | Number of columns = %lld\n", nRows, nCols); }
int getPixel(int _block_size, int _block, int _col, int _row) {
if(nCols % _block_size || nRows % _block_size) return INVALID_BLOCK_SIZE;
if(_row < 0 || _row >= _block_size) return INVALID_ROW_NUMBER;
if(_col < 0 || _col >= _block_size) return INVALID_COL_NUMBER;
int numOfBlockInRow = nCols / _block_size;
int numOfBlockInCol = nRows / _block_size;
if(_block < 0 || _block >= numOfBlockInRow * numOfBlockInCol) return INVALID_BLOCK_NUMBER;

int blockColOffset = (_block % numOfBlockInCol) * _block_size;
int blockRowOffset = (_block / numOfBlockInCol) * _block_size;

return (blockRowOffset + _row) * nCols + (blockColOffset + _col);
}
BGRColor getColor(int pixel) { return colors[pixel]; }			
std::vector< std::pair<BGRColor, int> > *calculateHistogramParallel(int n_threads) {
int *histogramArray = (int *) malloc (BGRColor::TOTAL_POSSIBLE_COLORS * sizeof(int));
long long i, j;
#pragma omp parallel for num_threads(n_threads) private(i)
for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) { histogramArray[i] = 0; }
long long _size = nRows * nCols;
#pragma omp parallel for num_threads(n_threads) private(i)
for(i = 0; i < _size; i++) { 
#pragma omp atomic
histogramArray[getColor(i).getValue()]++;
}
std::vector< std::pair<BGRColor, int> > *histogram = new std::vector< std::pair<BGRColor, int> >();
for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) 
{ if(histogramArray[i]) histogram->push_back(std::make_pair(BGRColor(i), histogramArray[i])); }
free(histogramArray);
return histogram;
}
std::vector< std::pair<BGRColor, int> > *calculateBlockHistogramParallel(int n_threads, int b_size, int _block) {
if(nRows % b_size != 0 || nCols % b_size != 0) return new std::vector< std::pair<BGRColor, int> >();
int *histogramArray = (int *) malloc (BGRColor::TOTAL_POSSIBLE_COLORS * sizeof(int));
long long i, j;
#pragma omp parallel for num_threads(n_threads) private(i)
for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) { histogramArray[i] = 0; }
#pragma omp parallel for num_threads(n_threads) private(i, j)
for(i = 0; i < b_size; i++) { 
for(j = 0; j < b_size; j++) {
#pragma omp atomic
histogramArray[getColor(getPixel(b_size, _block, i, j)).getValue()]++; 
}
}
std::vector< std::pair<BGRColor, int> > *histogram = new std::vector< std::pair<BGRColor, int> >();
for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) 
{ if(histogramArray[i]) histogram->push_back(std::make_pair(BGRColor(i), histogramArray[i])); }
free(histogramArray);
return histogram;
}
};

void printTime(double start, double end);
void writeToFile(std::vector< std::pair<BGRColor, int> > &hist, const char *name);
const int NUM_THREADS = 8;

int main(int argc, char *argv[]) {

FILE *colorDataFd;
long long x, y, total_input;
int *colorData;
double startTime, endTime;
BGRImage *image;


printf("File input: %s\n", argv[1]);
colorDataFd = fopen(argv[1], "r");	
fscanf(colorDataFd, "%lld %lld", &x, &y);
total_input = x * y;
colorData = (int *) malloc (total_input * sizeof(int));
for(long long i=0; i<total_input; i++) fscanf(colorDataFd, "%d", &colorData[i]);
fclose(colorDataFd);
image = new BGRImage(colorData, x / 3, y);
free(colorData);
image->print();

printf("## PARALLELIZE PROCESS  --- ");
startTime = omp_get_wtime();
std::vector< std::pair<BGRColor, int> > *parallelHistogram = image->calculateHistogramParallel(NUM_THREADS);
endTime = omp_get_wtime();


printTime(startTime, endTime);
writeToFile(*parallelHistogram, "histogram-parallel");
delete parallelHistogram;
delete image;

return 0;
}

void printTime(double start, double end) { printf("Elapsed time: %lf\n", end - start); }

void writeToFile(std::vector< std::pair<BGRColor, int> > &hist, const char *name) {
FILE *colorDataFd = fopen(name, "w");
fprintf(colorDataFd, "B,G,R,N\n");
std::vector< std::pair<BGRColor, int> >::iterator it;
int b, g, r, count;
for(it = hist.begin(); it != hist.end(); it++) {
b = it->first.getBlue();
g = it->first.getGreen();
r = it->first.getRed();
count = it->second;
fprintf(colorDataFd, "%d,%d,%d,%d\n", b, g, r, count);
}
fclose(colorDataFd);
}

