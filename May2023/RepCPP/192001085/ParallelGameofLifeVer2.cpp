
#pragma warning(disable:4996)
#include <Windows.h>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;



typedef struct {
unsigned char red, green, blue;
} PPMPixel;

typedef struct {
int x, y;
PPMPixel *data;
} PPMImage;

constexpr auto RGB_COMPONENT_COLOR = 255;

const int iter = 100;


static PPMImage *readPPM(const char *filename)
{
char buff[16];
PPMImage *img = 0;
FILE *fp;
int c, rgb_comp_color;
fp = fopen(filename, "rb");
if (!fp) {
fprintf(stderr, "Unable to open file '%s'\n", filename);
exit(1);
}


if (!fgets(buff, sizeof(buff), fp)) {
perror(filename);
exit(1);
}

if (buff[0] != 'P' || buff[1] != '6') {
fprintf(stderr, "Invalid image format (must be 'P6')\n");
exit(1);
}

img = (PPMImage *)malloc(sizeof(PPMImage));
if (!img) {
fprintf(stderr, "Unable to allocate memory\n");
exit(1);
}

c = getc(fp);
while (c == '#') {
while (getc(fp) != '\n');
c = getc(fp);
}

ungetc(c, fp);
if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
exit(1);
}

if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
exit(1);
}

if (rgb_comp_color != RGB_COMPONENT_COLOR) {
fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
exit(1);
}

while (fgetc(fp) != '\n');
img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

if (!img) {
fprintf(stderr, "Unable to allocate memory\n");
exit(1);
}

if (fread(img->data, 3 * img->x, img->y, fp) != img->y) { 

fprintf(stderr, "Error loading image '%s'\n", filename);
exit(1);
}

fclose(fp);
return img;
}



void writePPM(const char *filename, PPMImage *img)
{
FILE *fp;
fp = fopen(filename, "wb");
if (!fp) {
fprintf(stderr, "Unable to open file '%s'\n", filename);
exit(1);
}

fprintf(fp, "P6\n");

fprintf(fp, "%d %d\n", img->x, img->y);

fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

fwrite(img->data, 3 * img->x, img->y, fp);
fclose(fp);
}





int getNumLiveNeighbors(int cellMatrix[][3]) {
int count = 0;
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
if (cellMatrix[i][j] > 0)
count++;
}
}
return count;
}
int getNumDeadNeighbors(int cellMatrix[][3]) {
int count = 0;
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
if (cellMatrix[i][j] == 0)
count++;
}
}
return count;
}

void runGame(PPMImage *inputImage) {
int numLiveCells = 0;
int numDeadCells = 0;
char pixel = 0;
int numCol = inputImage->x;
int numRow = inputImage->y;
int width = inputImage->x;
int height = inputImage->y;
int size = width * height;
int i = 0;
int j = 0;
int N = 0, NE = 0, NW = 0, E = 0, W = 0, S = 0, SE = 0, SW = 0;
int cellMatrix[3][3];
for (i = 0; i < (inputImage->x) * (inputImage->y); i++) {
if (((int)inputImage->data[i].red > 0  ) && ((int)inputImage->data[i].green > 0) && ((int)inputImage->data[i].blue > 0)) {

E = (i + 1);
W = (i - 1);
N = (i - width);
S = (i + width);
NE = (i - width + 1);
NW = (i - width - 1);
SW = (i + width - 1);
SE = (i + width + 1);

if ((NW >= 0) && ((i % width) != 0))
cellMatrix[0][0] = inputImage->data[NW].red;
if (N >= 0)
cellMatrix[0][1] = inputImage->data[N].red;
if (NE >= 0 && ((E % width) != 0))
cellMatrix[0][2] = inputImage->data[NE].red;
if (i % width != 0)
cellMatrix[1][0] = inputImage->data[W].red;
cellMatrix[1][1] = inputImage->data[i].red;
if (E % width != 0)
cellMatrix[1][2] = inputImage->data[E].red;
if ((SW < size) && ((i % width) != 0))
cellMatrix[2][0] = inputImage->data[SW].red;
if ((S < size))
cellMatrix[2][1] = inputImage->data[S].red;
if ((SE < size) && ((E % width) != 0))
cellMatrix[2][2] = inputImage->data[SE].red;

numLiveCells = getNumLiveNeighbors(cellMatrix);
numDeadCells = getNumDeadNeighbors(cellMatrix);
if (numLiveCells == 1 || numLiveCells == 2) {
inputImage->data[i].red = 0;
inputImage->data[i].green = 0;
inputImage->data[i].blue = 0;
}
if (numLiveCells >= 4) {
inputImage->data[i].red = 0;
inputImage->data[i].green = 0;
inputImage->data[i].blue = 0;
}


}
else if (((int)inputImage->data[i].red < 100) && ((int)inputImage->data[i].green < 100) && ((int)inputImage->data[i].blue < 100)) {
E = (i + 1);
W = (i - 1);
N = (i - width);
S = (i + width);
NE = (i - width + 1);
NW = (i - width - 1);
SW = (i + width - 1);
SE = (i + width + 1);

if ( (NW >= 0) && ( (i % width) != 0))
cellMatrix[0][0] = inputImage->data[NW].red;
if(N >= 0)
cellMatrix[0][1] = inputImage->data[N].red;
if(NE >=0 && ((E % width)!=0))
cellMatrix[0][2] = inputImage->data[NE].red;
if(i % width != 0)
cellMatrix[1][0] = inputImage->data[W].red;
cellMatrix[1][1] = inputImage->data[i].red;
if( E % width !=0)
cellMatrix[1][2] = inputImage->data[E].red;
if( (SW <size) && ((i % width) !=0) )
cellMatrix[2][0] = inputImage->data[SW].red;
if( (S < size))
cellMatrix[2][1] = inputImage->data[S].red;
if((SE < size)&& ((E % width) !=0))
cellMatrix[2][2] = inputImage->data[SE].red;

numLiveCells = getNumLiveNeighbors(cellMatrix);
numDeadCells = getNumDeadNeighbors(cellMatrix);

if (numLiveCells == 3) {
inputImage->data[i].red = 255;
inputImage->data[i].green = 255;
inputImage->data[i].blue = 255;
}
}
}
}



int main() {

PPMImage *inputImage, *outputImage;
string temp = "";
int i = 0;
cout << "Maximum number of threads:  " << omp_get_max_threads() << endl;
cout << "Reading Edge-Detected model.ppm file...\n";
auto start = high_resolution_clock::now();
inputImage = readPPM("sobel.ppm");
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << "Finished in " << duration.count() << " microseconds" << endl;
outputImage = inputImage;

cout << "Running Conway's Game of Life...\n";
cout << "Serial...\n";

start = high_resolution_clock::now();

for (int i = 0; i < iter; i++) {
inputImage = outputImage;
runGame(inputImage);
temp = "Iteration" + to_string(i + 1) + ".ppm";
writePPM(temp.c_str(), outputImage);
}

stop = high_resolution_clock::now();
auto duration2 = duration_cast<seconds>(stop - start);
cout << "Finished in " << duration2.count() << " seconds" << endl;


cout << "Parallel...\n";
start = high_resolution_clock::now();
#pragma omp parallel for
for ( int i = 0; i < iter; i++) {
inputImage = outputImage;
runGame(inputImage);
temp = "Iteration" + to_string(i + 1) + ".ppm";
writePPM(temp.c_str(), outputImage);
}
stop = high_resolution_clock::now();
auto duration3 = duration_cast<seconds>(stop - start);
cout << "Finished in " << duration3.count() << " seconds" << endl;

auto speedup = (duration2.count() / duration3.count());
cout << "Speedup is " << speedup;

return 0;
}


