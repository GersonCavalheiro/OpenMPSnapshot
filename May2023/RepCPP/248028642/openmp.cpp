

#include <stdio.h>
#include <math.h>
#include <thread>
#include <vector>
#include <iostream>
#include <omp.h>


const int IterationMax = 200;

const double EscapeRadius = 2;
double ER2 = EscapeRadius * EscapeRadius;

unsigned char colorTheme[][3] = {{220, 230, 255}, {180, 190, 23}, {42, 129, 84}, {200, 10, 30}, {49, 23, 95}, {120, 90, 32}, {220, 220, 40}, {90, 255, 30}, {30, 30, 225}, {128, 190, 48}};

void mainRunner(int threadsCount, int imageSize) {

int iXmax = imageSize;
int iYmax = imageSize;

double CxMin = -2.5;
double CxMax = 1.5;
double CyMin = -2.0;
double CyMax = 2.0;

double PixelWidth = (CxMax - CxMin) / iXmax;
double PixelHeight = (CyMax - CyMin) / iYmax;
unsigned char*** color = new unsigned char**[iXmax];
for (int i=0; i<iXmax; i++) {
color[i] = new unsigned char*[iYmax];
for (int j=0; j<iYmax; j++) {
color[i][j] = new unsigned char[3];
}
}



const int MaxColorComponentValue = 255;
FILE *fp;
std::string fileName = "thread" + std::to_string(threadsCount) + "_" + std::to_string(iXmax) + ".ppm";
const char *filename = fileName.c_str(); 
char *comment = "# "; 

fp = fopen(filename, "wb"); 

fprintf(fp, "P6\n %s\n %d\n %d\n %d\n", comment, iXmax, iYmax, MaxColorComponentValue);

int iX, iY;
double Cx, Cy;
double Zx, Zy;
double Zx2, Zy2; 
int Iteration;
int threadNumber;
int *iterationCount = new int[threadsCount];
for (int i=0; i<threadsCount; i++) {
iterationCount[i] = 0;
}

auto start = omp_get_wtime();

#pragma omp parallel private(threadNumber) shared(color, iterationCount) num_threads(threadsCount)
{
threadNumber = omp_get_thread_num();
#pragma omp for private(iX, iY, Cx, Cy, Zx, Zy, Zx2, Zy2, Iteration) schedule(runtime)

for (iY = 0; iY < iYmax; iY++) {
Cy = CyMin + iY * PixelHeight;
for (iX = 0; iX < iXmax; iX++) {
Cx = CxMin + iX * PixelWidth;

Zx = 0.0;
Zy = 0.0;
Zx2 = Zx * Zx;
Zy2 = Zy * Zy;
for (Iteration = 0; Iteration < IterationMax && ((Zx2 + Zy2) < ER2); Iteration++) {
iterationCount[threadNumber]++;
Zy = 2 * Zx * Zy + Cy;
Zx = Zx2 - Zy2 + Cx;
Zx2 = Zx * Zx;
Zy2 = Zy * Zy;
};

if (Iteration == IterationMax) { 
color[iX][iY][0] = 0;
color[iX][iY][1] = 0;
color[iX][iY][2] = 0;
}
else {

color[iX][iY][0] = colorTheme[threadNumber][0]; 
color[iX][iY][1] = colorTheme[threadNumber][1]; 
color[iX][iY][2] = colorTheme[threadNumber][2]; 
};
}
}
}
auto end = omp_get_wtime();
auto elapsedTime = end - start; 
std::cout << elapsedTime << std::endl;
std::cout << "Iterations count:" << std::endl;
for (int i=0; i<threadsCount; i++) {
std::cout << iterationCount[i] << ", ";
}
std::cout << std::endl;


for (int iY = 0; iY < iYmax; iY++)
{
for (int iX = 0; iX < iXmax; iX++)
{
fwrite(color[iX][iY], 1, 3, fp);
}
}
fclose(fp);


for (int i=0; i<iXmax; i++) {
for (int j=0; j<iYmax; j++) {
delete[] color[i][j];
}
delete[] color[i];
}
delete[] color;

delete[] iterationCount;

}

int main() {
const int imageSizes[] = {800, 1600, 3200, 6400};
for (int j=0; j<4; j++) {
std::cout << "Image size " << imageSizes[j] << std::endl;
for (int i=1; i<9; i++) {
std::cout << "Threads: " << i << std::endl;
mainRunner(i, imageSizes[j]);
} 
std::cout << std::endl;
}
}
