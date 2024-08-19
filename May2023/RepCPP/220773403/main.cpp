#include <cstdlib> 
#include <iostream> 
#include "Bitmap.h"
#include <ctime>
#include "MedianFilter.h"
#include<omp.h>

#define ITERS 500


int CompareBitmaps( Bitmap* inputA, Bitmap* inputB )
{
int x = inputA->Width();
int y = inputA->Height();
int count = 0;
for(int i=0;i<x;i++)
{
for(int j=0;j<y;j++)
{
char a = inputA->GetPixel(i, j);
char b = inputB->GetPixel(i, j);
if(a != b)
{
count++;
}
}
}
return count;
}
void iSort(unsigned char* array, int size)
{
int i, key, j;  
for (i = 1; i < size; i++) 
{  
key = array[i];  
j = i - 1;
while (j >= 0 && array[j] > key) 
{  
array[j + 1] = array[j];  
j = j - 1;  
}  
array[j + 1] = key;  
}  
}  


void MedianFilter(Bitmap* image, Bitmap* outputImage)
{
#pragma omp parallel
{
int height = image->Height();
int width = image->Width();
int n=height*width;
unsigned char PixelVals[9];
int id=omp_get_thread_num();
#pragma omp for
for(int i = 0;i<width;i++)
{
for(int j = 0;j<height;j++)
{
PixelVals[0] = image->GetPixel(i,j);
if(i>0)
{
PixelVals[1] = image->GetPixel(i-1,j);
}
else
{
PixelVals[1] = 0;
}
if(i<(width-1))
{
PixelVals[2] = image->GetPixel(i+1,j);
}
else
{
PixelVals[2] = 0;
}
if(j>0)
{
PixelVals[3] = image->GetPixel(i,j-1);
}
else
{
PixelVals[3] = 0;
}
if(j>0 && i>0)
{
PixelVals[4] = image->GetPixel(i-1,j-1);
}
else
{
PixelVals[4] = 0;
}
if(j>0 && i<(width-1))
{
PixelVals[5] = image->GetPixel(i+1,j-1);
}
else
{
PixelVals[5] = 0;
}
if(j<(height-1))
{
PixelVals[6] = image->GetPixel(i,j+1);
}
else
{
PixelVals[6] = 0;
}
if(j<(height-1) && i>0)
{
PixelVals[7] = image->GetPixel(i-1,j+1);
}
else
{
PixelVals[7] = 0;
}
if(j<(height-1) && i<(width-1))
{
PixelVals[8] = image->GetPixel(i+1,j+1);
}
else
{
PixelVals[8] = 0;
}
iSort(PixelVals,9);
outputImage->SetPixel(i,j,PixelVals[4]);

}
}


}
}

float ComputeL2Norm( Bitmap* inputA, Bitmap* inputB )
{
int x = inputA->Width();
int y = inputA->Height();
float sum = 0, delta = 0;
unsigned char a,b;
for(int i=0;i<x;i++)
{
for(int j=0;j<y;j++)
{
a = inputA->GetPixel(i, j);
b = inputB->GetPixel(i, j);
delta += (a - b) * (a - b);
sum += (a * b);

}
}
float L2norm = sqrt(delta / sum);
return L2norm;
}

int main()
{
omp_set_num_threads(1);

float tcpu, tgpu;
clock_t start, end;
float L2Norm;
int pixelcount;
bool success;

Bitmap InputImage;
InputImage.Load("Lenna.bmp");
int width = InputImage.Width();
int height = InputImage.Height();

std::cout<<"\nNumber of iterations: "<<ITERS<<std::endl;
std::cout<<"operating on an image of size: "<<width<<" x "<<height<<std::endl;
Bitmap OutputImage(width,height);
OutputImage.Save("OutputImage.bmp");
start = clock();
for (int i = 0; i < ITERS; i++)
{
MedianFilter(&InputImage,&OutputImage);
}
end = clock();
OutputImage.Save("OutputImage.bmp");
tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
std::cout << "\nHost Computation took " << tcpu << " ms" << std::endl;
return 0;
}
