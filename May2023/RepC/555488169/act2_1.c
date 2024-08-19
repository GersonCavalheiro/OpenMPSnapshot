#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NUM_THREADS 20
void imageGray(char inputImageName[], char outputImageName[]);
void imageBlue(char inputImageName[], char outputImageName[]);
void imageGreen(char inputImageName[], char outputImageName[]);
void imageRed(char inputImageName[], char outputImageName[]);
int main()
{   
#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
imageGray("5.bmp", "imageGray_5.bmp");
#pragma omp section
imageBlue("5.bmp", "imageBlue_5.bmp");
#pragma omp section
imageGreen("5.bmp", "imageGreen_5.bmp");
#pragma omp section
imageRed("5.bmp", "imageRed_5.bmp");
}
}
return 0;
}
void imageGray(char inputImageName[], char outputImageName[]){
omp_set_num_threads(NUM_THREADS);
FILE *image, *outputImage, *lecturas;
image = fopen(inputImageName,"rb");          
outputImage = fopen(outputImageName,"wb");    
long ancho;
long alto;
int nthreads;
unsigned char r, g, b;               
unsigned char xx[54];
for(int i=0; i<54; i++){
xx[i] = fgetc(image);
fputc(xx[i], outputImage);   
}
ancho = (long)xx[20]*65536 + (long)xx[19]*256 + (long)xx[18];
alto = (long)xx[24]*65536 + (long)xx[23]*256 + (long)xx[22];
long n = ancho * alto * 3;
const double startTime = omp_get_wtime();
nthreads = omp_get_num_threads();
#pragma omp parallel for      
for (int i = 0; i < n; i++){
b = fgetc(image);
g = fgetc(image);
r = fgetc(image);
unsigned char pixel = 0.21*r + 0.72*g + 0.07*b;
fputc(pixel, outputImage);  
fputc(pixel, outputImage);  
fputc(pixel, outputImage);  
}
const double endTime = omp_get_wtime();
printf("Tiempo imagen gris = %f\n", endTime-startTime);
fclose(image);
fclose(outputImage);
}
void imageBlue(char inputImageName[], char outputImageName[]){
omp_set_num_threads(NUM_THREADS);
FILE *image, *outputImage, *lecturas;
image = fopen(inputImageName,"rb");          
outputImage = fopen(outputImageName,"wb");    
long ancho;
long alto;
int nthreads;
unsigned char r, g, b;               
unsigned char xx[54];
for(int i=0; i<54; i++){
xx[i] = fgetc(image);
fputc(xx[i], outputImage);   
}
ancho = (long)xx[20]*65536 + (long)xx[19]*256 + (long)xx[18];
alto = (long)xx[24]*65536 + (long)xx[23]*256 + (long)xx[22];
long n = ancho * alto * 3;
const double startTime = omp_get_wtime();
nthreads = omp_get_num_threads();
#pragma omp parallel for      
for (int i = 0; i < n; i++){
b = fgetc(image);
g = fgetc(image);
r = fgetc(image);
unsigned char pixel = 0.21*r + 0.72*g + 0.07*b;
fputc(b, outputImage);  
fputc(0, outputImage);  
fputc(0, outputImage);  
}
const double endTime = omp_get_wtime();
printf("Tiempo imagen azul = %f\n", endTime-startTime);
fclose(image);
fclose(outputImage);
}
void imageGreen(char inputImageName[], char outputImageName[]){
omp_set_num_threads(NUM_THREADS);
FILE *image, *outputImage, *lecturas;
image = fopen(inputImageName,"rb");          
outputImage = fopen(outputImageName,"wb");    
long ancho;
long alto;
int nthreads;
unsigned char r, g, b;               
unsigned char xx[54];
for(int i=0; i<54; i++){
xx[i] = fgetc(image);
fputc(xx[i], outputImage);   
}
ancho = (long)xx[20]*65536 + (long)xx[19]*256 + (long)xx[18];
alto = (long)xx[24]*65536 + (long)xx[23]*256 + (long)xx[22];
long n = ancho * alto * 3;
const double startTime = omp_get_wtime();
nthreads = omp_get_num_threads();
#pragma omp parallel for      
for (int i = 0; i < n; i++){
b = fgetc(image);
g = fgetc(image);
r = fgetc(image);
unsigned char pixel = 0.21*r + 0.72*g + 0.07*b;
fputc(0, outputImage);  
fputc(g, outputImage);  
fputc(0, outputImage);  
}
const double endTime = omp_get_wtime();
printf("Tiempo imagen verde = %f\n", endTime-startTime);
fclose(image);
fclose(outputImage);
}
void imageRed(char inputImageName[], char outputImageName[]){
omp_set_num_threads(NUM_THREADS);
FILE *image, *outputImage, *lecturas;
image = fopen(inputImageName,"rb");          
outputImage = fopen(outputImageName,"wb");    
long ancho;
long alto;
int nthreads;
unsigned char r, g, b;               
unsigned char xx[54];
for(int i=0; i<54; i++){
xx[i] = fgetc(image);
fputc(xx[i], outputImage);   
}
ancho = (long)xx[20]*65536 + (long)xx[19]*256 + (long)xx[18];
alto = (long)xx[24]*65536 + (long)xx[23]*256 + (long)xx[22];
long n = ancho * alto * 3;
const double startTime = omp_get_wtime();
nthreads = omp_get_num_threads();
#pragma omp parallel for      
for (int i = 0; i < n; i++){
b = fgetc(image);
g = fgetc(image);
r = fgetc(image);
unsigned char pixel = 0.21*r + 0.72*g + 0.07*b;
fputc(0, outputImage);  
fputc(0, outputImage);  
fputc(r, outputImage);  
}
const double endTime = omp_get_wtime();
printf("Tiempo imagen rojo = %f\n", endTime-startTime);
fclose(image);
fclose(outputImage);
}