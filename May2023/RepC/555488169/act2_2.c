#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NUM_THREADS 200
void blurring(char outputImage[], int sizeM);
void blurring_inv(char outputImgName[], int sizeM);
int main(){
const double startTime = omp_get_wtime();
#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
blurring("sample5_blur_sections_gris1.bmp", 3);
#pragma omp section
blurring("sample5_blur_sections_gris2.bmp", 5);
#pragma omp section
blurring("sample5_blur_sections_gris3.bmp", 7);
#pragma omp section
blurring("sample5_blur_sections_gris4.bmp", 9);
#pragma omp section
blurring("sample5_blur_sections_gris5.bmp", 11);
#pragma omp section
blurring("sample5_blur_sections_gris6.bmp", 13);
#pragma omp section
blurring("sample5_blur_sections_gris7.bmp", 15);
#pragma omp section
blurring("sample5_blur_sections_gris8.bmp", 17);
#pragma omp section
blurring("sample5_blur_sections_gris9.bmp", 19);
#pragma omp section
blurring_inv("sample5_blur_sections_gris10.bmp", 21);
#pragma omp section
blurring_inv("sample5_blur_sections_gris11.bmp", 19);   
#pragma omp section
blurring_inv("sample5_blur_sections_gris12.bmp", 17);
#pragma omp section
blurring_inv("sample5_blur_sections_gris13.bmp", 15);
#pragma omp section
blurring_inv("sample5_blur_sections_gris14.bmp", 13);
#pragma omp section
blurring_inv("sample5_blur_sections_gris15.bmp", 11);
#pragma omp section
blurring_inv("sample5_blur_sections_gris16.bmp", 9);
#pragma omp section
blurring_inv("sample5_blur_sections_gris17.bmp", 7);
#pragma omp section
blurring_inv("sample5_blur_sections_gris18.bmp", 5);  
#pragma omp section
blurring_inv("sample5_blur_sections_gris19.bmp", 3);
}
}
const double endTime = omp_get_wtime();
printf("\nTiempo total  = %f\n", endTime-startTime);
return 0;
}
void blurring(char outputImageName[], int sizeM){
FILE *image, *outputImage, *lecturas;
image = fopen("5.bmp","rb");                 
outputImage = fopen(outputImageName,"wb");      
long ancho;
long alto;
unsigned char r, g, b;    
unsigned char* ptr;
unsigned char xx[54];
long cuenta = 0, anchoR = 0, altoR = 0, anchoM = 0, altoM = 0;
long sum;
int iR, jR;
for(int i = 0; i < 54; i++) {
xx[i] = fgetc(image);
fputc(xx[i], outputImage);      
}
ancho = (long)xx[20]*65536 + (long)xx[19]*256 + (long)xx[18];
alto = (long)xx[24]*65536 + (long)xx[23]*256 + (long)xx[22];
ptr = (unsigned char*)malloc(alto*ancho*3* sizeof(unsigned char));
omp_set_num_threads(NUM_THREADS);
unsigned char foto[alto][ancho], fotoB[alto][ancho];
unsigned char pixel;
for(int i = 0; i < alto; i++){
for(int j = 0; j < ancho; j++){
b = fgetc(image);
g = fgetc(image);
r = fgetc(image);
pixel = 0.21*r + 0.72*g + 0.07*b;
foto[i][j] = pixel;
fotoB[i][j] = pixel;
}
}
anchoR = ancho/sizeM;
altoR = alto/sizeM;
anchoM = ancho%sizeM;
altoM = alto%sizeM;
int inicioX,inicioY,cierreX,cierreY,ladoX,ladoY;
for(int i = 0; i < alto; i++){
for(int j = 0; j < ancho; j++){
if(i < sizeM){
inicioX = 0;
cierreX = i+sizeM;
ladoX = i+sizeM;
} else if(i >= alto-sizeM){
inicioX = i-sizeM;
cierreX = alto;
ladoX = alto-i+sizeM;
}else{
inicioX = i-sizeM;
cierreX = i+sizeM;
ladoX = sizeM*2+1;
}
if(j < sizeM){
inicioY = 0;
cierreY = j+sizeM;
ladoY = j+sizeM;
}else if(j >= ancho-sizeM){
inicioY = j-sizeM;
cierreY = ancho;
ladoY = ancho-j+sizeM;
}else{
inicioY = j-sizeM;
cierreY = j+sizeM;
ladoY = sizeM*2+1;
}
sum = 0;
for(int x = inicioX; x < cierreX; x++){
for(int y = inicioY; y < cierreY; y++){
sum += foto[x][y];
}
}
sum = sum/(ladoX*ladoY);
fotoB[i][j] = sum;
}
}
cuenta = 0;
for(int i = 0; i < alto; i++){
for(int j = 0; j < ancho; j++){
ptr[cuenta] = fotoB[i][j]; 
ptr[cuenta+1] = fotoB[i][j]; 
ptr[cuenta+2] = fotoB[i][j]; 
cuenta++;
}
}       
const double startTime = omp_get_wtime();
#pragma omp parallel
{
#pragma omp for schedule(dynamic)
for (int i = 0; i < alto*ancho; ++i) {
fputc(ptr[i], outputImage);
fputc(ptr[i+1], outputImage);
fputc(ptr[i+2], outputImage);
}
}
const double endTime = omp_get_wtime();
printf("\nNormal\n");
printf("Threads: %d\n", NUM_THREADS);
printf("Tiempo = %f\n", endTime-startTime);
free(ptr);
fclose(image);
fclose(outputImage);
}
void blurring_inv(char outputImgName[], int sizeM){
FILE *image, *outputImage, *lecturas;
image = fopen("5.bmp","rb");
outputImage = fopen(outputImgName,"wb");
long width;
long height;
unsigned char r, g, b;
unsigned char* ptr;
unsigned char xx[54];
long counter = 0, widthR = 0, heightR = 0, widthM = 0, heightM = 0;
long sum;
int iR, jR;
for (int i = 0; i < 54; i++) {
xx[i] = fgetc(image);
fputc(xx[i], outputImage);
}
width = (long)xx[20]*65536 + (long)xx[19]*256 + (long)xx[18];
height = (long)xx[24]*65536 + (long)xx[23]*256 + (long)xx[22];
ptr = (unsigned char*)malloc(height*width*3* sizeof(unsigned char));
omp_set_num_threads(NUM_THREADS);
unsigned char photo[height][width], photoB[height][width];
unsigned char pixel;
for(int i = 0; i < height; i++){
for(int j = 0; j < width; j++){
b = fgetc(image);
g = fgetc(image);
r = fgetc(image);
pixel = 0.21*r + 0.72*g + 0.07*b;
photo[i][j] = pixel;
photoB[i][j] = pixel;
}
}
widthR = width / sizeM;
heightR = height / sizeM;
widthM = width % sizeM;
heightM = height % sizeM;
int startX, startY, endX, endY, sideX, sideY;
for(int i = 0; i < height; i++) {
for(int j = 0; j < width; j++) {
if (i < sizeM) {
startX = height;
endX = 0;
sideX = i+sizeM;
} else if (i >= height-sizeM) {
startX = i+sizeM;
endX = 0;
sideX = height-i+sizeM;
} else {
startX = i-sizeM;
endX = i+sizeM;
sideX = sizeM*2+1;
}
if (j < sizeM) {
startY = width;
endY = j+sizeM;
sideY = j+sizeM;
} else if (j >= width-sizeM) {
startY = j-sizeM;
endY = width;
sideY = width-j+sizeM;
} else {
startY = j-sizeM;
endY = j+sizeM;
sideY = sizeM*2+1;
}
sum = 0;
for (int x = startX; x < endX; x++) {
for (int y = startY; y < endY; y++) {
sum += photo[x][y];
}
}
sum = sum / (sideX*sideY);
photoB[i][j] = sum;
}
}
counter = 0;
for (int i = 0; i < height; i++) {
for (int j = 0; j < width; j++) {
ptr[counter] = photoB[i][width-j]; 
ptr[counter+1] = photoB[i][width-j]; 
ptr[counter+2] = photoB[i][width-j]; 
counter++;
}
}       
const double t1 = omp_get_wtime();
#pragma omp parallel
{
#pragma omp for schedule(dynamic)
for (int i = 0; i < height*width; ++i) {
fputc(ptr[i], outputImage);
fputc(ptr[i+1], outputImage);
fputc(ptr[i+2], outputImage);
}
}
const double t2 = omp_get_wtime();
printf("\nInvertida\n");
printf("Threads: %d\n", NUM_THREADS);
printf("Tiempo = %f\n", t2-t1);
free(ptr);
fclose(image);
fclose(outputImage);
}