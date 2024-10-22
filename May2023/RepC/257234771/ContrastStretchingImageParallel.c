#include <stdlib.h>
#include <stdio.h>
#include<string.h>
#include <png.h>
#include<omp.h>
void read_png_file(char *filename,char *outputFile) 
{
int width,height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;
int maskDimensions;
int **maskMatrix;
FILE *fp = fopen(filename, "rb");
png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
if(!png) abort();
png_infop info = png_create_info_struct(png);
if(!info) abort();
if(setjmp(png_jmpbuf(png))) abort();
png_init_io(png, fp);
png_read_info(png, info);
width     = png_get_image_width(png, info);
height    = png_get_image_height(png, info);
color_type = png_get_color_type(png, info);
bit_depth  = png_get_bit_depth(png, info);
if(bit_depth == 16)
png_set_strip_16(png);
if(color_type == PNG_COLOR_TYPE_PALETTE)
png_set_palette_to_rgb(png);
if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
png_set_expand_gray_1_2_4_to_8(png);
if(png_get_valid(png, info, PNG_INFO_tRNS))
png_set_tRNS_to_alpha(png);
if(color_type == PNG_COLOR_TYPE_RGB ||
color_type == PNG_COLOR_TYPE_GRAY ||
color_type == PNG_COLOR_TYPE_PALETTE)
png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
if(color_type == PNG_COLOR_TYPE_GRAY ||
color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
png_set_gray_to_rgb(png);
png_read_update_info(png, info);
row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
for(int y = 0; y < height; y++) {
row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
}
png_read_image(png, row_pointers);
fclose(fp);
png_destroy_read_struct(&png, &info, NULL);
int minRed = 255, maxRed = 0, minGreen = 255, maxGreen = 0, minBlue = 255, maxBlue = 0;
for(int i=0 ; i < height; i++)
{
png_bytep row = row_pointers[i];
for(int j=0 ; j < width ; j++)
{
png_bytep px = &(row[j * 4]);
if(px[0] < minRed)
minRed = px[0];
if(px[0] > maxRed)
maxRed = px[0];
if(px[1] < minGreen)
minGreen = px[1];
if(px[1] > maxGreen)
maxGreen = px[1];
if(px[2] < minBlue)
minBlue = px[2];
if(px[2] > maxBlue)
maxBlue = px[2];
}
}
int preComputationRed = 255/(maxRed - minRed);
int preComputationGreen = 255/(maxGreen - minGreen);
int preComputationBlue = 255/(maxBlue - minBlue);
for(int i=0 ; i < height; i++)
{
png_bytep row = row_pointers[i];
for(int j=0 ; j < width ; j++)
{
png_bytep px = &(row[j * 4]);
px[0] = (preComputationRed)*(px[0] - minRed);
if(px[0] < 0)
px[0] = 0;
if(px[0] > 255)
px[0] = 255;
px[1] = (preComputationGreen)*(px[1] - minGreen);
if(px[1] < 0)
px[1] = 0;
if(px[1] > 255)
px[1] = 255;
px[2] = (preComputationBlue)*(px[2] - minBlue);
if(px[2] < 0)
px[2] = 0;
if(px[2] > 255)
px[2] = 255;
}
}
fp = fopen(outputFile, "wb");
if(!fp) abort();
png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
if (!png) abort();
info = png_create_info_struct(png);
if (!info) abort();
if (setjmp(png_jmpbuf(png))) abort();
png_init_io(png, fp);
png_set_IHDR(
png,
info,
width, height,
8,
PNG_COLOR_TYPE_RGBA,
PNG_INTERLACE_NONE,
PNG_COMPRESSION_TYPE_DEFAULT,
PNG_FILTER_TYPE_DEFAULT
);
png_write_info(png, info);
if (!row_pointers) abort();
png_write_image(png, row_pointers);
png_write_end(png, NULL);
for(int y = 0; y < height; y++) {
free(row_pointers[y]);
}
free(row_pointers);
fclose(fp);
png_destroy_write_struct(&png, &info);
}
int main() 
{ 
double startTime = omp_get_wtime();
#pragma omp parallel for
for(int i=1;i<=800;i++)
{
char str[25]="cat (";
char out[25]="out (";
int a=i;
int tmp=i;
int cnt=0;
while(tmp)
{
tmp=tmp/10;
cnt++;
}
int j=cnt-1;
char pok[25]=").png";
char lok[25];
while(a)
{
int k=a%10;
lok[j]=(char)('0'+k);
a=a/10;
j--;
}
lok[cnt]='\0';
strcat(str,lok);
strcat(str,pok);
strcat(out,lok);
strcat(out,pok);
char* s=out;
char* p=str;
read_png_file(p,s);
}
double endTime = omp_get_wtime();
printf("Time taken is %lf",endTime - startTime);
return 0;
}
