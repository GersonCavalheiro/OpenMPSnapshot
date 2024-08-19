#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <iomanip>
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

typedef struct {
unsigned char red,green,blue;
} PPMPixel;

typedef struct {
int x, y;
PPMPixel *data;
} PPMImage;

using namespace std;

#define CREATOR "ICETROOPER"
#define RGB_COMPONENT_COLOR 255

inline static unsigned char GET_PIXEL_CHECK(PPMImage* img, int x, int y, int l) {
if ( (x<0) || (x >= img->x) || (y<0) || (y >= img->y) ) return 0;

if(l == 0)
{
return img->data[x + y*img->x].red;
}
else if(l == 1)
{
return img->data[x + y*img->x].green;
}
else if(l == 2)
{
return img->data[x + y*img->x].blue;
}
}

PPMImage * filter(PPMImage * im, double * K, int Ks, double divisor, double offset) {
PPMImage * oi;

oi = (PPMImage *)malloc(sizeof(PPMImage));
if (!oi) {
fprintf(stderr, "Unable to allocate memory\n");
exit(1);
}

oi->x = im->x;
oi->y = im->y;

oi->data = (PPMPixel*)malloc(im->x * im->y * sizeof(PPMPixel));
if (!oi) {
fprintf(stderr, "Unable to allocate memory\n");
exit(1);
}

if (oi != nullptr)
{


tbb::parallel_for(0, im->x, 1, [=](int ix)
{
double cp[3];

for (int iy = 0; iy < im->y; iy++)
{
cp[0] = cp[1] = cp[2] = 0.0;
for (int kx = -Ks; kx <= Ks; kx++)
{
for (int ky = -Ks; ky <= Ks; ky++)
{
for (int l = 0; l < 3; l++)
cp[l] += (K[(kx + Ks) +
(ky + Ks) * (2 * Ks + 1)] / divisor) *
((double) GET_PIXEL_CHECK(im, ix + kx, iy + ky, l)) + offset;
}
}
for (int l = 0; l < 3; l++)
cp[l] = (cp[l] > 255.0) ? 255.0 : ((cp[l] < 0.0) ? 0.0 : cp[l]);

oi->data[ix + iy*im->x].red = (unsigned char)cp[0];
oi->data[ix + iy*im->x].green = (unsigned char)cp[1];
oi->data[ix + iy*im->x].blue = (unsigned char)cp[2];


}
});
return oi;

}
return NULL;
}

static PPMImage *readPPM(const char *filename)
{
char buff[16];
PPMImage *img;
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
while (getc(fp) != '\n') ;
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

if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
exit(1);
}

while (fgetc(fp) != '\n') ;
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

fprintf(fp, "# Created by %s\n",CREATOR);

fprintf(fp, "%d %d\n",img->x,img->y);

fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

fwrite(img->data, 3 * img->x, img->y, fp);
fclose(fp);
}

void changeColorPPM(PPMImage *img)
{
int i;
if(img){

for(i=0; i < img->x * img->y; i++){
img->data[i].red=RGB_COMPONENT_COLOR-img->data[i].red;
img->data[i].green=RGB_COMPONENT_COLOR-img->data[i].green;
img->data[i].blue=RGB_COMPONENT_COLOR-img->data[i].blue;
}
}
}

double average[3*3] = {
1.0, 1.0, 1.0,
1.0, 1.0, 1.0,
1.0, 1.0, 1.0,
};

double blur[5*5] = {
0.0, 1.0, 2.0, 1.0, 0.0,
1.0, 4.0, 8.0, 4.0, 1.0,
2.0, 8.0, 16.0, 8.0, 2.0,
1.0, 4.0, 8.0, 4.0, 1.0,
0.0, 1.0, 2.0, 1.0, 0.0,
};


double sharpen [3*3] = {
-1.0, -1.0, -1.0,
-1.0, 9.0, -1.0,
-1.0, -1.0, -1.0
};

int main() {
tbb::tick_count start_time1, end_time1;

PPMImage *image;

PPMImage *afterImage1;
image = readPPM("carbajoFr.ppm");
cout.precision(3);

start_time1 = tbb::tick_count::now();
afterImage1 = filter(image, sharpen, 1, 1.0, 0.0);
end_time1 = tbb::tick_count::now();
cout << (end_time1 - start_time1).seconds() << endl;

writePPM("filter1.ppm", afterImage1);


printf("Press any key...");
getchar();
}
