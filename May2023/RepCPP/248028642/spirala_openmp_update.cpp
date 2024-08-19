#include <iostream>
#include <math.h>
#include <algorithm>
#include <omp.h>

const int imageWidth = 601;
unsigned char image[imageWidth][imageWidth][3];

int ulam_get_map(int x, int y, int n)
{
x -= (n - 1) / 2;
y -= n / 2;
int mx = abs(x), my = abs(y);
int l = 2 * std::max(mx, my);
int d = y >= x ? l * 3 + x + y : l - x - y;
return pow(l - 1, 2) + d;
}

int isprime(int n)
{
int p;
for (p = 2; p * p <= n; p++)
if (n % p == 0)
return 0;
return n > 2;
}

int main()
{
omp_set_num_threads(8);
FILE *fp;
char *filename = "new1.ppm";
char *comment = "# ";
fp = fopen(filename, "wb");
fprintf(fp, "P6\n %s\n %d\n %d\n %d\n", comment, imageWidth, imageWidth, 255);

#pragma omp parallel for collapse(2) schedule(dynamic) 
for (int i = 0; i < imageWidth; i++)
{
for (int j = 0; j < imageWidth; j++)
{
bool isCelPrime = isprime(ulam_get_map(i, j, imageWidth));
if (isCelPrime)
{
image[i][j][0] = 255;
image[i][j][1] = 255;
image[i][j][2] = 255;
}
else
{
image[i][j][0] = 0;
image[i][j][1] = 0;
image[i][j][2] = 0;
}
}
}

fwrite(image, 1, 3 * imageWidth * imageWidth, fp);
fclose(fp); return 0; }
