
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <omp.h>

const int triangleHeight = 1024;
const int levelLimit = 3;

unsigned char image[triangleHeight][triangleHeight][3];

void draw_triangle(int x, int y, int level)
{
std::cout << "I'm in " << omp_get_thread_num() << std::endl; 
int length = triangleHeight / pow(2, level);
for (int i = y; i < y + length; i++)
{
for (int j = x; j < x + length; j++)
{
if (i - y == j - x || i == y + length - 1 || j == x)
{
image[i][j][0] = 255;
image[i][j][1] = 255;
image[i][j][2] = 255;
}
}
}
if (level < levelLimit)
{
level++;
#pragma omp task
draw_triangle(x, y, level);
#pragma omp task
draw_triangle(x, y + length / 2, level);
#pragma omp task
draw_triangle(x + length / 2, y + length / 2, level);
#pragma omp taskwait
}
}

int main()
{
omp_set_num_threads(4);
FILE *fp;
char *filename = "new1.ppm";
char *comment = "# ";
fp = fopen(filename, "wb");
fprintf(fp, "P6\n %s\n %d\n %d\n %d\n", comment, triangleHeight, triangleHeight, 255);

#pragma omp parallel
{
#pragma omp single    
draw_triangle(0, 0, 0);
}

fwrite(image, 1, 3 * triangleHeight * triangleHeight, fp);
fclose(fp);
return 0;
}

