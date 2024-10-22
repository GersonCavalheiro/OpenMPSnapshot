#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "helper.h"
#include <omp.h>
void printarr(double *a, int n, int rank) {
char name[20];
sprintf(name, "heat_%d.svg", rank);
FILE *fp = fopen(name, "w");
const int size = 5;
fprintf(fp, "<html>\n<body>\n<svg xmlns=\"http:
fprintf(fp, "\n<rect x=\"0\" y=\"0\" width=\"%i\" height=\"%i\" style=\"stroke-width:1;fill:rgb(0,0,0);stroke:rgb(0,0,0)\"/>", size*n, size*n);
for(int i=1; i<n+1; ++i)
for(int j=1; j<n+1; ++j) {
int rgb = (a[map(i,j,n+2)] > 0) ? rgb = (int)round(255.0*a[map(i,j,n+2)]) : 0.0;
if(rgb>255) rgb=255;
if(rgb) fprintf(fp, "\n<rect x=\"%i\" y=\"%i\" width=\"%i\" height=\"%i\" style=\"stroke-width:1;fill:rgb(%i,0,0);stroke:rgb(%i,0,0)\"/>", size*(i-1), size*(j-1), size, size, rgb, rgb);
}
fprintf(fp, "</svg>\n</body>\n</html>");
fclose(fp);
}
double calculate_total_heat(double *h_new, int n)
{
double heat = 0.0; 
for (int i = 1; i < n + 1; ++i)
{
for (int j = 1; j < n + 1; ++j)
{
heat += h_new[map(i, j, n+2)];
}
}
return heat;
}
double calculate_total_heat_omp(double *h_new, int n, int num_threads)
{
double heat = 0.0; 
for (int i = 1; i < n + 1; ++i)
{
for (int j = 1; j < n + 1; ++j)
{
heat += h_new[map(i, j, n+2)];
}
}
return heat;
}
