#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#define RESOLUTION 100
#define CHUNKSIZE 10
#define DST_FILE "rdf.dat"
struct vec_t { float x, y, z; };
struct fluid_t {
struct vec_t *molecules;
int count;
float width;
};
struct fluid_t readFluid(char *fileName) {
char line[100], name[10];
struct vec_t pos;
struct fluid_t fluid;
int i, j = 0, n;
FILE *file = fopen(fileName, "r");
fscanf(file, "%d\n", &n);
fscanf(file, "%f", &(fluid.width));
fluid.molecules = (struct vec_t*) malloc(n * sizeof(struct vec_t));
fluid.count = 0;
for(i = 0; i < n; i++) {
fscanf(file, "%s %f %f %f", &name, &(pos.x), &(pos.y), &(pos.z));
if(!strcmp(name, "O2")) fluid.molecules[fluid.count++] = pos;
}
fclose(file);
return fluid;
}
void writeRdf(float *bins, struct fluid_t fluid, int res, char* fileName) {
int i;
float dr = fluid.width / 2.0F / res;
FILE *file = fopen(fileName, "w");
for(i = 0; i < res; i++) {
fprintf(file, "%f %f\n", dr * (i + 0.5F), bins[i]);
}
fclose(file);
}
float distance(struct vec_t pos1, struct vec_t pos2, float width) {
float xd, yd, zd;
xd = abs(pos2.x - pos1.x); if(xd > width / 2.0F) xd = width - xd;
yd = abs(pos2.y - pos1.y); if(yd > width / 2.0F) yd = width - yd;
zd = abs(pos2.z - pos1.z); if(zd > width / 2.0F) zd = width - zd;
return (float) sqrt(xd*xd + yd*yd + zd*zd);
}
void normalize(float *bins, struct fluid_t fluid, int res) {
int i;
float r, dv;
float dr = fluid.width / 2.0F / res;
float p = fluid.count / (float) pow(fluid.width, 3);
for(i = 0; i < res; i++) {
r = dr * (i+1);
dv = 4.0F * (float) M_PI * r * r * dr;
bins[i] /= fluid.count * dv * p;
}
}
float* rdf(struct fluid_t fluid, int res) {
int i, j, bin;
float dist;
float *p_bins, *bins = (float*) calloc(res, sizeof(float));
#pragma omp parallel shared(fluid, res, bins) private(i, j, dist, bin, p_bins)
{
p_bins = (float*) calloc(res, sizeof(float));
#pragma omp for schedule(dynamic, CHUNKSIZE)
for(i = 0; i < fluid.count; i++) {
for(j = i+1; j < fluid.count; j++) {
dist = distance(fluid.molecules[i], fluid.molecules[j], fluid.width);
if(dist < fluid.width / 2.0F) {
bin = (int) (res * dist / (fluid.width / 2.0F));
p_bins[bin] += 2.0F;
}
}
}
#pragma omp critical
for(i = 0; i < res; i++) bins[i] += p_bins[i];
free(p_bins);
}
normalize(bins, fluid, res);
return bins;
}
long timeMs() {
struct timeval time;
gettimeofday(&time, NULL);
return time.tv_sec * 1000 + time.tv_usec / 1000;
}
int main(int argc, char **argv) {
struct fluid_t fluid;
float *bins;
long t;
if(argc != 2) printf("Usage: ./rdf [src file]\n");
else {
fluid = readFluid(argv[1]);
t = timeMs();
bins = rdf(fluid, RESOLUTION);
printf("Calculated the radial distribution function of a ");
printf("%d-oxygen system in %dms.\n", fluid.count, timeMs() - t);
writeRdf(bins, fluid, RESOLUTION, DST_FILE);
free(fluid.molecules);
free(bins);
}
return 0;
}
