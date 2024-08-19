#include <inttypes.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
struct particle { float x, y, z; };
double wtime()
{
struct timeval t;
gettimeofday(&t, NULL);
return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}
const float G = 6.67e-11;
void calculate_forces(struct particle *p, struct particle *f, float *m, int n)
{
#pragma omp parallel for
for (int i = 0; i < n - 1; i++)
{
for (int j = i + 1; j < n; j++)
{
float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) + powf(p[i].z - p[j].z, 2));
float mag = (G * m[i] * m[j]) / powf(dist, 2);
struct particle dir = {.x = p[j].x - p[i].x, .y = p[j].y - p[i].y, .z = p[j].z - p[i].z};
#pragma omp critical
{
f[i].x += mag * dir.x / dist;
f[i].y += mag * dir.y / dist;
f[i].z += mag * dir.z / dist;
f[j].x -= mag * dir.x / dist;
f[j].y -= mag * dir.y / dist;
f[j].z -= mag * dir.z / dist;
}
}
}
}
void move_particles(struct particle *p, struct particle *f, struct particle *v, float *m, int n, double dt)
{
#pragma omp for nowait
for (int i = 0; i < n; i++)
{
struct particle dv = {.x = f[i].x / m[i] * dt, .y = f[i].y / m[i] * dt, .z = f[i].z / m[i] * dt,};
struct particle dp = {.x = (v[i].x + dv.x / 2) * dt, .y = (v[i].y + dv.y / 2) * dt, .z = (v[i].z + dv.z / 2) * dt,};
v[i].x += dv.x;
v[i].y += dv.y;
v[i].z += dv.z;
p[i].x += dp.x;
p[i].y += dp.y;
p[i].z += dp.z;
f[i].x = f[i].y = f[i].z = 0;
}
}
int main(int argc, char *argv[])
{
double ttotal, tinit = 0, tforces = 0, tmove = 0;
ttotal = wtime();
int n = (argc > 1) ? atoi(argv[1]) : 10;
int threads = (argc > 2) ? atoi(argv[2]) : 4;
omp_set_num_threads(threads);
char *filename = (argc > 3) ? argv[3] : NULL;
tinit = -wtime();
struct particle *p = malloc(sizeof(*p) * n);
struct particle *f = malloc(sizeof(*f) * n);
struct particle *v = malloc(sizeof(*v) * n);
float *m = malloc(sizeof(*m) * n);
for (int i = 0; i < n; i++)
{
p[i].x = rand() / (float)RAND_MAX - 0.5;
p[i].y = rand() / (float)RAND_MAX - 0.5;
p[i].z = rand() / (float)RAND_MAX - 0.5;
v[i].x = rand() / (float)RAND_MAX - 0.5;
v[i].y = rand() / (float)RAND_MAX - 0.5;
v[i].z = rand() / (float)RAND_MAX - 0.5;
m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
f[i].x = f[i].y = f[i].z = 0;
}
tinit += wtime();
double dt = 1e-5;
for (double t = 0; t <= 1; t += dt)
{ 
tforces -= wtime();
calculate_forces(p, f, m, n); 
tforces += wtime();
tmove -= wtime();
move_particles(p, f, v, m, n, dt); 
tmove += wtime();
}
ttotal = wtime() - ttotal;
printf("# nbody (n = %d)\n", n);
printf("# Elapsed time (sec): ttotal %.6f, tinit %.6f, tforces %.6f, tmove %.6f\n", ttotal, tinit, tforces, tmove);
if (filename)
{
FILE *fout = fopen(filename, "w");
if (!fout)
{
fprintf(stderr, "Can't save file\n");
exit(EXIT_FAILURE);
}
for (int i = 0; i < n; i++)
{
fprintf(fout, "%15f %15f %15f\n", p[i].x, p[i].y, p[i].z);
}
fclose(fout);
}
free(m);
free(v);
free(f);
free(p);
return 0;
}