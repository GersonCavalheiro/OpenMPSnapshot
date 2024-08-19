#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
int omp_get_thread_num(void) { return 0; }
int omp_get_num_threads(void) { return 1; }
#endif
#define NXPROB      3600                
#define NYPROB      3600                
#define STEPS       100			
struct Parms {
float cx;
float cy;
} parms = {0.1, 0.1};
int main(int argc, char *argv[])
{
void	inidat(),
update();
float *table_u;			
int	offset,			
ix, iy, iz, it;		
double 	total_time = 0;		
struct timeval start, end;
gettimeofday(&start, NULL);
table_u = (float*) malloc((2 * NXPROB * NYPROB) * sizeof(float));
if (table_u == NULL) {
printf("Main ERROR: Allocation memory.\n");
return(EXIT_FAILURE);
}
for (iz = 0; iz < 2; iz++) {
for (ix = 0; ix < NXPROB; ix++) {
for (iy = 0; iy < NYPROB; iy++) {
offset = iz * NXPROB * NYPROB + ix * NYPROB + iy;
*(table_u + offset) = 0.0;
}
}
}
inidat(NXPROB, NYPROB, table_u);
iz = 0;
for (it = 1; it <= STEPS; it++){
update(1, NXPROB - 2 + iz*NXPROB*NYPROB, table_u + (1-iz)*NXPROB*NYPROB);
iz = 1 - iz;
}
gettimeofday(&end, NULL);
total_time = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
printf("Number of steps: %d \nTime elapsed: %lf sec\n", STEPS, total_time);
free(table_u);
}
void update(int start, int end, int ny, float *u1, float *u2)
{
int ix, iy;
#pragma omp parallel
{
#pragma omp for
for (ix = start; ix <= end; ix++) {
for (iy = 1; iy <= ny-2; iy++) { 
*(u2+ix*ny+iy) = *(u1+ix*ny+iy)  + 
parms.cx * (*(u1+(ix+1)*ny+iy) +
*(u1+(ix-1)*ny+iy) - 
2.0 * *(u1+ix*ny+iy)) +
parms.cy * (*(u1+ix*ny+iy+1) +
*(u1+ix*ny+iy-1) - 
2.0 * *(u1+ix*ny+iy));
}
}
}
}
void inidat(int nx, int ny, float *u) {
int ix, iy;
for (ix = 0; ix <= nx-1; ix++) {
for (iy = 0; iy <= ny-1; iy++) {
*(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}
}
}
