#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#define MATCH(s) (!strcmp(argv[ac], (s)))
int MeshPlot(int t, int m, int n, char **mesh);
double real_rand();
int seed_rand(long sd);
static char **currWorld=NULL, **nextWorld=NULL, **tmesh=NULL;
static int maxiter = 200; 
static int population[2] = {0,0}; 
int nx = 100;      
int ny = 100;      
static int w_update = 0;
static int w_plot = 1;
double getTime();
extern FILE *gnu;
void allocate_worlds()
{
currWorld = (char**)malloc(sizeof(char*)*nx + sizeof(char)*nx*ny);
nextWorld = (char**)malloc(sizeof(char*)*nx + sizeof(char)*nx*ny);
int i;
for(i=0;i<nx;i++){
currWorld[i] = (char*)(currWorld+nx) + i*ny;
nextWorld[i] = (char*)(nextWorld+nx) + i*ny;
}
};
void set_ghost_cells()
{
int i;
for(i=1;i<nx-1;i++)
{
currWorld[i][0]=0;
currWorld[i][ny-1]=0;
currWorld[0][i]=0;
currWorld[nx-1][i]=0;
nextWorld[i][0]=0;
nextWorld[i][ny-1]=0;
nextWorld[0][i]=0;
nextWorld[nx-1][i]=0;
}
currWorld[0][0] = currWorld[0][ny-1] = currWorld[nx-1][ny-1] = currWorld[nx-1][0] = 0;
nextWorld[0][0] = nextWorld[0][ny-1] = nextWorld[nx-1][ny-1] = nextWorld[nx-1][0] = 0;
};
int main(int argc, char **argv){
float prob = 0.5;   
long seedVal = 0;
int game = 0;
int s_step = 0;
int numthreads = 1;
int disable_display= 0; 
int ac;
for(ac=1;ac<argc;ac++)
{
if(MATCH("-n")) {nx = atoi(argv[++ac]);}
else if(MATCH("-i")) {maxiter = atoi(argv[++ac]);}
else if(MATCH("-t"))  {numthreads = atof(argv[++ac]);}
else if(MATCH("-p"))  {prob = atof(argv[++ac]);}
else if(MATCH("-s"))  {seedVal = atof(argv[++ac]);}
else if(MATCH("-step"))  {s_step = 1;}
else if(MATCH("-d"))  {disable_display = 1;}
else if(MATCH("-g"))  {game = atoi(argv[++ac]);}
else {
printf("Usage: %s [-n < meshpoints>] [-i <iterations>] [-s seed] [-p prob] [-t numthreads] [-step] [-g <game #>] [-d]\n",argv[0]);
return(-1);
}
}
int rs = seed_rand(seedVal);
nx = nx+2;
ny = nx; 
printf("probability: %f\n",prob);
printf("Random # generator seed: %d\n", rs);
omp_set_num_threads(numthreads);
omp_set_nested(1);
allocate_worlds();
set_ghost_cells();
int i,j, sum = 0;
if(game == 0){
for(i=1;i<nx-1;i++){
for(j=1;j<ny-1;j++) {
currWorld[i][j] = (real_rand() < prob);
sum += currWorld[i][j];
}
}
population[w_plot] += sum;
} else if(game == 1){
printf("2x2 Block, still life\n");
int nx2 = nx/2;
int ny2 = ny/2;
currWorld[nx2+1][ny2+1] = currWorld[nx2][ny2+1] = currWorld[nx2+1][ny2] = currWorld[nx2][ny2] = 1;
population[w_plot] = 4;
} else if(game == 2){
printf("Glider (spaceship)\n");
} else {
printf("Unknown game %d\n",game);
return 1;
}
if(!disable_display)
MeshPlot(0,nx,ny,currWorld);
double t0 = getTime();
int t;
#pragma omp parallel num_threads(2) if(numthreads > 1)
{    
#pragma omp single
{   
for(t = 0; t < maxiter; t++)
{      
#pragma omp task shared(w_update, population)
{   
int i,j, sum = 0;
#pragma omp parallel num_threads(numthreads - 1) if(numthreads > 1)
{
#pragma omp for collapse(2) private(i,j) reduction(+:sum) schedule(static)
for(i=1;i<nx-1;i++){
for(j=1;j<ny-1;j++) {
int nn = currWorld[i+1][j] + currWorld[i-1][j] + 
currWorld[i][j+1] + currWorld[i][j-1] + 
currWorld[i+1][j+1] + currWorld[i-1][j-1] + 
currWorld[i-1][j+1] + currWorld[i+1][j-1];
nextWorld[i][j] = currWorld[i][j] ? (nn == 2 || nn == 3) : (nn == 3);
sum += nextWorld[i][j];
}
}
}
population[w_update] += sum;
}
#pragma omp task shared(disable_display, t, nx, ny, currWorld)
{   
if(!disable_display)
MeshPlot(t,nx,ny,currWorld);
}
#pragma omp taskwait
tmesh = nextWorld;
nextWorld = currWorld;
currWorld = tmesh;
if(s_step)
{
printf("Completed iteration: %d\n",t);
printf("Press enter to continue...\n");
getchar();
}
}
}
}
double t1 = getTime(); 
printf("Running time for the iterations: %f sec.\n",t1-t0);
printf("Press enter to end.\n");
getchar();
if(gnu != NULL)
pclose(gnu);
free(nextWorld);
free(currWorld);
return 0;
}
