


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "WKFUtils.h"

#define MAXATOMS 4000

#define UNROLLX       8
#define UNROLLY       1
#define BLOCKSIZEX    8
#define BLOCKSIZEY    8 
#define BLOCKSIZE    BLOCKSIZEX * BLOCKSIZEY

struct float4 {
float x;
float y;
float z;
float w;
};

struct int3 {
int x;
int y;
int z;
};

int copyatoms(float *atoms, int count, float zplane, float4* atominfo) {

if (count > MAXATOMS) {
printf("Atom count exceeds constant buffer storage capacity\n");
return -1;
}

int i;
for (i=0; i<count; i++) {
atominfo[i].x = atoms[i*4    ];
atominfo[i].y = atoms[i*4 + 1];
float dz = zplane - atoms[i*4 + 2];
atominfo[i].z  = dz*dz;
atominfo[i].w = atoms[i*4 + 3];
}
#pragma omp target update to(atominfo[0:count])
return 0;
}


int initatoms(float **atombuf, int count, int3 volsize, float gridspacing) {
float4 size;
int i;
float *atoms;
srand(2);

atoms = (float *) malloc(count * 4 * sizeof(float));
*atombuf = atoms;

size.x = gridspacing * volsize.x;
size.y = gridspacing * volsize.y;
size.z = gridspacing * volsize.z;

for (i=0; i<count; i++) {
int addr = i * 4;
atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  
}  

return 0;
}


int main(int argc, char** argv) {
float *energy = NULL;
float *atoms = NULL;
int3 volsize;
wkf_timerhandle runtimer, mastertimer, copytimer, hostcopytimer;
float copytotal, runtotal, mastertotal, hostcopytotal;
const char *statestr = "|/-\\.";
int state=0;

printf("GPU accelerated coulombic potential microbenchmark\n");
printf("--------------------------------------------------------\n");
printf("  Single-threaded single-device test run.\n");

int atomcount = 1000000;

volsize.x = 768;
volsize.y = 768;
volsize.z = 1;

float gridspacing = 0.1f;


runtimer = wkf_timer_create();
mastertimer = wkf_timer_create();
copytimer = wkf_timer_create();
hostcopytimer = wkf_timer_create();
copytotal = 0;
runtotal = 0;
hostcopytotal = 0;

printf("Grid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);
printf("Running kernel(atoms:%d, gridspacing %g, z %d)\n", atomcount, gridspacing, 0);

if (initatoms(&atoms, atomcount, volsize, gridspacing))
return -1;

int volmem = volsize.x * volsize.y * volsize.z;
int volmemsz = sizeof(float) * volmem;
printf("Allocating %.2fMB of memory for output buffer...\n", volmemsz / (1024.0 * 1024.0));

energy = (float *) malloc(volmemsz);
float4* atominfo = (float4*) malloc (MAXATOMS * sizeof(float4));

printf("starting run...\n");
wkf_timer_start(mastertimer);

int iterations=0;

#pragma omp target enter data map(alloc: atominfo[0:MAXATOMS]) map(alloc: energy[0:volmem])
{
#pragma omp target teams distribute parallel for simd
for (int i = 0; i < volmem; i++)
energy[i] = 0.f;

int atomstart;
for (atomstart=0; atomstart<atomcount; atomstart+=MAXATOMS) {   
iterations++;
int runatoms;
int atomsremaining = atomcount - atomstart;
if (atomsremaining > MAXATOMS)
runatoms = MAXATOMS;
else
runatoms = atomsremaining;

printf("%c\r", statestr[state]);
fflush(stdout);
state = (state+1) & 3;

wkf_timer_start(copytimer);
if (copyatoms(atoms + 4*atomstart, runatoms, 0*gridspacing, atominfo)) 
return -1;
wkf_timer_stop(copytimer);
copytotal += wkf_timer_time(copytimer);

wkf_timer_start(runtimer);

#pragma omp target teams distribute parallel for collapse(2) 
for (unsigned int yindex = 0; yindex < volsize.y; yindex++) { 
for (unsigned int xindex = 0; xindex < volsize.x / UNROLLX; xindex++) { 
unsigned int outaddr = yindex * volsize.x + xindex; 
float coory = gridspacing * yindex;
float coorx = gridspacing * xindex;

float energyvalx1=0.0f;
float energyvalx2=0.0f;
float energyvalx3=0.0f;
float energyvalx4=0.0f;
float energyvalx5=0.0f;
float energyvalx6=0.0f;
float energyvalx7=0.0f;
float energyvalx8=0.0f;

float gridspacing_u = gridspacing * BLOCKSIZEX;

int atomid;
for (atomid=0; atomid<runatoms; atomid++) {
float dy = coory - atominfo[atomid].y;
float dyz2 = (dy * dy) + atominfo[atomid].z;

float dx1 = coorx - atominfo[atomid].x;
float dx2 = dx1 + gridspacing_u;
float dx3 = dx2 + gridspacing_u;
float dx4 = dx3 + gridspacing_u;
float dx5 = dx4 + gridspacing_u;
float dx6 = dx5 + gridspacing_u;
float dx7 = dx6 + gridspacing_u;
float dx8 = dx7 + gridspacing_u;

energyvalx1 += atominfo[atomid].w / sqrtf(dx1*dx1 + dyz2);
energyvalx2 += atominfo[atomid].w / sqrtf(dx2*dx2 + dyz2);
energyvalx3 += atominfo[atomid].w / sqrtf(dx3*dx3 + dyz2);
energyvalx4 += atominfo[atomid].w / sqrtf(dx4*dx4 + dyz2);
energyvalx5 += atominfo[atomid].w / sqrtf(dx5*dx5 + dyz2);
energyvalx6 += atominfo[atomid].w / sqrtf(dx6*dx6 + dyz2);
energyvalx7 += atominfo[atomid].w / sqrtf(dx7*dx7 + dyz2);
energyvalx8 += atominfo[atomid].w / sqrtf(dx8*dx8 + dyz2);
}

energy[outaddr             ] += energyvalx1;
energy[outaddr+1*BLOCKSIZEX] += energyvalx2;
energy[outaddr+2*BLOCKSIZEX] += energyvalx3;
energy[outaddr+3*BLOCKSIZEX] += energyvalx4;
energy[outaddr+4*BLOCKSIZEX] += energyvalx5;
energy[outaddr+5*BLOCKSIZEX] += energyvalx6;
energy[outaddr+6*BLOCKSIZEX] += energyvalx7;
energy[outaddr+7*BLOCKSIZEX] += energyvalx8;
}
}
wkf_timer_stop(runtimer);
runtotal += wkf_timer_time(runtimer);
}
printf("Done\n");
wkf_timer_stop(mastertimer);
mastertotal = wkf_timer_time(mastertimer);
}
wkf_timer_start(hostcopytimer);
#pragma omp target exit data map(from: energy[0:volmem]) map(delete:atominfo[0:MAXATOMS])
wkf_timer_stop(hostcopytimer);
hostcopytotal=wkf_timer_time(hostcopytimer);

int i, j;
for (j=0; j<8; j++) {
for (i=0; i<8; i++) {
int addr = j*volsize.x + i;
printf("[%d] %.1f ", addr, energy[addr]);
}
printf("\n");
}

printf("Final calculation required %d iterations of %d atoms\n", iterations, MAXATOMS);
printf("Copy time: %f seconds, %f per iteration\n", copytotal, copytotal / (float) iterations);
printf("Kernel time: %f seconds, %f per iteration\n", runtotal, runtotal / (float) iterations);
printf("Total time: %f seconds\n", mastertotal);
printf("Kernel invocation rate: %f iterations per second\n", iterations / mastertotal);
printf("GPU to host copy bandwidth: %gMB/sec, %f seconds total\n",
(volmemsz / (1024.0 * 1024.0)) / hostcopytotal, hostcopytotal);

double atomevalssec = ((double) volsize.x * volsize.y * volsize.z * atomcount) / (mastertotal * 1000000000.0);
printf("Efficiency metric, %g billion atom evals per second\n", atomevalssec);


printf("FP performance: %g GFLOPS\n", atomevalssec * (59.0/8.0));

free(atoms);
free(atominfo);
free(energy);
return 0;
}
