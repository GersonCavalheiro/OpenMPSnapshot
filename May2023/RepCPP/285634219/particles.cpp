

#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "particles.h"
#include "particles_kernels.cpp"


static const size_t wgSize = 64;


static size_t uSnap(size_t a, size_t b){
return ((a % b) == 0) ? a : (a - (a % b) + b);
}

void integrateSystem(
float4* d_Pos,
float4* d_Vel,
const simParams_t &params,
const float deltaTime,
const unsigned int numParticles)
{
size_t globalWorkSize = uSnap(numParticles, wgSize);

#pragma omp target teams distribute parallel for num_teams(globalWorkSize/wgSize) thread_limit(wgSize)
for (unsigned int index = 0; index < numParticles; index++) {
float4 pos = d_Pos[index];
float4 vel = d_Vel[index];

pos.w = 1.0f;
vel.w = 0.0f;

float4 g = {params.gravity.x, params.gravity.y, params.gravity.z, 0};
vel += g * deltaTime;
vel *= params.globalDamping;

pos += vel * deltaTime;


if(pos.x < -1.0f + params.particleRadius){
pos.x = -1.0f + params.particleRadius;
vel.x *= params.boundaryDamping;
}
if(pos.x > 1.0f - params.particleRadius){
pos.x = 1.0f - params.particleRadius;
vel.x *= params.boundaryDamping;
}

if(pos.y < -1.0f + params.particleRadius){
pos.y = -1.0f + params.particleRadius;
vel.y *= params.boundaryDamping;
}
if(pos.y > 1.0f - params.particleRadius){
pos.y = 1.0f - params.particleRadius;
vel.y *= params.boundaryDamping;
}

if(pos.z < -1.0f + params.particleRadius){
pos.z = -1.0f + params.particleRadius;
vel.z *= params.boundaryDamping;
}
if(pos.z > 1.0f - params.particleRadius){
pos.z = 1.0f - params.particleRadius;
vel.z *= params.boundaryDamping;
}

d_Pos[index] = pos;
d_Vel[index] = vel;
}
}

void calcHash(
unsigned int *d_Hash,
unsigned int *d_Index,
float4 *d_Pos,
const simParams_t &params,
const int numParticles)
{
size_t globalWorkSize = uSnap(numParticles, wgSize);

#pragma omp target teams distribute parallel for num_teams(globalWorkSize/wgSize) thread_limit(wgSize)
for (unsigned int index = 0; index < numParticles; index++) {
float4 p = d_Pos[index];

int4  gridPos = getGridPos(p, params);
unsigned int gridHash = getGridHash(gridPos, params);

d_Hash[index] = gridHash;
d_Index[index] = index;
}
}

void memSet(
unsigned int* d_Data,
unsigned int val,
unsigned int N)
{
size_t globalWorkSize = uSnap(N, wgSize);

#pragma omp target teams distribute parallel for num_teams(globalWorkSize/wgSize) thread_limit(wgSize)
for(unsigned int i = 0; i < N; i++) {
d_Data[i] = val;
}
}

void findCellBoundsAndReorder(
unsigned int* d_CellStart,
unsigned int* d_CellEnd,
float4 *d_ReorderedPos,
float4 *d_ReorderedVel,
unsigned int *d_Hash,
unsigned int *d_Index,
float4 *d_Pos,
float4 *d_Vel,
const unsigned int numParticles,
const unsigned int numCells)
{
memSet(d_CellStart, 0xFFFFFFFFU, numCells);
size_t globalWorkSize = uSnap(numParticles, wgSize);

#pragma omp target teams num_teams(globalWorkSize/wgSize) thread_limit(wgSize)
{
unsigned int localHash[wgSize+1];
#pragma omp parallel 
{
unsigned int hash;
int lid = omp_get_thread_num();
int index = omp_get_team_num() * wgSize + lid;

if(index < numParticles) {
hash = d_Hash[index];

localHash[lid + 1] = hash;

if(index > 0 && lid == 0) 
localHash[0] = d_Hash[index - 1];
}

#pragma omp barrier

if(index < numParticles){
if(index == 0)
d_CellStart[hash] = 0;

else{
if(hash != localHash[lid])
d_CellEnd[localHash[lid]]  = d_CellStart[hash] = index;
};

if(index == numParticles - 1)
d_CellEnd[hash] = numParticles;


unsigned int sortedIndex = d_Index[index];
float4 pos = d_Pos[sortedIndex];
float4 vel = d_Vel[sortedIndex];

d_ReorderedPos[index] = pos;
d_ReorderedVel[index] = vel;
}
}
}
}

void collide(
float4 *d_Vel,
float4 *d_ReorderedPos,
float4 *d_ReorderedVel,
unsigned int *d_Index,
unsigned int *d_CellStart,
unsigned int *d_CellEnd,
const simParams_t &params,
const unsigned int   numParticles,
const unsigned int   numCells)
{
size_t globalWorkSize = uSnap(numParticles, wgSize);

#pragma omp target teams distribute parallel for num_teams(globalWorkSize/wgSize) thread_limit(wgSize)
for (unsigned int index = 0; index < numParticles; index++) {

float4   pos = d_ReorderedPos[index];
float4   vel = d_ReorderedVel[index];
float4 force = {0, 0, 0, 0};

int4 gridPos = getGridPos(pos, params);

for(int z = -1; z <= 1; z++)
for(int y = -1; y <= 1; y++)
for(int x = -1; x <= 1; x++){
int4 t = {x, y, z, 0};
unsigned int   hash = getGridHash(gridPos + t, params);
unsigned int startI = d_CellStart[hash];

if(startI == 0xFFFFFFFFU) continue;

unsigned int endI = d_CellEnd[hash];
for(unsigned int j = startI; j < endI; j++){
if(j == index) continue;

float4 pos2 = d_ReorderedPos[j];
float4 vel2 = d_ReorderedVel[j];

force += collideSpheres(
pos, pos2,
vel, vel2,
params.particleRadius, params.particleRadius, 
params.spring, params.damping, params.shear, params.attraction
);
}
}

force += collideSpheres(
pos, {params.colliderPos.x, params.colliderPos.y, params.colliderPos.z, 0},
vel, {0, 0, 0, 0},
params.particleRadius, params.colliderRadius,
params.spring, params.damping, params.shear, params.attraction
);

d_Vel[d_Index[index]] = vel + force;
}
}
