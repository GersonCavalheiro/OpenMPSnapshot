

#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "FDTD3dGPU.h"
#include "shrUtils.h"

bool fdtdGPU(float *output, float *input, const float *coeff, 
const int dimx, const int dimy, const int dimz, const int radius, 
const int timesteps, const int argc, const char **argv)
{
bool ok = true;
const int         outerDimx  = dimx + 2 * radius;
const int         outerDimy  = dimy + 2 * radius;
const int         outerDimz  = dimz + 2 * radius;
const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
size_t            teamSize[2];
size_t            threadSize[2];

const int padding = (128 / sizeof(float)) - radius;
const size_t paddedVolumeSize = volumeSize + padding;

float* bufferOut = (float*) malloc (paddedVolumeSize * sizeof(float));
float* bufferIn = (float*) malloc (paddedVolumeSize * sizeof(float));

memcpy(bufferIn + padding, input, volumeSize * sizeof(float));
memcpy(bufferOut + padding, input, volumeSize * sizeof(float)); 

size_t userWorkSize = 256;

threadSize[0] = k_localWorkX;
threadSize[1] = userWorkSize / k_localWorkX;
teamSize[0] = (unsigned int)ceil((float)dimx / threadSize[0]);
teamSize[1] = (unsigned int)ceil((float)dimy / threadSize[1]);

int teamX = teamSize[0];
int teamY = teamSize[1];
int numTeam = teamX * teamY;

shrLog(" set thread size to %dx%d\n", threadSize[0], threadSize[1]);
shrLog(" set team size to %dx%d\n", teamSize[0], teamSize[1]);

shrLog(" GPU FDTD loop\n");

#pragma omp target data map(to: bufferIn[0:paddedVolumeSize], \
bufferOut[0:paddedVolumeSize], \
coeff[0:radius+1]) 
{
auto start = std::chrono::steady_clock::now();

for (int it = 0 ; it < timesteps ; it++)
{
#pragma omp target teams num_teams(numTeam) thread_limit(userWorkSize)
{
float tile [localWorkMaxY + 2*k_radius_default][localWorkMaxX + 2*k_radius_default];
#pragma omp parallel 
{
bool valid = true;
const int ltidx = omp_get_thread_num() % k_localWorkX;
const int ltidy = omp_get_thread_num() / k_localWorkX;
const int workx = k_localWorkX;
const int worky = userWorkSize / k_localWorkX;
const int gtidx = (omp_get_team_num() % teamX) * workx + ltidx;
const int gtidy = (omp_get_team_num() / teamX) * worky + ltidy;

const int stride_y = dimx + 2 * k_radius_default;
const int stride_z = stride_y * (dimy + 2 * k_radius_default);

int inputIndex  = 0;
int outputIndex = 0;

inputIndex += k_radius_default * stride_y + k_radius_default + padding;

inputIndex += gtidy * stride_y + gtidx;

float infront[k_radius_default];
float behind[k_radius_default];
float current;

const int tx = ltidx + k_radius_default;
const int ty = ltidy + k_radius_default;

if (gtidx >= dimx) valid = false;
if (gtidy >= dimy) valid = false;

for (int i = k_radius_default - 2 ; i >= 0 ; i--)
{
behind[i] = bufferIn[inputIndex];
inputIndex += stride_z;
}

current = bufferIn[inputIndex];
outputIndex = inputIndex;
inputIndex += stride_z;

for (int i = 0 ; i < k_radius_default ; i++)
{
infront[i] = bufferIn[inputIndex];
inputIndex += stride_z;
}

for (int iz = 0 ; iz < dimz ; iz++)
{
for (int i = k_radius_default - 1 ; i > 0 ; i--)
behind[i] = behind[i - 1];
behind[0] = current;
current = infront[0];
for (int i = 0 ; i < k_radius_default - 1 ; i++)
infront[i] = infront[i + 1];
infront[k_radius_default - 1] = bufferIn[inputIndex];

inputIndex  += stride_z;
outputIndex += stride_z;
#pragma omp barrier


if (ltidy < k_radius_default)
{
tile[ltidy][tx]                  = bufferIn[outputIndex - k_radius_default * stride_y];
tile[ltidy + worky + k_radius_default][tx] = bufferIn[outputIndex + worky * stride_y];
}
if (ltidx < k_radius_default)
{
tile[ty][ltidx]                  = bufferIn[outputIndex - k_radius_default];
tile[ty][ltidx + workx + k_radius_default] = bufferIn[outputIndex + workx];
}
tile[ty][tx] = current;
#pragma omp barrier

float value = coeff[0] * current;
for (int i = 1 ; i <= k_radius_default ; i++)
{
value += coeff[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + 
tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
}

if (valid) bufferOut[outputIndex] = value;
}
}
}

float* tmp = bufferIn;
bufferIn = bufferOut;
bufferOut = tmp;
}

auto end = std::chrono::steady_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / timesteps);

#pragma omp target update from (bufferIn[0:paddedVolumeSize])
}

memcpy(output, bufferIn+padding, volumeSize*sizeof(float));
free(bufferIn);
free(bufferOut);
return ok;
}
