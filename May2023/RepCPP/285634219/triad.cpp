#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "OptionParser.h"
#include "Timer.h"
#include "Utility.h"

void addBenchmarkSpecOptions(OptionParser &op)
{
;
}
void RunBenchmark(OptionParser &op)
{
const bool verbose = op.getOptionBool("verbose");
const int n_passes = op.getOptionInt("passes");

const int nSizes = 9;
const int blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
const int memSize = 16384;
const int numMaxFloats = 1024 * memSize / sizeof(float);
const int halfNumFloats = numMaxFloats / 2;
const int maxBlockSize = blockSizes[nSizes - 1] * 1024;

srand48(8650341L);
float *h_mem = (float*) malloc (sizeof(float) * numMaxFloats);

float *A0 = (float*) malloc (sizeof(float) * maxBlockSize);
float *B0 = (float*) malloc (sizeof(float) * maxBlockSize);
float *C0 = (float*) malloc (sizeof(float) * maxBlockSize);
float *A1 = (float*) malloc (sizeof(float) * maxBlockSize);
float *B1 = (float*) malloc (sizeof(float) * maxBlockSize);
float *C1 = (float*) malloc (sizeof(float) * maxBlockSize);

const float scalar = 1.75f;
const int blockSize = 128;

#pragma omp target data map(alloc: A0[0:maxBlockSize],\
B0[0:maxBlockSize],\
C0[0:maxBlockSize],\
A1[0:maxBlockSize],\
B1[0:maxBlockSize],\
C1[0:maxBlockSize])
{
for (int i = 0; i < nSizes; ++i)
{
for (int j=0; j<numMaxFloats; ++j)
C0[j] = C1[j] = 0.0f;

for (int j = 0; j < halfNumFloats; ++j) {
A0[j] = A0[halfNumFloats + j] = B0[j] = B0[halfNumFloats + j] = \
A1[j] = A1[halfNumFloats + j] = B1[j] = B1[halfNumFloats + j] \
= (float) (drand48() * 10.0);
}

int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
if (verbose) {
std::cout << ">> Executing Triad with vectors of length "
<< numMaxFloats << " and block size of "
<< elemsInBlock << " elements." << "\n";
std::cout << "Block: " << blockSizes[i] << "KB" << "\n";
}

int crtIdx = 0;

int TH = Timer::Start();

for (int pass = 0; pass < n_passes; ++pass)
{
#pragma omp target update to (A0[0:elemsInBlock]) nowait
#pragma omp target update to (B0[0:elemsInBlock]) nowait

#pragma omp target teams distribute parallel for thread_limit(blockSize) nowait
for (int gid = 0; gid < elemsInBlock; gid++) 
C0[gid] = A0[gid] + scalar*B0[gid];

if (elemsInBlock < numMaxFloats)
{
#pragma omp target update to (A1[elemsInBlock:2*elemsInBlock]) nowait
#pragma omp target update to (B1[elemsInBlock:2*elemsInBlock]) nowait
}

int blockIdx = 1;
unsigned int currStream = 1;
while (crtIdx < numMaxFloats)
{
currStream = blockIdx & 1;
if (currStream)
{
#pragma omp target update from(C0[crtIdx:crtIdx+elemsInBlock]) nowait
}
else
{
#pragma omp target update from(C1[crtIdx:crtIdx+elemsInBlock]) nowait
}

crtIdx += elemsInBlock;

if (crtIdx < numMaxFloats)
{
if (currStream)
{
#pragma omp target teams distribute parallel for thread_limit(blockSize) nowait
for (int gid = 0; gid < elemsInBlock; gid++) 
C1[crtIdx+gid] = A1[crtIdx+gid] + scalar*B1[crtIdx+gid];
}
else
{
#pragma omp target teams distribute parallel for thread_limit(blockSize) nowait
for (int gid = 0; gid < elemsInBlock; gid++) 
C0[crtIdx+gid] = A0[crtIdx+gid] + scalar*B0[crtIdx+gid];
}
}

if (crtIdx+elemsInBlock < numMaxFloats)
{
if (currStream)
{
#pragma omp target update to (A0[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
#pragma omp target update to (B0[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
}
else
{
#pragma omp target update to (A1[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
#pragma omp target update to (B1[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
}
}
blockIdx += 1;
currStream = !currStream;
}
} 

double time = Timer::Stop(TH, "Warning: no thread synchronization");

double triad = ((double)numMaxFloats*2.0*n_passes) / (time*1e9);
if (verbose) std::cout << "Average TriadFlops " << triad << " GFLOPS/s\n";

double bdwth = ((double)numMaxFloats*sizeof(float)*3.0*n_passes)
/ (time*1000.*1000.*1000.);
if (verbose) std::cout << "Average TriadBdwth " << bdwth << " GB/s\n";

bool ok = true;
for (int j=0; j<numMaxFloats; j=j+elemsInBlock) {
if (((j / elemsInBlock) & 1) == 0) {
memcpy(h_mem+j, C0+j, elemsInBlock*sizeof(float));
}
else {
memcpy(h_mem+j, C1+j, elemsInBlock*sizeof(float));
}
}

for (int j=0; j<halfNumFloats; ++j)
{
if (h_mem[j] != h_mem[j+halfNumFloats])
{
std::cout << "hostMem[" << j << "]=" << h_mem[j]
<< " is different from its twin element hostMem["
<< (j+halfNumFloats) << "]: "
<< h_mem[j+halfNumFloats] << "stopping check\n";
ok = false;
break;
}
}

if (ok)
std::cout << "PASS\n";
else
std::cout << "FAIL\n";
}
}

free(h_mem);
free(A0);
free(B0);
free(C0);
free(A1);
free(B1);
free(C1);
}
