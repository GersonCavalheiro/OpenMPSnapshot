

#include "Scan.h"
#include "Scan_kernels.cpp"


void scanExclusiveLocal1(
unsigned int* d_Dst,
unsigned int* d_Src,
const unsigned int n,
const unsigned int size)
{
size_t localWorkSize = WORKGROUP_SIZE;
size_t globalWorkSize = (n * size) / 4;
unsigned int totalBlocks = globalWorkSize/localWorkSize;

#pragma omp target teams num_teams(totalBlocks) thread_limit(localWorkSize)
{
unsigned int l_Data[2 * WORKGROUP_SIZE];
#pragma omp parallel 
{
int i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();

uint4 idata4 = reinterpret_cast<uint4*>(d_Src)[i];

uint4 odata4 = scan4Exclusive(idata4, l_Data, size);

reinterpret_cast<uint4*>(d_Dst)[i] = odata4;
}
}
}

void scanExclusiveLocal2(
unsigned int* d_Buf,
unsigned int* d_Dst,
unsigned int* d_Src,
const unsigned int n,
const unsigned int size)
{
const unsigned int elements = n * size;
size_t localWorkSize = WORKGROUP_SIZE;
size_t globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);
unsigned int totalBlocks = globalWorkSize/localWorkSize;

#pragma omp target teams num_teams(totalBlocks) thread_limit(localWorkSize)
{
unsigned int l_Data[2 * WORKGROUP_SIZE];
#pragma omp parallel 
{
unsigned int data = 0;
int i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();
if(i < elements)
data = d_Dst[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * i] + 
d_Src[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * i];

unsigned int odata = scan1Exclusive(data, l_Data, size);

if(i < elements) d_Buf[i] = odata;
}
}
}

void uniformUpdate(
unsigned int* d_Dst,
unsigned int* d_Buf,
const unsigned int n)
{
#pragma omp target teams num_teams(n) thread_limit(WORKGROUP_SIZE)
{
unsigned int buf[1];
#pragma omp parallel 
{
int localId = omp_get_thread_num();
int groupId = omp_get_team_num();
int i = groupId * omp_get_num_threads() + localId;

uint4 data4 = reinterpret_cast<uint4*>(d_Dst)[i];
if(localId == 0)
buf[0] = d_Buf[groupId];

#pragma omp barrier
data4 += {buf[0], buf[0], buf[0], buf[0]};

reinterpret_cast<uint4*>(d_Dst)[i] = data4;
}
}
}

void scanExclusiveLarge(
unsigned int* d_Dst,
unsigned int* d_Src,
unsigned int* d_Buf,
const unsigned int batchSize,
const unsigned int arrayLength,
const unsigned int numElements)
{

#ifdef DEBUG
unsigned int CTA_SIZE = 128;
unsigned int WARP_SIZE = 32;
assert(numElements == arrayLength / 16 * CTA_SIZE * 2);
unsigned int numBlocks = (numElements / (CTA_SIZE * 4));
#endif


scanExclusiveLocal1(
d_Dst,
d_Src,
(batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
4 * WORKGROUP_SIZE
);

#ifdef DEBUG
#pragma omp target update from(d_Dst[0:WARP_SIZE*numBlocks])
for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("local1 %d: %x\n", i, d_Dst[i]);
#endif

scanExclusiveLocal2(
d_Buf,
d_Dst,
d_Src,
batchSize,
arrayLength / (4 * WORKGROUP_SIZE)
);

#ifdef DEBUG
#pragma omp target update from(d_Buf[0:arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE])
for (int i = 0; i < arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE; i++) printf("local2 %d: %x\n", i, d_Buf[i]);
#endif

uniformUpdate(
d_Dst,
d_Buf,
(batchSize * arrayLength) / (4 * WORKGROUP_SIZE)
);

#ifdef DEBUG
#pragma omp target update from(d_Dst[0:WARP_SIZE*numBlocks])
for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("uniform %d: %x\n", i, d_Dst[i]);
#endif

}
