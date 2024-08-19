

#include "RadixSort.h"
#include "RadixSort_kernels.cpp"

void radixSortBlocksKeysOnly(
unsigned int* d_keys, 
unsigned int* d_tempKeys, 
const unsigned int nbits, 
const unsigned int startbit, 
const unsigned int numElements)
{
unsigned int totalBlocks = numElements/4/CTA_SIZE;

#ifdef DEBUG
printf("nbits: %d startbit: %d\n", nbits, startbit);
#endif

#pragma omp target teams num_teams(totalBlocks) thread_limit(CTA_SIZE)
{
unsigned int numtrue[1];
unsigned int sMem[4*CTA_SIZE];
#pragma omp parallel 
{

int localId = omp_get_thread_num();
int localSize = omp_get_num_threads();
int globalId = omp_get_team_num() * localSize + localId;

uint4 key = reinterpret_cast<uint4*>(d_keys)[globalId];

#pragma omp barrier


for(unsigned int shift = startbit; shift < (startbit + nbits); ++shift)
{
uint4 lsb;
lsb.x = !((key.x >> shift) & 0x1);
lsb.y = !((key.y >> shift) & 0x1);
lsb.z = !((key.z >> shift) & 0x1);
lsb.w = !((key.w >> shift) & 0x1);

uint4 r;

r = rank4(lsb, sMem, numtrue);

sMem[(r.x & 3) * localSize + (r.x >> 2)] = key.x;
sMem[(r.y & 3) * localSize + (r.y >> 2)] = key.y;
sMem[(r.z & 3) * localSize + (r.z >> 2)] = key.z;
sMem[(r.w & 3) * localSize + (r.w >> 2)] = key.w;
#pragma omp barrier

key.x = sMem[localId];
key.y = sMem[localId +     localSize];
key.z = sMem[localId + 2 * localSize];
key.w = sMem[localId + 3 * localSize];

#pragma omp barrier
}

reinterpret_cast<uint4*>(d_tempKeys)[globalId] = key;  
}
}
}

void findRadixOffsets(
unsigned int* d_tempKeys, 
unsigned int* d_counters, 
unsigned int* d_blockOffsets, 
const unsigned int startbit, 
const unsigned int numElements)
{
unsigned int totalBlocks = numElements/2/CTA_SIZE;


#pragma omp target teams num_teams(totalBlocks) thread_limit(CTA_SIZE)
{
unsigned int  sStartPointers[16];
unsigned int  sRadix1[2*CTA_SIZE];
#pragma omp parallel 
{
unsigned int groupId = omp_get_team_num();
unsigned int localId = omp_get_thread_num();
unsigned int groupSize = omp_get_num_threads();
unsigned int globalId = groupId * groupSize + localId;

uint2 radix2 = reinterpret_cast<uint2*>(d_tempKeys)[globalId];

sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

if(localId < 16) 
{
sStartPointers[localId] = 0; 
}
#pragma omp barrier

if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
{
sStartPointers[sRadix1[localId]] = localId;
}
if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1]) 
{
sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
}
#pragma omp barrier

if(localId < 16) 
{
d_blockOffsets[groupId*16 + localId] = sStartPointers[localId];
}
#pragma omp barrier

if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
{
sStartPointers[sRadix1[localId - 1]] = 
localId - sStartPointers[sRadix1[localId - 1]];
}
if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] ) 
{
sStartPointers[sRadix1[localId + groupSize - 1]] = 
localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
}

if(localId == groupSize - 1) 
{
sStartPointers[sRadix1[2 * groupSize - 1]] = 
2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
}
#pragma omp barrier

if(localId < 16) 
{
d_counters[localId * totalBlocks + groupId] = sStartPointers[localId];
}
}
}

}

void reorderDataKeysOnly(
unsigned int* d_keys, 
unsigned int* d_tempKeys, 
unsigned int* d_blockOffsets, 
unsigned int* d_countersSum, 
unsigned int* d_counters, 
const unsigned int startbit, 
const unsigned int numElements)
{
unsigned int totalBlocks = numElements/2/CTA_SIZE;

#pragma omp target teams num_teams(totalBlocks) thread_limit(CTA_SIZE)
{
unsigned int sOffsets[16];
unsigned int sBlockOffsets[16];
uint2 sKeys2[CTA_SIZE];
#pragma omp parallel 
{

unsigned int *sKeys1 = (unsigned int*)sKeys2;

unsigned int groupId = omp_get_team_num();
unsigned int localId = omp_get_thread_num();
unsigned int groupSize = omp_get_num_threads();
unsigned int globalId = groupId * groupSize + localId;

sKeys2[localId] = reinterpret_cast<uint2*>(d_tempKeys)[globalId];

if(localId < 16)
{
sOffsets[localId]      = d_countersSum[localId * totalBlocks + groupId];
sBlockOffsets[localId] = d_blockOffsets[groupId * 16 + localId];
}
#pragma omp barrier

unsigned int radix = (sKeys1[localId] >> startbit) & 0xF;
unsigned int globalOffset = sOffsets[radix] + localId - sBlockOffsets[radix];

if (globalOffset < numElements)
{
d_keys[globalOffset]   = sKeys1[localId];
}

radix = (sKeys1[localId + groupSize] >> startbit) & 0xF;
globalOffset = sOffsets[radix] + localId + groupSize - sBlockOffsets[radix];

if (globalOffset < numElements)
{
d_keys[globalOffset]   = sKeys1[localId + groupSize];
}
}
}
}

void radixSortStepKeysOnly(unsigned int* d_keys, 
unsigned int* d_tempKeys, 
unsigned int* d_counters, 
unsigned int* d_blockOffsets, 
unsigned int* d_countersSum, 
unsigned int* d_buffer, 
const unsigned int nbits, 
const unsigned int startbit, 
const unsigned int numElements, 
const unsigned int batchSize )
{
radixSortBlocksKeysOnly(d_keys, d_tempKeys, nbits, startbit, numElements);

#ifdef DEBUG
#pragma omp target update from(d_tempKeys[0:numElements])
for (int i = 0; i < numElements; i++) printf("temp key %d: %x\n", i, d_tempKeys[i]);
#endif

findRadixOffsets(d_tempKeys, d_counters, d_blockOffsets, startbit, numElements);

#ifdef DEBUG
unsigned int numBlocks = (numElements / (CTA_SIZE * 4));
#pragma omp target update from(d_counters[0:WARP_SIZE*numBlocks])
for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("counters %d: %x\n", i, d_counters[i]);
#endif

scanExclusiveLarge(d_countersSum, d_counters, d_buffer,
batchSize, numElements/2/CTA_SIZE*16, numElements);

#ifdef DEBUG
#pragma omp target update from(d_countersSum[0:WARP_SIZE*numBlocks])
for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("countersSum %d: %x\n", i, d_countersSum[i]);
#endif

reorderDataKeysOnly(d_keys, d_tempKeys, d_blockOffsets, d_countersSum, d_counters, startbit, numElements);

}


void radixSortKeys(unsigned int* d_keys, 
unsigned int* d_tempKeys, 
unsigned int* d_counters, 
unsigned int* d_blockOffsets, 
unsigned int* d_countersSum, 
unsigned int* d_buffer, 
const unsigned int numElements, 
const unsigned int keyBits,
const unsigned int batchSize)
{
int i = 0;
while (keyBits > i*bitStep) 
{
radixSortStepKeysOnly(d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum,
d_buffer, bitStep, i*bitStep, numElements, batchSize);
i++;
}
}
