

#define WARP_SIZE 32

#pragma omp declare target 

unsigned int scanwarp(unsigned int val, volatile unsigned int* sData, const int maxlevel)
{
int localId = omp_get_thread_num();
int idx = 2 * localId - (localId & (WARP_SIZE - 1));
sData[idx] = 0;
idx += WARP_SIZE;
sData[idx] = val;     

if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; }
if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; }
if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; }
if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; }
if (4 <= maxlevel) { sData[idx] += sData[idx -16]; }

return sData[idx] - val;  
}


uint4 scan4(const uint4 idata, unsigned int* ptr)
{    

unsigned int idx = omp_get_thread_num();

uint4 val4 = idata;
unsigned int sum[3];
sum[0] = val4.x;
sum[1] = val4.y + sum[0];
sum[2] = val4.z + sum[1];

unsigned int val = val4.w + sum[2];

val = scanwarp(val, ptr, 4);
#pragma omp barrier

if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
{
ptr[idx >> 5] = val + val4.w + sum[2];
}
#pragma omp barrier

if (idx < WARP_SIZE)
ptr[idx] = scanwarp(ptr[idx], ptr, 2);

#pragma omp barrier

val += ptr[idx >> 5];

val4.x = val;
val4.y = val + sum[0];
val4.z = val + sum[1];
val4.w = val + sum[2];

return val4;
}

uint4 rank4(const uint4 preds, unsigned int* sMem, unsigned int* numtrue)
{
int localId = omp_get_thread_num();
int localSize = omp_get_num_threads();

uint4 address = scan4(preds, sMem);

if (localId == localSize - 1) 
{
numtrue[0] = address.w + preds.w;
}
#pragma omp barrier

uint4 rank;
int idx = localId*4;
rank.x = (preds.x) ? address.x : numtrue[0] + idx - address.x;
rank.y = (preds.y) ? address.y : numtrue[0] + idx + 1 - address.y;
rank.z = (preds.z) ? address.z : numtrue[0] + idx + 2 - address.z;
rank.w = (preds.w) ? address.w : numtrue[0] + idx + 3 - address.w;

return rank;
}


#pragma omp end declare target 
