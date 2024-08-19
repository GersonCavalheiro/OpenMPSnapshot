



static unsigned int iSnapUp(const unsigned int dividend, const unsigned int divisor)
{
return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
}
unsigned int factorRadix2(unsigned int& log2L, unsigned int L)
{
if(!L)
{
log2L = 0;
return 0;
} else {
for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
return L;
}
}


#pragma omp declare target

#if(0)

inline unsigned int scan1Inclusive(const unsigned int idata, 
unsigned int* l_Data, 
const unsigned int size)
{
int lid = omp_get_thread_num();
unsigned int pos = 2 * lid - (lid & (size - 1));
l_Data[pos] = 0;
pos += size;
l_Data[pos] = idata;

for(unsigned int offset = 1; offset < size; offset <<= 1){
#pragma omp barrier
unsigned int t = l_Data[pos] + l_Data[pos - offset];
#pragma omp barrier
l_Data[pos] = t;
}

return l_Data[pos];
}

#else

static const unsigned int WARP_SIZE = 32;
static const unsigned int LOG2_WARP_SIZE = 5;

inline unsigned int warpScanInclusive(const unsigned int idata, 
volatile unsigned int* l_Data, 
const unsigned int size)
{
int lid = omp_get_thread_num();
unsigned int pos = 2 * lid - (lid & (size - 1));
l_Data[pos] = 0;
pos += size;
l_Data[pos] = idata;

for(unsigned int offset = 1; offset < size; offset <<= 1)
l_Data[pos] += l_Data[pos - offset];

return l_Data[pos];
}

inline unsigned int warpScanExclusive(const unsigned int idata, 
unsigned int* l_Data, const unsigned int size)
{
return warpScanInclusive(idata, l_Data, size) - idata;
}


inline unsigned int scan1Inclusive(const unsigned int idata, 
unsigned int* l_Data, const unsigned int size)
{
if(size > WARP_SIZE){
unsigned int warpResult = warpScanInclusive(idata, l_Data, WARP_SIZE);

#pragma omp barrier

int lid = omp_get_thread_num();
if( (lid & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
l_Data[lid >> LOG2_WARP_SIZE] = warpResult;

#pragma omp barrier
if( lid < (WORKGROUP_SIZE / WARP_SIZE) ){
unsigned int val = l_Data[lid] ;
l_Data[lid] = warpScanExclusive(val, l_Data, size >> LOG2_WARP_SIZE);
}

#pragma omp barrier
return warpResult + l_Data[lid >> LOG2_WARP_SIZE];
}else{
return warpScanInclusive(idata, l_Data, size);
}
}
#endif

inline unsigned int scan1Exclusive(const unsigned int idata, 
unsigned int* l_Data, const unsigned int size)
{
return scan1Inclusive(idata, l_Data, size) - idata;
}




inline uint4 scan4Inclusive(uint4 data4, 
unsigned int* l_Data, const unsigned int size){
data4.y += data4.x;
data4.z += data4.y;
data4.w += data4.z;

unsigned int val = scan1Inclusive(data4.w, l_Data, size / 4) - data4.w;

uint4 val4 = {val,val,val,val};
return (data4 + val4);
}


inline uint4 scan4Exclusive(uint4 data4, 
unsigned int* l_Data, const unsigned int size)
{
return scan4Inclusive(data4, l_Data, size) - data4;
}

#pragma omp end declare target
