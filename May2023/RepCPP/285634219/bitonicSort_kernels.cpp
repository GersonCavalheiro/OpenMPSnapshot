

#pragma omp declare target
inline void ComparatorPrivate(
unsigned int *keyA,
unsigned int *valA,
unsigned int *keyB,
unsigned int *valB,
unsigned int dir)
{
if( (*keyA > *keyB) == dir ){
unsigned int t;
t = *keyA; *keyA = *keyB; *keyB = t;
t = *valA; *valA = *valB; *valB = t;
}
}

inline void ComparatorLocal(
unsigned int* keyA,
unsigned int* valA,
unsigned int* keyB,
unsigned int* valB,
const unsigned int dir)
{
if( (*keyA > *keyB) == dir ){
unsigned int t;
t = *keyA; *keyA = *keyB; *keyB = t;
t = *valA; *valA = *valB; *valB = t;
}
}
#pragma omp end declare target

void bitonicSortLocal(
unsigned int*__restrict d_DstKey,
unsigned int*__restrict d_DstVal,
const unsigned int*__restrict d_SrcKey,
const unsigned int*__restrict d_SrcVal,
const unsigned int arrayLength,
const unsigned int dir,
const unsigned int teams,
const unsigned int threads)
{
#pragma omp target teams num_teams(teams) thread_limit(threads)
{
unsigned int l_key[LOCAL_SIZE_LIMIT];
unsigned int l_val[LOCAL_SIZE_LIMIT];
#pragma omp parallel 
{
unsigned int tid = omp_get_team_num();
unsigned int lid = omp_get_thread_num();
d_SrcKey += tid * LOCAL_SIZE_LIMIT + lid;
d_SrcVal += tid * LOCAL_SIZE_LIMIT + lid;
d_DstKey += tid * LOCAL_SIZE_LIMIT + lid;
d_DstVal += tid * LOCAL_SIZE_LIMIT + lid;
l_key[lid +                      0] = d_SrcKey[                     0];
l_val[lid +                      0] = d_SrcVal[                     0];
l_key[lid + (LOCAL_SIZE_LIMIT / 2)] = d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
l_val[lid + (LOCAL_SIZE_LIMIT / 2)] = d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

for(unsigned int size = 2; size < arrayLength; size <<= 1){
unsigned int ddd = dir ^ ( (lid & (size / 2)) != 0 );
for(unsigned int stride = size / 2; stride > 0; stride >>= 1){
#pragma omp barrier
unsigned int pos = 2 * lid - (lid & (stride - 1));
ComparatorLocal(
&l_key[pos +      0], &l_val[pos +      0],
&l_key[pos + stride], &l_val[pos + stride],
ddd);
}
}

{
for(unsigned int stride = arrayLength / 2; stride > 0; stride >>= 1){
#pragma omp barrier
unsigned int pos = 2 * lid - (lid & (stride - 1));
ComparatorLocal(
&l_key[pos +      0], &l_val[pos +      0],
&l_key[pos + stride], &l_val[pos + stride],
dir);
}
}

#pragma omp barrier
d_DstKey[                     0] = l_key[lid +                      0];
d_DstVal[                     0] = l_val[lid +                      0];
d_DstKey[(LOCAL_SIZE_LIMIT / 2)] = l_key[lid + (LOCAL_SIZE_LIMIT / 2)];
d_DstVal[(LOCAL_SIZE_LIMIT / 2)] = l_val[lid + (LOCAL_SIZE_LIMIT / 2)];
}
}
}

void bitonicSortLocal1(
unsigned int*__restrict d_DstKey,
unsigned int*__restrict d_DstVal,
const unsigned int*__restrict d_SrcKey,
const unsigned int*__restrict d_SrcVal,
const unsigned int teams,
const unsigned int threads)
{
#pragma omp target teams num_teams(teams) thread_limit(threads)
{
unsigned int l_key[LOCAL_SIZE_LIMIT];
unsigned int l_val[LOCAL_SIZE_LIMIT];
#pragma omp parallel 
{
unsigned int tid = omp_get_team_num();
unsigned int lid = omp_get_thread_num();
d_SrcKey += tid * LOCAL_SIZE_LIMIT + lid;
d_SrcVal += tid * LOCAL_SIZE_LIMIT + lid;
d_DstKey += tid * LOCAL_SIZE_LIMIT + lid;
d_DstVal += tid * LOCAL_SIZE_LIMIT + lid;
l_key[lid +                      0] = d_SrcKey[                     0];
l_val[lid +                      0] = d_SrcVal[                     0];
l_key[lid + (LOCAL_SIZE_LIMIT / 2)] = d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
l_val[lid + (LOCAL_SIZE_LIMIT / 2)] = d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

unsigned int comparatorI = (tid * threads + lid) & ((LOCAL_SIZE_LIMIT / 2) - 1);

for(unsigned int size = 2; size < LOCAL_SIZE_LIMIT; size <<= 1){
unsigned int ddd = (comparatorI & (size / 2)) != 0;
for(unsigned int stride = size / 2; stride > 0; stride >>= 1){
#pragma omp barrier
unsigned int pos = 2 * lid - (lid & (stride - 1));
ComparatorLocal(
&l_key[pos +      0], &l_val[pos +      0],
&l_key[pos + stride], &l_val[pos + stride],
ddd);
}
}

{
unsigned int ddd = (tid & 1);
for(unsigned int stride = LOCAL_SIZE_LIMIT / 2; stride > 0; stride >>= 1){
#pragma omp barrier
unsigned int pos = 2 * lid - (lid & (stride - 1));
ComparatorLocal(
&l_key[pos +      0], &l_val[pos +      0],
&l_key[pos + stride], &l_val[pos + stride],
ddd);
}
}

#pragma omp barrier
d_DstKey[                     0] = l_key[lid +                      0];
d_DstVal[                     0] = l_val[lid +                      0];
d_DstKey[(LOCAL_SIZE_LIMIT / 2)] = l_key[lid + (LOCAL_SIZE_LIMIT / 2)];
d_DstVal[(LOCAL_SIZE_LIMIT / 2)] = l_val[lid + (LOCAL_SIZE_LIMIT / 2)];
}
}
}

void bitonicMergeGlobal(
unsigned int*__restrict d_DstKey,
unsigned int*__restrict d_DstVal,
const unsigned int*__restrict d_SrcKey,
const unsigned int*__restrict d_SrcVal,
const unsigned int arrayLength,
const unsigned int size,
const unsigned int stride,
const unsigned int dir,
const unsigned int teams,
const unsigned int threads)
{
#pragma omp target teams num_teams(teams) thread_limit(threads)
{
#pragma omp parallel 
{
unsigned int tid = omp_get_team_num();
unsigned int lid = omp_get_thread_num();
unsigned int global_comparatorI = tid * threads + lid;
unsigned int        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

unsigned int ddd = dir ^ ( (comparatorI & (size / 2)) != 0 );
unsigned int pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

unsigned int keyA = d_SrcKey[pos +      0];
unsigned int valA = d_SrcVal[pos +      0];
unsigned int keyB = d_SrcKey[pos + stride];
unsigned int valB = d_SrcVal[pos + stride];

ComparatorPrivate(
&keyA, &valA,
&keyB, &valB,
ddd);

d_DstKey[pos +      0] = keyA;
d_DstVal[pos +      0] = valA;
d_DstKey[pos + stride] = keyB;
d_DstVal[pos + stride] = valB;
}
}
}

void bitonicMergeLocal(
unsigned int*__restrict d_DstKey,
unsigned int*__restrict d_DstVal,
const unsigned int*__restrict d_SrcKey,
const unsigned int*__restrict d_SrcVal,
const unsigned int arrayLength,
const unsigned int size,
unsigned int stride,
const unsigned int dir,
const unsigned int teams,
const unsigned int threads)
{
#pragma omp target teams num_teams(teams) thread_limit(threads)
{
unsigned int l_key[LOCAL_SIZE_LIMIT];
unsigned int l_val[LOCAL_SIZE_LIMIT];
#pragma omp parallel 
{
unsigned int tid = omp_get_team_num();
unsigned int lid = omp_get_thread_num();
d_SrcKey += tid * LOCAL_SIZE_LIMIT + lid;
d_SrcVal += tid * LOCAL_SIZE_LIMIT + lid;
d_DstKey += tid * LOCAL_SIZE_LIMIT + lid;
d_DstVal += tid * LOCAL_SIZE_LIMIT + lid;
l_key[lid +                      0] = d_SrcKey[                     0];
l_val[lid +                      0] = d_SrcVal[                     0];
l_key[lid + (LOCAL_SIZE_LIMIT / 2)] = d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
l_val[lid + (LOCAL_SIZE_LIMIT / 2)] = d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

unsigned int comparatorI = (tid * threads + lid) & ((arrayLength / 2) - 1);
unsigned int         ddd = dir ^ ( (comparatorI & (size / 2)) != 0 );
for(; stride > 0; stride >>= 1){
#pragma omp barrier
unsigned int pos = 2 * lid - (lid & (stride - 1));
ComparatorLocal(
&l_key[pos +      0], &l_val[pos +      0],
&l_key[pos + stride], &l_val[pos + stride],
ddd);
}

#pragma omp barrier
d_DstKey[                     0] = l_key[lid +                      0];
d_DstVal[                     0] = l_val[lid +                      0];
d_DstKey[(LOCAL_SIZE_LIMIT / 2)] = l_key[lid + (LOCAL_SIZE_LIMIT / 2)];
d_DstVal[(LOCAL_SIZE_LIMIT / 2)] = l_val[lid + (LOCAL_SIZE_LIMIT / 2)];
}
}
}
