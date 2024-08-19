#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <omp.h>

typedef union {
unsigned char *c; 
unsigned int *i;
float *f;
} _in_unip;

void print128_num(__m128i var) 
{
int64_t *v64val = (int64_t*) &var;
printf("%.16llX %.16llX\n", v64val[1], v64val[0]);
}

void print_content(float *a, int len)
{
_in_unip tmp;

for(int i = len-1; i >= 0; i--) {
tmp.f = a + i;
fprintf(stdout, "%X  ", *tmp.i, tmp.i);
}
fprintf(stdout, "\n");
}

void rand_gen(float *a, int len)
{
for(int i = 0; i < len; i++)
a[i] = (float) drand48();
}

void zero_gen(float *a, int len)
{
for(int i = 0; i < len; i++)
a[i] = (float) -1;
}

void pack_omp(float *in, int size, int bits, float *out) 
{
const int BYTE = 8;
const int FP_BYTES = 4;
const int FP_LEN = BYTE * FP_BYTES;
const int MANTISSA = 23;
const int bround = MANTISSA - bits;
int ebits = FP_LEN - bround;
ebits = bits;

const int tot_float_in = size;
const int tot_int8_in = tot_float_in * FP_BYTES;
const int int8_per_float = (ebits + BYTE - 1) / BYTE;
const int tot_int8_out = tot_float_in * int8_per_float;
const int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

_in_unip input, output, src, dst;

input.f = (float *) in;
output.f = out;

int workers = omp_get_max_threads();
printf("workers: %d\n", workers);
int i;
#pragma omp parallel for shared(i, input, output) private(src, dst)
for(i = 0; i < tot_float_in; i++) {
int skip = FP_BYTES - int8_per_float;
src.f = input.f + i;
src.c = src.c + skip;
dst.c = output.c + (i*int8_per_float);
int chunk_size = int8_per_float;
memcpy(dst.c, src.c, chunk_size);
}
}

void pack_avx(float *in, int size, int ebits, float *out) 
{
int BYTE = 8;
int FP_BYTES = 4;
int FP_LEN = BYTE * FP_BYTES;

int b;
if (ebits % BYTE) {
b = (ebits + BYTE - 1) / BYTE;
ebits = b * BYTE;
}
int dbits = FP_LEN - ebits;

int tot_float_in = size;
int tot_int8_in = tot_float_in * FP_BYTES;
int int8_per_float = (ebits + BYTE - 1) / BYTE;
int tot_int8_out = tot_float_in * int8_per_float;
int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

_in_unip conv0, conv1;
int vector_len = 8; 
int vector_lane = 4; 
int vpos = vector_len * FP_BYTES;
int iters = (tot_float_in + vector_len - 1) / vector_len;

int eint32 = int8_per_float * vector_lane / FP_BYTES;
int dint32 = vector_lane - eint32;
int dbytes = vector_lane * (FP_BYTES - int8_per_float);
int ebytes = vector_lane * int8_per_float;
char e8[vpos];
int e32[vector_len];

int top = FP_BYTES * vector_lane - 1;
int cfloat = 0;
int boff = 0;


for (int j = top; j >= dbytes; j--) {
int pos = top - boff - cfloat * FP_BYTES;
e8[j] = e8[j+vpos/2] = pos;
boff += (boff == int8_per_float - 1) ? (-boff) : 1;
cfloat += (boff == 0) ? 1 : 0;
}
for (int j = 0; j < dbytes; j++) 
e8[j] = e8[j+vpos/2] = 0x80; 
__m256i ymm_128 = _mm256_set_epi8(e8[31], e8[30], e8[29], e8[28], e8[27], e8[26], e8[25], e8[24], e8[23], e8[22], e8[21], e8[20], \
e8[19], e8[18], e8[17], e8[16], e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);


int loff = 4;
cfloat = 0;
for (int j = vector_len - 1; j >= eint32 * 2; j--) {
e32[j] = 0;
}
for (int j = eint32 * 2 - 1; j >= 0; j--) {
e32[j] = vector_lane - 1 + loff - cfloat;
cfloat += (cfloat + 1 == eint32) ? (-cfloat) : 1;
loff -= (cfloat == 0) ? 4 : 0;
}
__m256i ymm_256 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

for (int j = 0; j < vector_len; j++) {
if (j >= eint32*2)
e32[j] = 0;
else
e32[j] = -1;
}
__m256i ymm_256_2 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

conv1.f = out;
for (int i = 0; i < iters; i++) {
conv0.f = in + (i * vector_len);

__m256i ymm0 = _mm256_loadu_si256((__m256i *) conv0.i);

__m256i ymm2 = _mm256_shuffle_epi8(ymm0, ymm_128);

__m256i ymm4 = _mm256_permutevar8x32_epi32(ymm2, ymm_256);

_mm256_maskstore_epi32(conv1.i, ymm_256_2, ymm4);

conv1.i += eint32 * 2;
}
}

void pack_avx_omp(float *in, int size, int ebits, float *out) 
{
int BYTE = 8;
int FP_BYTES = 4;
int FP_LEN = BYTE * FP_BYTES;

int b;
if (ebits % BYTE) {
b = (ebits + BYTE - 1) / BYTE;
ebits = b * BYTE;
}
int dbits = FP_LEN - ebits;

int tot_float_in = size;
int tot_int8_in = tot_float_in * FP_BYTES;
int int8_per_float = (ebits + BYTE - 1) / BYTE;
int tot_int8_out = tot_float_in * int8_per_float;
int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

typedef union {
unsigned char *c; 
unsigned int *i;
float *f;
} _in_unip;
_in_unip conv0, conv1;

int vec_len = 8; 
int vec_lane = 4; 
int vpos = vec_len * FP_BYTES;
int omp_workers = omp_get_max_threads();
int tot_256 = (tot_float_in + vec_len - 1) / vec_len; 
int thd_256 = (tot_256 + omp_workers - 1) / omp_workers; 
int thd_32 = thd_256 * vec_len; 
int thd_8_out = int8_per_float * thd_32;

int eint32 = int8_per_float * vec_lane / FP_BYTES;
int dint32 = vec_lane - eint32;
int dbytes = vec_lane * (FP_BYTES - int8_per_float);
int ebytes = vec_lane * int8_per_float;
char e8[vpos];
int e32[vec_len];

int top = FP_BYTES * vec_lane - 1;
int cfloat = 0;
int boff = 0;


for (int j = top; j >= dbytes; j--) {
int pos = top - boff - cfloat * FP_BYTES;
e8[j] = e8[j+vpos/2] = pos;
boff += (boff == int8_per_float - 1) ? (-boff) : 1;
cfloat += (boff == 0) ? 1 : 0;
}
for (int j = 0; j < dbytes; j++) 
e8[j] = e8[j+vpos/2] = 0x80; 
__m256i ymm_128 = _mm256_set_epi8(e8[31], e8[30], e8[29], e8[28], e8[27], e8[26], e8[25], e8[24], e8[23], e8[22], e8[21], e8[20], \
e8[19], e8[18], e8[17], e8[16], e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);


int loff = 4;
cfloat = 0;
for (int j = vec_len - 1; j >= eint32 * 2; j--) {
e32[j] = 0;
}
for (int j = eint32 * 2 - 1; j >= 0; j--) {
e32[j] = vec_lane - 1 + loff - cfloat;
cfloat += (cfloat + 1 == eint32) ? (-cfloat) : 1;
loff -= (cfloat == 0) ? 4 : 0;
}
__m256i ymm_256 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

for (int j = 0; j < vec_len; j++) {
if (j >= eint32*2)
e32[j] = 0;
else
e32[j] = -1;
}
__m256i ymm_256_2 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

#pragma omp parallel for shared(in, out, ymm_128, ymm_256, ymm_256_2) private(conv0, conv1)
for (int i = 0; i < omp_workers; i++) {
int off_256 = i * thd_256;
int iter_256 = (i < omp_workers - 1) ? thd_256 : (tot_256 - off_256);

conv1.c = (unsigned char*) out + i * thd_8_out;
conv0.f = in + (i * thd_32);

for (int j = 0; j < iter_256; j++) {
__m256i ymm0 = _mm256_loadu_si256((__m256i*) conv0.i);

__m256i ymm2 = _mm256_shuffle_epi8(ymm0, ymm_128);

__m256i ymm4 = _mm256_permutevar8x32_epi32(ymm2, ymm_256);

_mm256_maskstore_epi32((int*) conv1.i, ymm_256_2, ymm4);

conv0.f += vec_len;

conv1.i += eint32 * 2;
}
}
}

void unpack_omp(float *in, int size, int bits, float *out) 
{
typedef union {
unsigned char *c;
unsigned int *i;
float *f;
} _in_unip;
_in_unip srcp, dstp;

const int BYTE = 8;
const int FP_BYTES = 4;
const int FP_LEN = BYTE * FP_BYTES;
int ebits = bits;
int b;
if (ebits % BYTE) {
b = (ebits + BYTE - 1) / BYTE;
ebits = b * BYTE;
}
int dbits = FP_LEN - ebits;

const int int8_per_float = (ebits + BYTE -1) / BYTE;
const int tot_float_out = size;
const int tot_int8_out = tot_float_out * FP_BYTES;
const int tot_int8_in = tot_float_out * int8_per_float;
const int tot_float_in = (tot_int8_in + FP_BYTES - 1) / FP_BYTES;

int i;
#pragma omp parallel for shared(i, in, out) private(srcp, dstp)
for(i = 0; i < tot_float_out; i++) {
srcp.c = (unsigned char *) in + i * int8_per_float;
dstp.f = out + i;
int skip = FP_BYTES - int8_per_float;
memset(dstp.f, 0, skip);
unsigned char *dst = dstp.c + skip;
memcpy(dst, srcp.c, int8_per_float);
unsigned int aux = ~0;
aux <<= FP_LEN - bits;
*dstp.i = *dstp.i & aux;
}
}

void unpack_sse(float *in, int size, int bits, float *out) 
{
typedef union {
unsigned char *c;
unsigned int *i;
float *f;
} _in_unip;
_in_unip srcp, dstp;

const int BYTE = 8;
const int FP_BYTES = 4;
const int FP_LEN = BYTE * FP_BYTES;
int ebits = bits;
int b;
if (ebits % BYTE) {
b = (ebits + BYTE - 1) / BYTE;
ebits = b * BYTE;
}
int dbits = FP_LEN - ebits;

const int int8_per_float = (ebits + BYTE -1) / BYTE;
const int tot_float_out = size;
const int tot_int8_out = tot_float_out * FP_BYTES;
const int tot_int8_in = tot_float_out * int8_per_float;
const int tot_float_in = (tot_int8_in + FP_BYTES - 1) / FP_BYTES;

const int vec_len = 4;
const int vpos = vec_len * FP_BYTES;
char e8[vpos];
int e32[vec_len];

for (int i = 0, idx = 0; i < vpos; i++) {
if ((i % FP_BYTES) >= int8_per_float) {
e8[i] = idx; 
idx += 1;
} else {
e8[i] = 0x80;
}
}
__m128i ymm_128 = _mm_set_epi8(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], e8[9], e8[8], \
e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

unsigned int aux = ~0;
aux <<= FP_LEN - bits;
__m128i ymm_and = _mm_set_epi32(aux, aux, aux, aux);

int iters = (tot_float_out + vec_len - 1) / vec_len;
srcp.f = in;
dstp.f = out;

for (int i = 0; i < iters; i++) {
__m128i ymm0 = _mm_lddqu_si128((__m128i *) srcp.i);

__m128i ymm2 = _mm_shuffle_epi8(ymm0, ymm_128);

__m128i ymm4 = _mm_and_si128(ymm2, ymm_and);

_mm_store_si128((__m128i*) dstp.f, ymm4);

srcp.c += vec_len * int8_per_float;

dstp.i += vec_len;
}
}

void unpack_sse_omp(float *in, int size, int bits, float *out) 
{
typedef union {
unsigned char *c;
unsigned int *i;
float *f;
} _in_unip;
_in_unip srcp, dstp;

const int BYTE = 8;
const int FP_BYTES = 4;
const int FP_LEN = BYTE * FP_BYTES;
int ebits = bits;
int b;
if (ebits % BYTE) {
b = (ebits + BYTE - 1) / BYTE;
ebits = b * BYTE;
}
int dbits = FP_LEN - ebits;

const int int8_per_float = (ebits + BYTE -1) / BYTE;
const int tot_float_out = size;
const int tot_int8_out = tot_float_out * FP_BYTES;
const int tot_int8_in = tot_float_out * int8_per_float;
const int tot_float_in = (tot_int8_in + FP_BYTES - 1) / FP_BYTES;

const int vec_len = 4;
const int vpos = vec_len * FP_BYTES;
char e8[vpos];
int e32[vec_len];

int omp_workers = omp_get_max_threads();
int tot_128 = (tot_float_out + vec_len - 1) / vec_len; 
int thd_128 = (tot_128 + omp_workers - 1) / omp_workers; 
int thd_32 = thd_128 * vec_len; 
int thd_8_out = FP_BYTES * thd_32;

for (int i = 0, idx = 0; i < vpos; i++) {
if ((i % FP_BYTES) >= int8_per_float) {
e8[i] = idx; 
idx += 1;
} else {
e8[i] = 0x80;
}
}
__m128i ymm_128 = _mm_set_epi8(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], e8[9], e8[8], \
e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

unsigned int aux = ~0;
aux <<= FP_LEN - bits;
__m128i ymm_and = _mm_set_epi32(aux, aux, aux, aux);

#pragma omp parallel for shared(in, out, ymm_128, ymm_and) private(srcp, dstp)
for (int i = 0; i < omp_workers; i++) {
int off_128 = i * thd_128;
int iter_128 = (i < omp_workers - 1) ? thd_128 : (tot_128 - off_128);

dstp.f = out + i * thd_32;
srcp.c = (char*) in + i * thd_32 * int8_per_float;


for (int j = 0; j < iter_128; j++) {
__m128i ymm0 = _mm_lddqu_si128((__m128i *) srcp.i);

__m128i ymm2 = _mm_shuffle_epi8(ymm0, ymm_128);

__m128i ymm4 = _mm_and_si128(ymm2, ymm_and);

_mm_store_si128((__m128i*) dstp.f, ymm4);

srcp.c += vec_len * int8_per_float;

dstp.i += vec_len;
}
}
}

int main(int argc, char *argv[])
{
if (argc < 5) {
fprintf(stderr, "use %s [omp/avx/hybrid] [ebits] [tot_float_in] [rep]\n", argv[0]);
return 1;
}

int mode = atoi(argv[1]);

int BYTE = 8;
int FP_BYTES = 4;
int FP_LEN = BYTE * FP_BYTES;

int ebits = atoi(argv[2]);
int tot_float_in = atoi(argv[3]);
int dbits = FP_LEN - ebits;

int rep = atoi(argv[4]);

int tot_int8_in = tot_float_in * FP_BYTES;
int int8_per_float = (ebits + BYTE - 1) / BYTE;
int tot_int8_out = tot_float_in * int8_per_float;
int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

float *original = calloc(tot_float_in, sizeof(float));
float *packed = calloc(tot_float_out, sizeof(float));
float *restored = calloc(tot_float_in, sizeof(float));

srand48(time(0));

fprintf(stdout, "ebits: %d, tot_float_in: %d, tot_float_out: %d\n", ebits, tot_float_in, tot_float_out);

rand_gen(original, tot_float_in);
zero_gen(packed, tot_float_out);

int *elp = calloc(rep, sizeof(int));
double *delp = calloc(rep, sizeof(double));
struct timeval st, et;
double std, etd;
for (int i = 0; i < rep; i++) {
gettimeofday(&st,NULL);
std = omp_get_wtime(); 
switch(mode) {
case 0: pack_omp(original, tot_float_in, ebits, packed);
unpack_omp(packed, tot_float_in, ebits, restored);
break;
case 1: pack_avx(original, tot_float_in, ebits, packed);
unpack_sse(packed, tot_float_in, ebits, restored);
break;
case 2: pack_avx_omp(original, tot_float_in, ebits, packed);
unpack_sse_omp(packed, tot_float_in, ebits, restored);
break;
default: fprintf(stderr, "mode unknown");
}
gettimeofday(&et,NULL);
etd = omp_get_wtime(); 
int elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
double delapsed = (etd - std) * 1000000;
if (i > 0) {
elp[i-1] = elapsed;
delp[i-1] = delapsed;
}
}

if ( mode < 0 ) {
int elp_all = 0;
for (int i = 0; i < rep; i++) {
printf("%d ", elp[i]);
elp_all += elp[i];
}
printf("\n%d\n", elp_all/rep);
} else {
double delp_all = 0;
for (int i = 0; i < rep; i++) {
printf("%ld ", (long) delp[i]);
delp_all += delp[i];
}
printf("\n%ld\n", (long) (delp_all/rep));
}

if ( tot_float_in <= 8*5) {
print_content(original, tot_float_in);
print_content(packed, tot_float_out);
print_content(restored, tot_float_in);
}
return 0;
}
