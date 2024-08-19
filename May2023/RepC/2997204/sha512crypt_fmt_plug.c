#if FMT_EXTERNS_H
extern struct fmt_main fmt_cryptsha512;
#elif FMT_REGISTERS_H
john_register_one(&fmt_cryptsha512);
#else
#define _GNU_SOURCE 1
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "arch.h"
#include "sha2.h"
#include "params.h"
#include "common.h"
#include "formats.h"
#include "johnswap.h"
#include "simd-intrinsics.h"
#ifndef OMP_SCALE
#define OMP_SCALE			1 
#endif
#ifdef SIMD_COEF_64
#define SIMD_COEF_SCALE     32
#else
#define SIMD_COEF_SCALE     1
#endif
#define FORMAT_LABEL			"sha512crypt"
#ifdef SIMD_COEF_64
#define ALGORITHM_NAME          SHA512_ALGORITHM_NAME
#else
#if ARCH_BITS >= 64
#define ALGORITHM_NAME         "64/" ARCH_BITS_STR
#else
#define ALGORITHM_NAME         "32/" ARCH_BITS_STR
#endif
#endif
#define PLAINTEXT_LENGTH		79
#define BINARY_ALIGN			4
#define SALT_SIZE			sizeof(struct saltstruct)
#define SALT_ALIGN			4
#ifdef SIMD_COEF_64
#define MIN_KEYS_PER_CRYPT		(SIMD_COEF_64*SIMD_PARA_SHA512)
#define MAX_KEYS_PER_CRYPT		(SIMD_COEF_64*SIMD_PARA_SHA512)
#if ARCH_LITTLE_ENDIAN==1
#define GETPOS(i, index)        ( (index&(SIMD_COEF_64-1))*8 + ((i)&(0xffffffff-7))*SIMD_COEF_64 + (7-((i)&7)) + (unsigned int)index/SIMD_COEF_64*SHA_BUF_SIZ*SIMD_COEF_64*8 )
#else
#define GETPOS(i, index)        ( (index&(SIMD_COEF_64-1))*8 + ((i)&(0xffffffff-7))*SIMD_COEF_64 + ((i)&7) + (unsigned int)index/SIMD_COEF_64*SHA_BUF_SIZ*SIMD_COEF_64*8 )
#endif
#else
#define MIN_KEYS_PER_CRYPT		1
#define MAX_KEYS_PER_CRYPT		1
#endif
#define BINARY_SIZE			64
#define SALT_LENGTH			16
#define CIPHERTEXT_LENGTH		86
#define __CRYPTSHA512_CREATE_PROPER_TESTS_ARRAY__
#include "sha512crypt_common.h"
#define BLKS MIN_KEYS_PER_CRYPT
typedef struct cryptloopstruct_t {
unsigned char buf[8*2*128*BLKS];	
unsigned char *bufs[BLKS][42];	
#ifdef SIMD_COEF_64
int offs[BLKS][42];
#endif
unsigned char *cptr[BLKS][42];	
int datlen[42];				
} cryptloopstruct;
static int (*saved_len);
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static uint32_t (*crypt_out)[BINARY_SIZE / sizeof(uint32_t)];
static const unsigned char padding[256] = { 0x80, 0  };
#if !defined (SIMD_COEF_64)
static const uint64_t ctx_init[8] =
{0x6A09E667F3BCC908ULL,0xBB67AE8584CAA73BULL,0x3C6EF372FE94F82BULL,0xA54FF53A5F1D36F1ULL,0x510E527FADE682D1ULL,0x9B05688C2B3E6C1FULL,0x1F83D9ABFB41BD6BULL,0x5BE0CD19137E2179ULL};
#endif
static struct saltstruct {
unsigned int len;
unsigned int rounds;
unsigned char salt[SALT_LENGTH];
} *cur_salt;
static void init(struct fmt_main *self)
{
omp_autotune(self, OMP_SCALE);
self->params.max_keys_per_crypt *= SIMD_COEF_SCALE;
saved_len = mem_calloc(1 + self->params.max_keys_per_crypt, sizeof(*saved_len));
saved_key = mem_calloc(1 + self->params.max_keys_per_crypt, sizeof(*saved_key));
crypt_out = mem_calloc(1 + self->params.max_keys_per_crypt, sizeof(*crypt_out));
}
static void done(void)
{
MEM_FREE(crypt_out);
MEM_FREE(saved_key);
MEM_FREE(saved_len);
}
#define COMMON_GET_HASH_VAR crypt_out
#include "common-get-hash.h"
static void set_key(char *key, int index)
{
saved_len[index] = strnzcpyn(saved_key[index], key, sizeof(*saved_key));
}
static char *get_key(int index)
{
saved_key[index][saved_len[index]] = 0;
return saved_key[index];
}
static void LoadCryptStruct(cryptloopstruct *crypt_struct, int index, int idx, char *p_bytes, char *s_bytes) {
unsigned len_pc, len_ppsc, len_ppc, len_psc; 
unsigned tot_pc, tot_ppsc, tot_ppc, tot_psc; 
unsigned off_pc, off_pspc, off_ppc, off_psc; 
unsigned dlen_pc, dlen_ppsc, dlen_ppc, dlen_psc; 
unsigned plen=saved_len[index];
unsigned char *cp = crypt_struct->buf;
cryptloopstruct *pstr = crypt_struct;
#ifdef SIMD_COEF_64
unsigned char *next_cp;
cp += idx*2*128;
#endif
len_pc   = plen + BINARY_SIZE;
len_ppsc = (plen<<1) + cur_salt->len + BINARY_SIZE;
len_ppc  = (plen<<1) + BINARY_SIZE;
len_psc  = plen + cur_salt->len + BINARY_SIZE;
if (len_pc  <=111) {tot_pc  =128; dlen_pc  =128;}else{tot_pc  =256; dlen_pc  =256; }
if (len_ppsc<=111) {tot_ppsc=128; dlen_ppsc=128;}else{tot_ppsc=256; dlen_ppsc=256; }
if (len_ppc <=111) {tot_ppc =128; dlen_ppc =128;}else{tot_ppc =256; dlen_ppc =256; }
if (len_psc <=111) {tot_psc =128; dlen_psc =128;}else{tot_psc =256; dlen_psc =256; }
off_pc   = len_pc   - BINARY_SIZE;
off_pspc = len_ppsc - BINARY_SIZE;
off_ppc  = len_ppc  - BINARY_SIZE;
off_psc  = len_psc  - BINARY_SIZE;
#ifdef SIMD_COEF_64
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][0] = pstr->cptr[idx][41] = cp;
memcpy(cp, crypt_out[index], BINARY_SIZE); cp += BINARY_SIZE;
memcpy(cp, p_bytes, plen); cp += plen;
if (!idx) pstr->datlen[0] = dlen_pc;
memcpy(cp, padding, tot_pc-2-len_pc); cp += (tot_pc-len_pc);
pstr->bufs[idx][0][tot_pc-2] = (len_pc<<3)>>8;
pstr->bufs[idx][0][tot_pc-1] = (len_pc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][1] = cp;
pstr->cptr[idx][0] = cp + off_pspc;
memcpy(cp, p_bytes, plen); cp += plen;
memcpy(cp, s_bytes, cur_salt->len); cp += cur_salt->len;
memcpy(cp, p_bytes, plen); cp += (plen+BINARY_SIZE);
if (!idx) pstr->datlen[1] = dlen_ppsc;
memcpy(cp, padding, tot_ppsc-2-len_ppsc);  cp += (tot_ppsc-len_ppsc);
pstr->bufs[idx][1][tot_ppsc-2] = (len_ppsc<<3)>>8;
pstr->bufs[idx][1][tot_ppsc-1] = (len_ppsc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][2] = pstr->cptr[idx][1] = cp;
cp += BINARY_SIZE;
memcpy(cp, s_bytes, cur_salt->len); cp += cur_salt->len;
memcpy(cp, p_bytes, plen); cp += plen;
memcpy(cp, p_bytes, plen); cp += plen;
if (!idx) pstr->datlen[2] = dlen_ppsc;
memcpy(cp, padding, tot_ppsc-2-len_ppsc);  cp += (tot_ppsc-len_ppsc);
pstr->bufs[idx][2][tot_ppsc-2] = (len_ppsc<<3)>>8;
pstr->bufs[idx][2][tot_ppsc-1] = (len_ppsc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][3] = cp;
pstr->cptr[idx][2] = cp + off_ppc;
memcpy(cp, p_bytes, plen); cp += plen;
memcpy(cp, p_bytes, plen); cp +=(plen+BINARY_SIZE);
if (!idx) pstr->datlen[3] = dlen_ppc;
memcpy(cp, padding, tot_ppc-2-len_ppc);  cp += (tot_ppc-len_ppc);
pstr->bufs[idx][3][tot_ppc-2] = (len_ppc<<3)>>8;
pstr->bufs[idx][3][tot_ppc-1] = (len_ppc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][4] = pstr->cptr[idx][3] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[4] = dlen_ppsc;
pstr->bufs[idx][5] = pstr->bufs[idx][1]; pstr->cptr[idx][4] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[5] = dlen_ppsc;
pstr->bufs[idx][6] = pstr->cptr[idx][5] = cp;
cp += BINARY_SIZE;
memcpy(cp, p_bytes, plen); cp += plen;
memcpy(cp, p_bytes, plen); cp += plen;
if (!idx) pstr->datlen[6] = dlen_ppc;
memcpy(cp, padding, tot_ppc-2-len_ppc);  cp += (tot_ppc-len_ppc);
pstr->bufs[idx][6][tot_ppc-2] = (len_ppc<<3)>>8;
pstr->bufs[idx][6][tot_ppc-1] = (len_ppc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][7] = cp;
pstr->cptr[idx][6] = cp + off_psc;
memcpy(cp, p_bytes, plen); cp += plen;
memcpy(cp, s_bytes, cur_salt->len); cp += (cur_salt->len+BINARY_SIZE);
if (!idx) pstr->datlen[7] = dlen_psc;
memcpy(cp, padding, tot_psc-2-len_psc);  cp += (tot_psc-len_psc);
pstr->bufs[idx][7][tot_psc-2] = (len_psc<<3)>>8;
pstr->bufs[idx][7][tot_psc-1] = (len_psc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][8] = pstr->cptr[idx][7] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[8] = dlen_ppsc;
pstr->bufs[idx][9] = pstr->bufs[idx][3]; pstr->cptr[idx][8] = pstr->cptr[idx][2];
if (!idx) pstr->datlen[9] = dlen_ppc;
pstr->bufs[idx][10] = pstr->cptr[idx][9] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[10] = dlen_ppsc;
pstr->bufs[idx][11] = pstr->bufs[idx][1]; pstr->cptr[idx][10] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[11] = dlen_ppsc;
pstr->bufs[idx][12] = pstr->cptr[idx][11] = pstr->bufs[idx][6];
if (!idx) pstr->datlen[12] = dlen_ppc;
pstr->bufs[idx][13] = pstr->bufs[idx][1]; pstr->cptr[idx][12] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[13] = dlen_ppsc;
pstr->bufs[idx][14] = pstr->cptr[idx][13] = cp;
cp += BINARY_SIZE;
memcpy(cp, s_bytes, cur_salt->len); cp += cur_salt->len;
memcpy(cp, p_bytes, plen); cp += plen;
if (!idx) pstr->datlen[14] = dlen_psc;
memcpy(cp, padding, tot_psc-2-len_psc);  cp += (tot_psc-len_psc);
pstr->bufs[idx][14][tot_psc-2] = (len_psc<<3)>>8;
pstr->bufs[idx][14][tot_psc-1] = (len_psc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][15] = pstr->bufs[idx][3]; pstr->cptr[idx][14] = pstr->cptr[idx][2];
if (!idx) pstr->datlen[15] = dlen_ppc;
pstr->bufs[idx][16] = pstr->cptr[idx][15] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[16] = dlen_ppsc;
pstr->bufs[idx][17] = pstr->bufs[idx][1]; pstr->cptr[idx][16] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[17] = dlen_ppsc;
pstr->bufs[idx][18] = pstr->cptr[idx][17] = pstr->bufs[idx][6];
if (!idx) pstr->datlen[18] = dlen_ppc;
pstr->bufs[idx][19] = pstr->bufs[idx][1]; pstr->cptr[idx][18] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[19] = dlen_ppsc;
pstr->bufs[idx][20] = pstr->cptr[idx][19] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[20] = dlen_ppsc;
pstr->bufs[idx][21] = cp;
pstr->cptr[idx][20] = cp + off_pc;
memcpy(cp, p_bytes, plen); cp += (plen+BINARY_SIZE);
if (!idx) pstr->datlen[21] = dlen_pc;
memcpy(cp, padding, tot_psc-2-len_pc);
pstr->bufs[idx][21][tot_pc-2] = (len_pc<<3)>>8;
pstr->bufs[idx][21][tot_pc-1] = (len_pc<<3)&0xFF;
#ifdef SIMD_COEF_64
cp = next_cp;
next_cp = cp + (2*128*BLKS);
#endif
pstr->bufs[idx][22] = pstr->cptr[idx][21] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[22] = dlen_ppsc;
pstr->bufs[idx][23] = pstr->bufs[idx][1]; pstr->cptr[idx][22] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[23] = dlen_ppsc;
pstr->bufs[idx][24] = pstr->cptr[idx][23] = pstr->bufs[idx][6];
if (!idx) pstr->datlen[24] = dlen_ppc;
pstr->bufs[idx][25] = pstr->bufs[idx][1]; pstr->cptr[idx][24] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[25] = dlen_ppsc;
pstr->bufs[idx][26] = pstr->cptr[idx][25] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[26] = dlen_ppsc;
pstr->bufs[idx][27] = pstr->bufs[idx][3]; pstr->cptr[idx][26] = pstr->cptr[idx][2];
if (!idx) pstr->datlen[27] = dlen_ppc;
pstr->bufs[idx][28] = pstr->cptr[idx][27] = pstr->bufs[idx][14];
if (!idx) pstr->datlen[28] = dlen_psc;
pstr->bufs[idx][29] = pstr->bufs[idx][1]; pstr->cptr[idx][28] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[29] = dlen_ppsc;
pstr->bufs[idx][30] = pstr->cptr[idx][29] = pstr->bufs[idx][6];
if (!idx) pstr->datlen[30] = dlen_ppc;
pstr->bufs[idx][31] = pstr->bufs[idx][1]; pstr->cptr[idx][30] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[31] = dlen_ppsc;
pstr->bufs[idx][32] = pstr->cptr[idx][31] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[32] = dlen_ppsc;
pstr->bufs[idx][33] = pstr->bufs[idx][3]; pstr->cptr[idx][32] = pstr->cptr[idx][2];
if (!idx) pstr->datlen[33] = dlen_ppc;
pstr->bufs[idx][34] = pstr->cptr[idx][33] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[34] = dlen_ppsc;
pstr->bufs[idx][35] = pstr->bufs[idx][7]; pstr->cptr[idx][34] = pstr->cptr[idx][6];
if (!idx) pstr->datlen[35] = dlen_psc;
pstr->bufs[idx][36] = pstr->cptr[idx][35] = pstr->bufs[idx][6];
if (!idx) pstr->datlen[36] = dlen_ppc;
pstr->bufs[idx][37] = pstr->bufs[idx][1]; pstr->cptr[idx][36] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[37] = dlen_ppsc;
pstr->bufs[idx][38] = pstr->cptr[idx][37] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[38] = dlen_ppsc;
pstr->bufs[idx][39] = pstr->bufs[idx][3]; pstr->cptr[idx][38] = pstr->cptr[idx][2];
if (!idx) pstr->datlen[39] = dlen_ppc;
pstr->bufs[idx][40] = pstr->cptr[idx][39] = pstr->bufs[idx][2];
if (!idx) pstr->datlen[40] = dlen_ppsc;
pstr->bufs[idx][41] = pstr->bufs[idx][1]; pstr->cptr[idx][40] = pstr->cptr[idx][0];
if (!idx) pstr->datlen[41] = dlen_ppsc;
}
static int crypt_all(int *pcount, struct db_salt *salt)
{
const int count = *pcount;
int index = 0;
int *MixOrder, tot_todo;
#ifdef SIMD_COEF_64
MixOrder = mem_calloc((count+6*MIN_KEYS_PER_CRYPT), sizeof(int));
{
static const int lens[17][6] = {
{0,24,48,88,89,90},  
{0,24,48,88,89,90},  
{0,23,24,46,48,87},  
{0,23,24,45,48,87},  
{0,22,24,44,48,86},  
{0,22,24,43,48,86},  
{0,21,24,42,48,85},  
{0,21,24,41,48,85},  
{0,20,24,40,48,84},  
{0,20,24,39,48,84},  
{0,19,24,38,48,83},  
{0,19,24,37,48,83},  
{0,18,24,36,48,82},  
{0,18,24,35,48,82},  
{0,17,24,34,48,81},  
{0,17,24,33,48,81},  
{0,16,24,32,48,80} };
int j;
tot_todo = 0;
saved_len[count] = 0; 
for (j = 0; j < 5; ++j) {
for (index = 0; index < count; ++index) {
if (saved_len[index] >= lens[cur_salt->len][j] && saved_len[index] < lens[cur_salt->len][j+1])
MixOrder[tot_todo++] = index;
}
while (tot_todo % MIN_KEYS_PER_CRYPT)
MixOrder[tot_todo++] = count;
}
}
#else
MixOrder = mem_calloc(count, sizeof(int));
for (index = 0; index < count; ++index)
MixOrder[index] = index;
tot_todo = count;
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (index = 0; index < tot_todo; index += MIN_KEYS_PER_CRYPT)
{
union xx {
unsigned char c[BINARY_SIZE];
ARCH_WORD a[BINARY_SIZE/sizeof(ARCH_WORD)];
} u;
unsigned char *temp_result = u.c;
SHA512_CTX ctx;
SHA512_CTX alt_ctx;
size_t cnt;
int idx;
char *cp;
char p_bytes[PLAINTEXT_LENGTH+1];
char s_bytes[PLAINTEXT_LENGTH+1];
char tmp_cls[sizeof(cryptloopstruct)+MEM_ALIGN_SIMD];
cryptloopstruct *crypt_struct;
#ifdef SIMD_COEF_64
char tmp_sse_out[8*MIN_KEYS_PER_CRYPT*8+MEM_ALIGN_SIMD];
uint64_t *sse_out;
sse_out = (uint64_t *)mem_align(tmp_sse_out, MEM_ALIGN_SIMD);
#endif
crypt_struct = (cryptloopstruct *)mem_align(tmp_cls,MEM_ALIGN_SIMD);
for (idx = 0; idx < MIN_KEYS_PER_CRYPT; ++idx)
{
SHA512_Init(&ctx);
SHA512_Update(&ctx, (unsigned char*)saved_key[MixOrder[index+idx]], saved_len[MixOrder[index+idx]]);
SHA512_Update(&ctx, cur_salt->salt, cur_salt->len);
SHA512_Init(&alt_ctx);
SHA512_Update(&alt_ctx, (unsigned char*)saved_key[MixOrder[index+idx]], saved_len[MixOrder[index+idx]]);
SHA512_Update(&alt_ctx, cur_salt->salt, cur_salt->len);
SHA512_Update(&alt_ctx, (unsigned char*)saved_key[MixOrder[index+idx]], saved_len[MixOrder[index+idx]]);
SHA512_Final((unsigned char*)crypt_out[MixOrder[index+idx]], &alt_ctx);
for (cnt = saved_len[MixOrder[index+idx]]; cnt > BINARY_SIZE; cnt -= BINARY_SIZE)
SHA512_Update(&ctx, (unsigned char*)crypt_out[MixOrder[index+idx]], BINARY_SIZE);
SHA512_Update(&ctx, (unsigned char*)crypt_out[MixOrder[index+idx]], cnt);
for (cnt = saved_len[MixOrder[index+idx]]; cnt > 0; cnt >>= 1)
if ((cnt & 1) != 0)
SHA512_Update(&ctx, (unsigned char*)crypt_out[MixOrder[index+idx]], BINARY_SIZE);
else
SHA512_Update(&ctx, (unsigned char*)saved_key[MixOrder[index+idx]], saved_len[MixOrder[index+idx]]);
SHA512_Final((unsigned char*)crypt_out[MixOrder[index+idx]], &ctx);
SHA512_Init(&alt_ctx);
for (cnt = 0; cnt < saved_len[MixOrder[index+idx]]; ++cnt)
SHA512_Update(&alt_ctx, (unsigned char*)saved_key[MixOrder[index+idx]], saved_len[MixOrder[index+idx]]);
SHA512_Final(temp_result, &alt_ctx);
cp = p_bytes;
for (cnt = saved_len[MixOrder[index+idx]]; cnt >= BINARY_SIZE; cnt -= BINARY_SIZE)
cp = (char *) memcpy (cp, temp_result, BINARY_SIZE) + BINARY_SIZE;
memcpy (cp, temp_result, cnt);
SHA512_Init(&alt_ctx);
for (cnt = 0; cnt < 16 + ((unsigned char*)crypt_out[MixOrder[index+idx]])[0]; ++cnt)
SHA512_Update(&alt_ctx, cur_salt->salt, cur_salt->len);
SHA512_Final(temp_result, &alt_ctx);
cp = s_bytes;
for (cnt = cur_salt->len; cnt >= BINARY_SIZE; cnt -= BINARY_SIZE)
cp = (char *) memcpy (cp, temp_result, BINARY_SIZE) + BINARY_SIZE;
memcpy (cp, temp_result, cnt);
LoadCryptStruct(crypt_struct, MixOrder[index+idx], idx, p_bytes, s_bytes);
}
idx = 0;
#ifdef SIMD_COEF_64
for (cnt = 1; ; ++cnt) {
if (crypt_struct->datlen[idx]==256) {
unsigned char *cp = crypt_struct->bufs[0][idx];
SIMDSHA512body(cp, sse_out, NULL, SSEi_FLAT_IN|SSEi_2BUF_INPUT_FIRST_BLK);
SIMDSHA512body(&cp[128], sse_out, sse_out, SSEi_FLAT_IN|SSEi_2BUF_INPUT_FIRST_BLK|SSEi_RELOAD);
} else {
unsigned char *cp = crypt_struct->bufs[0][idx];
SIMDSHA512body(cp, sse_out, NULL, SSEi_FLAT_IN|SSEi_2BUF_INPUT_FIRST_BLK);
}
if (cnt == cur_salt->rounds)
break;
{
unsigned int j, k;
for (k = 0; k < MIN_KEYS_PER_CRYPT; ++k) {
uint64_t *o = (uint64_t *)crypt_struct->cptr[k][idx];
#if !ARCH_ALLOWS_UNALIGNED
if (!is_aligned(o, 8)) {
unsigned char *cp = (unsigned char*)o;
for (j = 0; j < 64; ++j)
*cp++ = ((unsigned char*)sse_out)[GETPOS(j, k)];
} else
#endif
for (j = 0; j < 8; ++j)
#if ARCH_LITTLE_ENDIAN==1
*o++ = JOHNSWAP64(sse_out[j*SIMD_COEF_64+(k&(SIMD_COEF_64-1))+k/SIMD_COEF_64*8*SIMD_COEF_64]);
#else
*o++ = sse_out[j*SIMD_COEF_64+(k&(SIMD_COEF_64-1))+k/SIMD_COEF_64*8*SIMD_COEF_64];
#endif
}
}
if (++idx == 42)
idx = 0;
}
{
unsigned int j, k;
for (k = 0; k < MIN_KEYS_PER_CRYPT; ++k) {
uint64_t *o = (uint64_t *)crypt_out[MixOrder[index+k]];
for (j = 0; j < 8; ++j)
#if ARCH_LITTLE_ENDIAN==1
*o++ = JOHNSWAP64(sse_out[j*SIMD_COEF_64+(k&(SIMD_COEF_64-1))+k/SIMD_COEF_64*8*SIMD_COEF_64]);
#else
*o++ = sse_out[j*SIMD_COEF_64+(k&(SIMD_COEF_64-1))+k/SIMD_COEF_64*8*SIMD_COEF_64];
#endif
}
}
#else
SHA512_Init(&ctx);
for (cnt = 1; ; ++cnt) {
SHA512_Update(&ctx, crypt_struct->bufs[0][idx], crypt_struct->datlen[idx]);
if (cnt == cur_salt->rounds)
break;
#if ARCH_LITTLE_ENDIAN
{
int j;
uint64_t *o = (uint64_t *)crypt_struct->cptr[0][idx];
for (j = 0; j < 8; ++j)
*o++ = JOHNSWAP64(ctx.h[j]);
}
#else
memcpy(crypt_struct->cptr[0][idx], ctx.h, BINARY_SIZE);
#endif
if (++idx == 42)
idx = 0;
memcpy(ctx.h, ctx_init, sizeof(ctx_init));
}
#if ARCH_LITTLE_ENDIAN
{
int j;
uint64_t *o = (uint64_t *)crypt_out[MixOrder[index]];
for (j = 0; j < 8; ++j)
*o++ = JOHNSWAP64(ctx.h[j]);
}
#else
memcpy(crypt_out[MixOrder[index]], ctx.h, BINARY_SIZE);
#endif
#endif
}
MEM_FREE(MixOrder);
return count;
}
static void set_salt(void *salt)
{
cur_salt = salt;
}
static void *get_salt(char *ciphertext)
{
static struct saltstruct out;
int len;
memset(&out, 0, sizeof(out));
out.rounds = ROUNDS_DEFAULT;
ciphertext += FORMAT_TAG_LEN;
if (!strncmp(ciphertext, ROUNDS_PREFIX,
sizeof(ROUNDS_PREFIX) - 1)) {
const char *num = ciphertext + sizeof(ROUNDS_PREFIX) - 1;
char *endp;
unsigned long int srounds = strtoul(num, &endp, 10);
if (*endp == '$')
{
ciphertext = endp + 1;
srounds = srounds < ROUNDS_MIN ?
ROUNDS_MIN : srounds;
out.rounds = srounds > ROUNDS_MAX ?
ROUNDS_MAX : srounds;
}
}
for (len = 0; ciphertext[len] != '$'; len++);
if (len > SALT_LENGTH)
len = SALT_LENGTH;
memcpy(out.salt, ciphertext, len);
out.len = len;
return &out;
}
static int cmp_all(void *binary, int count)
{
int index;
for (index = 0; index < count; index++)
if (!memcmp(binary, crypt_out[index], ARCH_SIZE))
return 1;
return 0;
}
static int cmp_one(void *binary, int index)
{
return !memcmp(binary, crypt_out[index], BINARY_SIZE);
}
static int cmp_exact(char *source, int index)
{
return 1;
}
static unsigned int iteration_count(void *salt)
{
struct saltstruct *sha512crypt_salt;
sha512crypt_salt = salt;
return (unsigned int)sha512crypt_salt->rounds;
}
static int salt_hash(void *salt)
{
unsigned char *s = salt;
unsigned int hash = 5381;
unsigned int i;
for (i = 0; i < SALT_SIZE; i++)
hash = ((hash << 5) + hash) ^ s[i];
return hash & (SALT_HASH_SIZE - 1);
}
struct fmt_main fmt_cryptsha512 = {
{
FORMAT_LABEL,
FORMAT_NAME,
"SHA512 " ALGORITHM_NAME,
BENCHMARK_COMMENT,
BENCHMARK_LENGTH,
0,
PLAINTEXT_LENGTH,
BINARY_SIZE,
BINARY_ALIGN,
SALT_SIZE,
SALT_ALIGN,
MIN_KEYS_PER_CRYPT,
MAX_KEYS_PER_CRYPT,
FMT_CASE | FMT_8_BIT | FMT_OMP,
{
"iteration count",
},
{ FORMAT_TAG },
tests
}, {
init,
done,
fmt_default_reset,
fmt_default_prepare,
valid,
fmt_default_split,
get_binary,
get_salt,
{
iteration_count,
},
fmt_default_source,
{
fmt_default_binary_hash_0,
fmt_default_binary_hash_1,
fmt_default_binary_hash_2,
fmt_default_binary_hash_3,
fmt_default_binary_hash_4,
fmt_default_binary_hash_5,
fmt_default_binary_hash_6
},
salt_hash,
NULL,
set_salt,
set_key,
get_key,
fmt_default_clear_keys,
crypt_all,
{
#define COMMON_GET_HASH_LINK
#include "common-get-hash.h"
},
cmp_all,
cmp_one,
cmp_exact
}
};
#endif 
