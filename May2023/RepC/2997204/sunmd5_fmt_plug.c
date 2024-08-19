#if FMT_EXTERNS_H
extern struct fmt_main fmt_sunmd5;
#elif FMT_REGISTERS_H
john_register_one(&fmt_sunmd5);
#else
#include <string.h>
#include "os.h"
#if (!AC_BUILT || HAVE_UNISTD_H) && !_MSC_VER
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "arch.h"
#if !ARCH_LITTLE_ENDIAN
#undef SIMD_COEF_32
#undef SIMD_PARA_MD5
#endif
#include "misc.h"
#include "options.h"
#include "params.h"
#include "memory.h"
#include "common.h"
#include "formats.h"
#include "loader.h"
#include "md5.h"
#include "simd-intrinsics.h"
#ifndef OMP_SCALE
#if SIMD_COEF_32
#define OMP_SCALE 2 
#else
#define OMP_SCALE 8 
#endif
#endif
#ifndef MD5_CBLOCK
#define MD5_CBLOCK 64
#endif
#ifndef MD5_DIGEST_LENGTH
#define MD5_DIGEST_LENGTH 16
#endif
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define PLAINTEXT_LENGTH		120
#define FULL_BINARY_SIZE		16
#define BINARY_SIZE			4
#define BINARY_ALIGN			4
#define SALT_SIZE			40
#define SALT_ALIGN			1
#if SIMD_COEF_32
#define MIN_KEYS_PER_CRYPT  (SIMD_COEF_32 * SIMD_PARA_MD5)
#define MAX_KEYS_PER_CRYPT  (SIMD_COEF_32 * SIMD_PARA_MD5 * 4)
#else
#define MIN_KEYS_PER_CRYPT	1
#define MAX_KEYS_PER_CRYPT	2
#endif
#define FORMAT_LABEL			"SunMD5"
#define FORMAT_NAME			""
#define FORMAT_TAG           "$md5$"
#define FORMAT_TAG2          "$md5,"
#define FORMAT_TAG_LEN       (sizeof(FORMAT_TAG)-1)
#define ALGORITHM_NAME			"MD5 " MD5_ALGORITHM_NAME
#define BENCHMARK_COMMENT		""
#define BENCHMARK_LENGTH		0x107
static struct fmt_tests tests[] = {
{"$md5$rounds=904$Vc3VgyFx44iS8.Yu$Scf90iLWN6O6mT9TA06NK/", "test"},
{"$md5$rounds=904$ZZZig8GS.S0pRNhc$dw5NMYJoxLlnFq4E.phLy.", "Don41dL33"},
{"$md5$rounds=904$zSuVTn567UJLv14u$q2n2ZBFwKg2tElFBIzUq/0", "J4ck!3Wood"},
{"$md5$rounds=904$zuZVga3IOSfOshxU$gkUlHjR6apc6cr.7Bu5tt/", "K!m!M4rt!n"},
{"$md5$rounds=904$/KP7bVaKYTOcplkx$i74NBQdysLaDTUSEu5FtQ.", "people"},
{"$md5$rounds=904$/p4qqfWbTQcUqjNc$leW.8/vzyDpFQxSZrV0x.0", "me"},
{"$md5$rounds=904$wOyGLc0NMRiXJTvI$v69lVSnLif78hZbZWhuEG1", "private"},
{"$md5$rounds=904$Vc3VgyFx44iS8.Yu$mEyEet31IlEkO4HTeobmq0", "012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789"},
{NULL}
};
#ifdef SIMD_PARA_MD5
#define PARA SIMD_PARA_MD5
#else
#define PARA 1
#endif
#ifdef SIMD_COEF_32
#define COEF SIMD_COEF_32
#define BLK_CNT (PARA*COEF)
#if PARA > 1
#define MIN_DROP_BACK 1
#else
#define MIN_DROP_BACK 1
#endif
#define GETPOS(i, index)		    ( (((index)&(SIMD_COEF_32-1))<<2) + (((i)&(0xffffffff-3))*SIMD_COEF_32) + ((i)&3) )
#define PARAGETPOS(i, index)		( (((index)&(SIMD_COEF_32-1))<<2) + (((i)&(0xffffffff-3))*SIMD_COEF_32) + ((i)&3) + (((unsigned int)index/SIMD_COEF_32*SIMD_COEF_32)<<6) )
#define GETPOS0(i)					(                               (((i)&(0xffffffff-3))*SIMD_COEF_32) + ((i)&3) )
#define PARAGETOUTPOS(i, index)		( (((index)&(SIMD_COEF_32-1))<<2) + (((i)&(0xffffffff-3))*SIMD_COEF_32) + ((i)&3) + (((unsigned int)index/SIMD_COEF_32*SIMD_COEF_32)<<4) )
static uint32_t (*input_buf)[BLK_CNT * MD5_CBLOCK / sizeof(uint32_t)];
static uint32_t (*out_buf)[BLK_CNT * MD5_DIGEST_LENGTH / sizeof(uint32_t)];
static uint32_t (*input_buf_big)[25][BLK_CNT * MD5_CBLOCK / sizeof(uint32_t)];
#else
#define COEF 1
#endif
static uint32_t (*crypt_out)[FULL_BINARY_SIZE / sizeof(uint32_t)];
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static char *saved_salt;
#define	BASIC_ROUND_COUNT 4096 
#define	DIGEST_LEN	16
#define	ROUND_BUFFER_LEN	64
typedef struct {
MD5_CTX context;	
uint32_t digest[DIGEST_LEN / sizeof(uint32_t)]; 
} JTR_ALIGN(MEM_ALIGN_CACHE) Contx, *pConx;
static Contx *data;
#define constant_phrase_size 1517
static const char constant_phrase[] =
"To be, or not to be,--that is the question:--\n"
"Whether 'tis nobler in the mind to suffer\n"
"The slings and arrows of outrageous fortune\n"
"Or to take arms against a sea of troubles,\n"
"And by opposing end them?--To die,--to sleep,--\n"
"No more; and by a sleep to say we end\n"
"The heartache, and the thousand natural shocks\n"
"That flesh is heir to,--'tis a consummation\n"
"Devoutly to be wish'd. To die,--to sleep;--\n"
"To sleep! perchance to dream:--ay, there's the rub;\n"
"For in that sleep of death what dreams may come,\n"
"When we have shuffled off this mortal coil,\n"
"Must give us pause: there's the respect\n"
"That makes calamity of so long life;\n"
"For who would bear the whips and scorns of time,\n"
"The oppressor's wrong, the proud man's contumely,\n"
"The pangs of despis'd love, the law's delay,\n"
"The insolence of office, and the spurns\n"
"That patient merit of the unworthy takes,\n"
"When he himself might his quietus make\n"
"With a bare bodkin? who would these fardels bear,\n"
"To grunt and sweat under a weary life,\n"
"But that the dread of something after death,--\n"
"The undiscover'd country, from whose bourn\n"
"No traveller returns,--puzzles the will,\n"
"And makes us rather bear those ills we have\n"
"Than fly to others that we know not of?\n"
"Thus conscience does make cowards of us all;\n"
"And thus the native hue of resolution\n"
"Is sicklied o'er with the pale cast of thought;\n"
"And enterprises of great pith and moment,\n"
"With this regard, their currents turn awry,\n"
"And lose the name of action.--Soft you now!\n"
"The fair Ophelia!--Nymph, in thy orisons\n"
"Be all my sins remember'd.\n";
static unsigned char mod5[0x100];
static int ngroups = 1;
static void init(struct fmt_main *self)
{
int i;
#ifdef SIMD_COEF_32
int j, k;
#endif
ngroups = omp_autotune(self, OMP_SCALE);
#ifdef SIMD_COEF_32
input_buf     = mem_calloc_align(ngroups, sizeof(*input_buf), MEM_ALIGN_CACHE);
input_buf_big = mem_calloc_align(ngroups, sizeof(*input_buf_big), MEM_ALIGN_CACHE);
out_buf       = mem_calloc_align(ngroups, sizeof(*out_buf), MEM_ALIGN_CACHE);
for (k = 0; k < ngroups; ++k) {
for (i = 0; i < constant_phrase_size; ++i) {
for (j = 0; j < BLK_CNT; ++j)
((unsigned char*)input_buf_big[k][(i+16)/64])[PARAGETPOS((16+i)%64,j)] = constant_phrase[i];
}
}
#endif
saved_key = mem_calloc(self->params.max_keys_per_crypt, sizeof(*saved_key));
saved_salt = mem_calloc(1, SALT_SIZE + 1);
crypt_out = mem_calloc(self->params.max_keys_per_crypt, sizeof(*crypt_out));
data = mem_calloc_align(self->params.max_keys_per_crypt, sizeof(*data), MEM_ALIGN_CACHE);
for (i = 0; i < 0x100; i++)
mod5[i] = i % 5;
}
static void done(void)
{
MEM_FREE(data);
MEM_FREE(crypt_out);
MEM_FREE(saved_salt);
MEM_FREE(saved_key);
#ifdef SIMD_COEF_32
MEM_FREE(input_buf);
MEM_FREE(input_buf_big);
MEM_FREE(out_buf);
#endif
}
static int valid(char *ciphertext, struct fmt_main *self)
{
char *pos;
if (strncmp(ciphertext, FORMAT_TAG, FORMAT_TAG_LEN) &&
strncmp(ciphertext, FORMAT_TAG2, FORMAT_TAG_LEN))
return 0;
ciphertext += FORMAT_TAG_LEN-1;
if (!strncmp(ciphertext, ",rounds=", 8) ||
!strncmp(ciphertext, "$rounds=", 8)) {
pos = ciphertext += 8;
while (*ciphertext >= '0' && *ciphertext <= '9')
ciphertext++;
if (ciphertext - pos < 1 || ciphertext - pos > 6)
return 0;
}
if (*ciphertext++ != '$')
return 0;
pos = ciphertext;
while (atoi64[ARCH_INDEX(*ciphertext)] != 0x7F)
ciphertext++;
if (ciphertext - pos > 16)
return 0;
if (*ciphertext++ != '$')
return 0;
if (*ciphertext == '$')
ciphertext++;
pos = ciphertext;
while (atoi64[ARCH_INDEX(*ciphertext)] != 0x7F)
ciphertext++;
if (ciphertext - pos != 22
)
return 0;
if (*ciphertext)
return 0;
return 1;
}
static long from64 (unsigned char *s, int n) {
long l = 0;
while (--n >= 0) {
l <<= 6;
l += atoi64[s[n]];
}
return l;
}
static void *get_binary(char *ciphertext)
{
static union {
char c[FULL_BINARY_SIZE];
uint32_t w[FULL_BINARY_SIZE / sizeof(uint32_t)];
} out;
unsigned l;
unsigned char *cp;
cp = (unsigned char*)strrchr(ciphertext, '$');
++cp;
l = from64(cp, 4);
out.c[0] = l>>16;  out.c[6] = (l>>8)&0xFF;  out.c[12] = l&0xFF;
l = from64(&cp[4], 4);
out.c[1] = l>>16;  out.c[7] = (l>>8)&0xFF;  out.c[13] = l&0xFF;
l = from64(&cp[8], 4);
out.c[2] = l>>16;  out.c[8] = (l>>8)&0xFF;  out.c[14] = l&0xFF;
l = from64(&cp[12], 4);
out.c[3] = l>>16;  out.c[9] = (l>>8)&0xFF;  out.c[15] = l&0xFF;
l = from64(&cp[16], 4);
out.c[4] = l>>16;  out.c[10] = (l>>8)&0xFF;  out.c[5] = l&0xFF;
l = from64(&cp[20], 2);
out.c[11] = l;
return out.c;
}
static void *get_salt(char *ciphertext)
{
static char out[SALT_SIZE];
char *p = strrchr(ciphertext, '$');
memset(out, 0, sizeof(out));
memcpy(out, ciphertext, p - ciphertext);
return out;
}
#define COMMON_GET_HASH_VAR crypt_out
#include "common-get-hash.h"
static int salt_hash(void *salt)
{
int h;
char *sp = (char*)salt;
char *cp = strrchr(sp, '$');
if (cp) --cp;
else cp = &sp[strlen(sp)-1];
h = atoi64[ARCH_INDEX(*cp--)];
h ^= (unsigned char)*cp--;
h <<= 5;
h ^= atoi64[ARCH_INDEX(*cp--)];
h ^= (unsigned char)*cp++;
return h & (SALT_HASH_SIZE - 1);
}
static void set_salt(void *salt)
{
memcpy(saved_salt, salt, SALT_SIZE);
}
static void set_key(char *key, int index)
{
strnzcpy(saved_key[index], key, sizeof(*saved_key));
}
static char *get_key(int index)
{
return saved_key[index];
}
static int cmp_all(void *binary, int count)
{
int index;
for (index = 0; index < count; index++)
if (((uint32_t*)binary)[0] == crypt_out[index][0])
return 1;
return 0;
}
static int cmp_one(void *binary, int index)
{
return ((uint32_t*)binary)[0] == crypt_out[index][0];
}
static int cmp_exact(char *source, int index)
{
return !memcmp(get_binary(source), crypt_out[index], FULL_BINARY_SIZE);
}
#define md5bit_1(d,b) ((d[((b)>>3)&0xF]&(1<<((b)&7))) ? 1 : 0)
#define md5bit_2(d,b) (((d[((b)>>3)&0xF]>>((b)&7)))&1)
inline static int
md5bit(unsigned char *digest, int bit_num)
{
return (((digest[((bit_num)>>3)&0xF]>>((bit_num)&7)))&1);
#if 0
int byte_off;
int bit_off;
bit_num %= 128; 
byte_off = bit_num / 8;
bit_off = bit_num % 8;
return ((digest[byte_off] & (0x01 << bit_off)) ? 1 : 0);
#endif
}
inline static int
coin_step(unsigned char *digest, int i, int j, int shift)
{
return md5bit(digest, digest[(digest[i] >> mod5[digest[j]]) & 0x0F] >> ((digest[j] >> (digest[i] & 0x07)) & 0x01)) << shift;
}
#define	ROUNDS		"rounds="
#define	ROUNDSLEN	7
static unsigned int
getrounds(const char *s)
{
char *r, *p, *e;
long val;
if (s == NULL)
return (0);
if ((r = strstr(s, ROUNDS)) == NULL) {
return (0);
}
if (strncmp(r, ROUNDS, ROUNDSLEN) != 0) {
return (0);
}
p = r + ROUNDSLEN;
val = strtol(p, &e, 10);
if (val < 0 ||
!(*e == '\0' || *e == ',' || *e == '$')) {
fprintf(stderr, "crypt_sunmd5: invalid rounds specification \"%s\"", s);
return (0);
}
return ((unsigned int)val);
}
static int crypt_all(int *pcount, struct db_salt *salt)
{
const int count = *pcount;
unsigned int k, group_idx;
int group_sz = (count + ngroups - 1) / ngroups;
for (k = 0; k < count; ++k) {
MD5_Init(&data[k].context);
MD5_Update(&data[k].context, (unsigned char*)saved_key[k], strlen(saved_key[k]));
MD5_Update(&data[k].context, (unsigned char*)saved_salt, strlen(saved_salt));
MD5_Final((unsigned char*)data[k].digest, &data[k].context);
}
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (group_idx = 0; group_idx < ngroups; ++group_idx) {
int roundasciilen;
int round, maxrounds = BASIC_ROUND_COUNT + getrounds(saved_salt);
char roundascii[8];
int idx_begin = group_idx * group_sz;
int idx_end = idx_begin + group_sz > count ?
count : idx_begin + group_sz;
#ifdef SIMD_COEF_32
int i, j, zs, zb, zs0, zb0;
int bigs[MAX_KEYS_PER_CRYPT], smalls[MAX_KEYS_PER_CRYPT];
int nbig, nsmall;
memset(input_buf[group_idx], 0, BLK_CNT*MD5_CBLOCK);
#endif
roundascii[0] = '0';
roundascii[1] = 0;
roundasciilen=1;
for (round = 0; round < maxrounds; round++) {
#ifdef SIMD_COEF_32
nbig = nsmall = 0;
#endif
unsigned int idx;
for (idx = idx_begin; idx < idx_end; ++idx) {
pConx px = &data[idx];
int indirect_a =
md5bit((unsigned char*)px->digest, round) ?
coin_step((unsigned char*)px->digest, 1,  4,  0) |
coin_step((unsigned char*)px->digest, 2,  5,  1) |
coin_step((unsigned char*)px->digest, 3,  6,  2) |
coin_step((unsigned char*)px->digest, 4,  7,  3) |
coin_step((unsigned char*)px->digest, 5,  8,  4) |
coin_step((unsigned char*)px->digest, 6,  9,  5) |
coin_step((unsigned char*)px->digest, 7, 10,  6)
:
coin_step((unsigned char*)px->digest, 0,  3,  0) |
coin_step((unsigned char*)px->digest, 1,  4,  1) |
coin_step((unsigned char*)px->digest, 2,  5,  2) |
coin_step((unsigned char*)px->digest, 3,  6,  3) |
coin_step((unsigned char*)px->digest, 4,  7,  4) |
coin_step((unsigned char*)px->digest, 5,  8,  5) |
coin_step((unsigned char*)px->digest, 6,  9,  6);
int indirect_b =
md5bit((unsigned char*)px->digest, round + 64) ?
coin_step((unsigned char*)px->digest,  9, 12,  0) |
coin_step((unsigned char*)px->digest, 10, 13,  1) |
coin_step((unsigned char*)px->digest, 11, 14,  2) |
coin_step((unsigned char*)px->digest, 12, 15,  3) |
coin_step((unsigned char*)px->digest, 13,  0,  4) |
coin_step((unsigned char*)px->digest, 14,  1,  5) |
coin_step((unsigned char*)px->digest, 15,  2,  6)
:
coin_step((unsigned char*)px->digest,  8, 11,  0) |
coin_step((unsigned char*)px->digest,  9, 12,  1) |
coin_step((unsigned char*)px->digest, 10, 13,  2) |
coin_step((unsigned char*)px->digest, 11, 14,  3) |
coin_step((unsigned char*)px->digest, 12, 15,  4) |
coin_step((unsigned char*)px->digest, 13,  0,  5) |
coin_step((unsigned char*)px->digest, 14,  1,  6);
int bit = md5bit((unsigned char*)px->digest, indirect_a) ^ md5bit((unsigned char*)px->digest, indirect_b);
#ifndef SIMD_COEF_32
MD5_Init(&px->context);
MD5_Update(&px->context, (unsigned char*)px->digest, sizeof(px->digest));
if (bit)
MD5_Update(&px->context, (unsigned char*)constant_phrase, constant_phrase_size);
MD5_Update(&px->context, (unsigned char*)roundascii, roundasciilen);
MD5_Final((unsigned char*)px->digest, &px->context);
#else
if (bit)
bigs[nbig++] = idx;
else
smalls[nsmall++] = idx;
#endif
}
#ifdef SIMD_COEF_32
for (j = 0; j < BLK_CNT; ++j) {
unsigned char *cpo = &((unsigned char*)input_buf[group_idx])[PARAGETPOS(0, j)];
int k;
for (k = 0; k < roundasciilen; ++k) {
cpo[GETPOS0(k+16)] = roundascii[k];
}
cpo[GETPOS0(k+16)] = 0x80;
((uint32_t*)cpo)[14 * SIMD_COEF_32]=((16+roundasciilen)<<3);
}
zs = zs0 = zb = zb0 = 0;
for (i = 0; i < nsmall-MIN_DROP_BACK; i += BLK_CNT) {
for (j = 0; j < BLK_CNT && zs < nsmall; ++j) {
pConx px = &data[smalls[zs++]];
uint32_t *pi = px->digest;
uint32_t *po = (uint32_t*)&((unsigned char*)input_buf[group_idx])[PARAGETPOS(0, j)];
po[0] = pi[0];
po[COEF] = pi[1];
po[COEF+COEF] = pi[2];
po[COEF+COEF+COEF] = pi[3];
}
SIMDmd5body(input_buf[group_idx], out_buf[group_idx], NULL, SSEi_MIXED_IN);
for (j = 0; j < BLK_CNT && zs0 < nsmall; ++j) {
uint32_t *pi, *po;
pConx px = &data[smalls[zs0++]];
pi = (uint32_t*)&((unsigned char*)out_buf[group_idx])[PARAGETOUTPOS(0, j)];
po = px->digest;
po[0] = pi[0];
po[1] = pi[COEF];
po[2] = pi[COEF+COEF];
po[3] = pi[COEF+COEF+COEF];
}
}
while (zs < nsmall) {
pConx px = &data[smalls[zs++]];
MD5_Init(&px->context);
MD5_Update(&px->context, (unsigned char*)px->digest, sizeof(px->digest));
MD5_Update(&px->context, (unsigned char*)roundascii, roundasciilen);
MD5_Final((unsigned char*)px->digest, &px->context);
}
for (j = 0; j < BLK_CNT; ++j) {
unsigned char *cpo23 = &((unsigned char*)input_buf_big[group_idx][23])[PARAGETPOS(0, j)];
unsigned char *cpo24 = &((unsigned char*)input_buf_big[group_idx][24])[PARAGETPOS(0, j)];
uint32_t *po24 = (uint32_t*)cpo24;
*po24 = 0; 
cpo23[GETPOS0(61)] = roundascii[0];
switch(roundasciilen) {
case 1:
cpo23[GETPOS0(62)] = 0x80;
cpo23[GETPOS0(63)] = 0; 
break;
case 2:
cpo23[GETPOS0(62)] = roundascii[1];
cpo23[GETPOS0(63)] = 0x80;
break;
case 3:
cpo23[GETPOS0(62)] = roundascii[1];
cpo23[GETPOS0(63)] = roundascii[2];
cpo24[0] = 0x80;
break;
case 4:
cpo23[GETPOS0(62)] = roundascii[1];
cpo23[GETPOS0(63)] = roundascii[2];
cpo24[0] = roundascii[3];
cpo24[1] = 0x80;
break;
case 5:
cpo23[GETPOS0(62)] = roundascii[1];
cpo23[GETPOS0(63)] = roundascii[2];
cpo24[0] = roundascii[3];
cpo24[1] = roundascii[4];
cpo24[2] = 0x80;
break;
case 6:
cpo23[GETPOS0(62)] = roundascii[1];
cpo23[GETPOS0(63)] = roundascii[2];
cpo24[0] = roundascii[3];
cpo24[1] = roundascii[4];
cpo24[2] = roundascii[5];
cpo24[3] = 0x80;
break;
}
po24[14*SIMD_COEF_32]=((16+constant_phrase_size+roundasciilen)<<3);
}
for (i = 0; i < nbig-MIN_DROP_BACK; i += BLK_CNT) {
for (j = 0; j < BLK_CNT && zb < nbig; ++j) {
pConx px = &data[bigs[zb++]];
uint32_t *pi = px->digest;
uint32_t *po = (uint32_t*)&((unsigned char*)input_buf_big[group_idx][0])[PARAGETPOS(0, j)];
po[0] = pi[0];
po[COEF] = pi[1];
po[COEF+COEF] = pi[2];
po[COEF+COEF+COEF] = pi[3];
}
SIMDmd5body(input_buf_big[group_idx][0], out_buf[group_idx], NULL, SSEi_MIXED_IN);
for (j = 1; j < 25; ++j)
SIMDmd5body(input_buf_big[group_idx][j], out_buf[group_idx], out_buf[group_idx], SSEi_RELOAD|SSEi_MIXED_IN);
for (j = 0; j < BLK_CNT && zb0 < nbig; ++j) {
uint32_t *pi, *po;
pConx px = &data[bigs[zb0++]];
pi = (uint32_t*)&((unsigned char*)out_buf[group_idx])[PARAGETOUTPOS(0, j)];
po = px->digest;
po[0] = pi[0];
po[1] = pi[COEF];
po[2] = pi[COEF+COEF];
po[3] = pi[COEF+COEF+COEF];
}
}
while (zb < nbig) {
pConx px = &data[bigs[zb++]];
MD5_Init(&px->context);
MD5_Update(&px->context, (unsigned char*)px->digest, sizeof(px->digest));
MD5_Update(&px->context, (unsigned char*)constant_phrase, constant_phrase_size);
MD5_Update(&px->context, (unsigned char*)roundascii, roundasciilen);
MD5_Final((unsigned char*)px->digest, &px->context);
}
#endif
if (++roundascii[roundasciilen-1] == '9'+1) {
int j = roundasciilen-1;
if (j > 0) {
do {
roundascii[j] = '0';
++roundascii[--j];
} while (j > 0 && roundascii[j] == '9'+1);
}
if (!j && roundascii[0] == '9'+1) {
roundascii[0] = '1';
roundascii[roundasciilen++] = '0';
roundascii[roundasciilen] = 0;
}
}
}
}
for (k = 0; k < count; ++k) {
pConx px = &data[k];
memcpy(crypt_out[k], px->digest, FULL_BINARY_SIZE);
}
return count;
}
unsigned int sunmd5_cost(void *salt)
{
return (unsigned int) (BASIC_ROUND_COUNT + getrounds(salt));
}
struct fmt_main fmt_sunmd5 = {
{
FORMAT_LABEL,
FORMAT_NAME,
ALGORITHM_NAME,
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
{ FORMAT_TAG, FORMAT_TAG2 },
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
sunmd5_cost,
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
