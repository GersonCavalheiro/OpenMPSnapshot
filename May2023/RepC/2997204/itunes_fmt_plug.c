#if FMT_EXTERNS_H
extern struct fmt_main fmt_itunes;
#elif FMT_REGISTERS_H
john_register_one(&fmt_itunes);
#else
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "arch.h"
#include "misc.h"
#include "common.h"
#include "formats.h"
#include "params.h"
#include "options.h"
#include "johnswap.h"
#include "pbkdf2_hmac_sha1.h"
#include "pbkdf2_hmac_sha256.h"
#include "jumbo.h"
#include "itunes_common.h"
#define FORMAT_LABEL            "itunes-backup"
#ifdef SIMD_COEF_32
#define ALGORITHM_NAME          "PBKDF2-SHA1 AES " SHA1_ALGORITHM_NAME
#else
#define ALGORITHM_NAME          "PBKDF2-SHA1 AES 32/" ARCH_BITS_STR
#endif
#define BENCHMARK_COMMENT       ""
#define BENCHMARK_LENGTH        0x107
#define BINARY_SIZE             0
#define PLAINTEXT_LENGTH        125
#define SALT_SIZE               sizeof(struct custom_salt)
#define BINARY_ALIGN            1
#define SALT_ALIGN              sizeof(uint64_t)
#ifdef SIMD_COEF_32
#define MIN_KEYS_PER_CRYPT      (SSE_GROUP_SZ_SHA1 * SSE_GROUP_SZ_SHA256)
#define MAX_KEYS_PER_CRYPT      (SSE_GROUP_SZ_SHA1 * SSE_GROUP_SZ_SHA256)
#else
#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      1
#endif
#ifndef OMP_SCALE
#define OMP_SCALE               1 
#endif
static struct fmt_tests itunes_tests[] = {
{"$itunes_backup$*9*bc707ac0151660426c8114d04caad9d9ee2678a7b7ab05c18ee50cafb2613c31c8978e8b1e9cad2a*10000*266343aaf99102ba7f6af64a3a2d62637793f753**", "123456"},
{"$itunes_backup$*10*31021f9c5a705c3625af21739d397082d90f7a00718a9307687625abc35fc3e4d78371e95cc708b6*10000*8840233131165307147445064802216857558435*1000*c77a159b325d10efee51a1c05701ef63fb85b599", "855632538858211"},
{"$itunes_backup$*10*b3d3f05b5367345fcb654b9b628e2ed24d8b8726f1f74707a956c776475d6ebfffc962340d9cbbca*10000*6832814730342072666684158073107301064276*1000*46de5e844e0ee1c81d2cca6acefb77789c1a7cd0", "1"},
{"$itunes_backup$*9*06dc04bca4eeea2fbc1bc7356fa758243bead479673640a668db285c8f48c402cc435539d935509e*10000*37d2bd7caefbb24a9729e41a3257ef06188dc01e**", "test123"},
{NULL}
};
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static int *cracked, cracked_count;
static struct custom_salt *cur_salt;
static void init(struct fmt_main *self)
{
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(sizeof(*saved_key),  self->params.max_keys_per_crypt);
cracked = mem_calloc(sizeof(*cracked), self->params.max_keys_per_crypt);
cracked_count = self->params.max_keys_per_crypt;
}
static void done(void)
{
MEM_FREE(cracked);
MEM_FREE(saved_key);
}
static void set_salt(void *salt)
{
cur_salt = (struct custom_salt *)salt;
}
static void itunes_set_key(char *key, int index)
{
strnzcpy(saved_key[index], key, sizeof(*saved_key));
}
static char *get_key(int index)
{
return saved_key[index];
}
static int crypt_all(int *pcount, struct db_salt *salt)
{
const int count = *pcount;
int index = 0;
memset(cracked, 0, sizeof(cracked[0])*cracked_count);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (index = 0; index < count; index += MIN_KEYS_PER_CRYPT) {
unsigned char master[MIN_KEYS_PER_CRYPT][32];
int i;
if (cur_salt->version == 9) { 
#ifdef SIMD_COEF_32
int lens[MIN_KEYS_PER_CRYPT];
unsigned char *pin[MIN_KEYS_PER_CRYPT], *pout[MIN_KEYS_PER_CRYPT];
int loops = MIN_KEYS_PER_CRYPT / SSE_GROUP_SZ_SHA1;
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i) {
lens[i] = strlen(saved_key[index+i]);
pin[i] = (unsigned char*)saved_key[index+i];
pout[i] = master[i];
}
for (i = 0; i < loops; i++)
pbkdf2_sha1_sse((const unsigned char**)(pin + i * SSE_GROUP_SZ_SHA1), &lens[i * SSE_GROUP_SZ_SHA1], cur_salt->salt, SALTLEN, cur_salt->iterations, pout + (i * SSE_GROUP_SZ_SHA1), 32, 0);
#else
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i)
pbkdf2_sha1((unsigned char *)saved_key[index+i], strlen(saved_key[index+i]), cur_salt->salt, SALTLEN, cur_salt->iterations, master[i], 32, 0);
#endif
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i) {
cracked[index+i] = itunes_common_decrypt(cur_salt, master[i]);
}
} else { 
#if defined(SIMD_COEF_64) && defined(SIMD_COEF_32)
int lens[MIN_KEYS_PER_CRYPT];
unsigned char *pin[MIN_KEYS_PER_CRYPT], *pout[MIN_KEYS_PER_CRYPT];
int loops = MIN_KEYS_PER_CRYPT / SSE_GROUP_SZ_SHA256;
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i) {
lens[i] = strlen(saved_key[index+i]);
pin[i] = (unsigned char*)saved_key[index+i];
pout[i] = master[i];
}
for (i = 0; i < loops; i++)
pbkdf2_sha256_sse((const unsigned char**)(pin + i * SSE_GROUP_SZ_SHA256), &lens[i * SSE_GROUP_SZ_SHA256], cur_salt->dpsl, SALTLEN, cur_salt->dpic, pout + (i * SSE_GROUP_SZ_SHA256), 32, 0);
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i) {
lens[i] = 32;
pin[i] = (unsigned char*)master[i];
pout[i] = master[i];
}
loops = MIN_KEYS_PER_CRYPT / SSE_GROUP_SZ_SHA1;
for (i = 0; i < loops; i++)
pbkdf2_sha1_sse((const unsigned char**)(pin + i * SSE_GROUP_SZ_SHA1), &lens[i * SSE_GROUP_SZ_SHA1], cur_salt->salt, SALTLEN, cur_salt->iterations, pout + (i * SSE_GROUP_SZ_SHA1), 32, 0);
#else
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i) {
pbkdf2_sha256((unsigned char *)saved_key[index+i], strlen(saved_key[index+i]), cur_salt->dpsl, SALTLEN, cur_salt->dpic, master[i], 32, 0);
pbkdf2_sha1(master[i], 32, cur_salt->salt, SALTLEN, cur_salt->iterations, master[i], 32, 0);
}
#endif
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i) {
cracked[index+i] = itunes_common_decrypt(cur_salt, master[i]);
}
}
}
return count;
}
static int cmp_all(void *binary, int count)
{
int index;
for (index = 0; index < count; index++)
if (cracked[index])
return 1;
return 0;
}
static int cmp_one(void *binary, int index)
{
return cracked[index];
}
static int cmp_exact(char *source, int index)
{
return 1;
}
struct fmt_main fmt_itunes = {
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
"version",
"iteration count",
},
{ FORMAT_TAG },
itunes_tests
}, {
init,
done,
fmt_default_reset,
fmt_default_prepare,
itunes_common_valid,
fmt_default_split,
fmt_default_binary,
itunes_common_get_salt,
{
itunes_common_tunable_version,
itunes_common_tunable_iterations,
},
fmt_default_source,
{
fmt_default_binary_hash
},
fmt_default_salt_hash,
NULL,
set_salt,
itunes_set_key,
get_key,
fmt_default_clear_keys,
crypt_all,
{
fmt_default_get_hash
},
cmp_all,
cmp_one,
cmp_exact
}
};
#endif 
