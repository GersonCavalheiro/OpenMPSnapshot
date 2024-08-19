#if FMT_EXTERNS_H
extern struct fmt_main fmt_xmpp_scram;
#elif FMT_REGISTERS_H
john_register_one(&fmt_xmpp_scram);
#else
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "arch.h"
#include "misc.h"
#include "memory.h"
#include "common.h"
#include "formats.h"
#include "johnswap.h"
#include "sha.h"
#include "hmac_sha.h"
#include "simd-intrinsics.h"
#include "pbkdf2_hmac_sha1.h"
#if defined SIMD_COEF_32
#define SIMD_KEYS               (SIMD_COEF_32 * SIMD_PARA_SHA1)
#endif
#define FORMAT_LABEL            "xmpp-scram"
#define FORMAT_NAME             ""
#define ALGORITHM_NAME          "XMPP SCRAM PBKDF2-SHA1 " SHA1_ALGORITHM_NAME
#define PLAINTEXT_LENGTH        125
#define HASH_LENGTH             28
#define SALT_SIZE               sizeof(struct custom_salt)
#define SALT_ALIGN              sizeof(uint32_t)
#define BINARY_SIZE             20
#define BINARY_ALIGN            sizeof(uint32_t)
#define BENCHMARK_COMMENT       ""
#define BENCHMARK_LENGTH        0x107
#define FORMAT_TAG              "$xmpp-scram$"
#define FORMAT_TAG_LENGTH       (sizeof(FORMAT_TAG) - 1)
#if !defined(SIMD_COEF_32)
#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      4
#else
#define MIN_KEYS_PER_CRYPT      SIMD_KEYS
#define MAX_KEYS_PER_CRYPT      (SIMD_KEYS * 2)
#endif
#ifndef OMP_SCALE
#define OMP_SCALE               16 
#endif
static struct fmt_tests tests[] = {
{"$xmpp-scram$0$4096$36$37333536663261622d613666622d346333642d396232622d626432646237633338343064$38f79a6e3e64c07f731570d531ec05365aa05306", "openwall123"},
{"$xmpp-scram$0$4096$16$4f67aec1bd53f5f2f74652e69a3b8f32$4aec3caa8ace5180efa7a671092646c041ab1496", "qwerty"},
{"$xmpp-scram$0$4096$16$1f7fcb384d5bcc61dfb1231ae1b32a2f$a2d076d56b0152ed557ad7d38fce93159bc63c9b", "password 123"},
{"$xmpp-scram$0$4096$24$bc1bd6638a1231ffd54f608983425eacf729d8455a469197$aee9254762b23a3950fd7c803caab5f6654587c8", "openwall123"},
{NULL}
};
static struct custom_salt {
uint32_t saltlen;
uint32_t iterations;
uint32_t type;
unsigned char salt[64+1];
} *cur_salt;
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static uint32_t (*crypt_out)[BINARY_SIZE / sizeof(uint32_t)];
static void init(struct fmt_main *self)
{
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*saved_key));
crypt_out = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*crypt_out));
}
static void done(void)
{
MEM_FREE(crypt_out);
MEM_FREE(saved_key);
}
static int valid(char *ciphertext, struct fmt_main *self)
{
char *ctcopy, *keeptr, *p;
int res, extra;
if (strncmp(ciphertext, FORMAT_TAG, FORMAT_TAG_LENGTH) != 0)
return 0;
ctcopy = xstrdup(ciphertext);
keeptr = ctcopy;
ctcopy += FORMAT_TAG_LENGTH;
if ((p = strtokm(ctcopy, "$")) == NULL)	
goto err;
if (!isdec(p))
goto err;
if (atoi(p) != 0)
goto err;
if ((p = strtokm(NULL, "$")) == NULL)	
goto err;
if (!isdec(p))
goto err;
if ((p = strtokm(NULL, "$")) == NULL)	
goto err;
if (!isdec(p))
goto err;
res = atoi(p);
if (res > 64)
goto err;
if ((p = strtokm(NULL, "$")) == NULL)	
goto err;
if (hexlenl(p, &extra) != res * 2 || extra)
goto err;
if ((p = strtokm(NULL, "$")) == NULL)	
goto err;
if (hexlenl(p, &extra) != BINARY_SIZE * 2 || extra)
goto err;
MEM_FREE(keeptr);
return 1;
err:
MEM_FREE(keeptr);
return 0;
}
static void *get_salt(char *ciphertext)
{
static struct custom_salt cs;
char *ctcopy, *keeptr, *p;
int i;
memset(&cs, 0, sizeof(cs));
ctcopy = xstrdup(ciphertext);
keeptr = ctcopy;;
ctcopy += FORMAT_TAG_LENGTH;
p = strtokm(ctcopy, "$");
cs.type = atoi(p);
p = strtokm(NULL, "$");
cs.iterations = atoi(p);
p = strtokm(NULL, "$");
cs.saltlen = atoi(p);
p = strtokm(NULL, "$");
for (i = 0; i < cs.saltlen; i++) {
cs.salt[i] = (atoi16[ARCH_INDEX(*p)] << 4) | atoi16[ARCH_INDEX(p[1])];
p += 2;
}
MEM_FREE(keeptr);
return (void *)&cs;
}
static void *get_binary(char *ciphertext)
{
static union {
unsigned char c[BINARY_SIZE + 1];
ARCH_WORD dummy;
} buf;
unsigned char *out = buf.c;
char *p;
int i;
p = strrchr(ciphertext, '$') + 1;
for (i = 0; i < BINARY_SIZE; i++) {
out[i] = (atoi16[ARCH_INDEX(*p)] << 4) | atoi16[ARCH_INDEX(p[1])];
p += 2;
}
return out;
}
static void set_salt(void *salt)
{
cur_salt = (struct custom_salt *)salt;
}
#define COMMON_GET_HASH_VAR crypt_out
#include "common-get-hash.h"
static int crypt_all(int *pcount, struct db_salt *salt)
{
int index;
const int count = *pcount;
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (index = 0; index < count; index += MIN_KEYS_PER_CRYPT) {
#if !defined (SIMD_COEF_32)
unsigned char out[BINARY_SIZE];
SHA_CTX ctx;
pbkdf2_sha1((unsigned char*)saved_key[index],
strlen(saved_key[index]), cur_salt->salt,
cur_salt->saltlen, cur_salt->iterations, out,
BINARY_SIZE, 0);
hmac_sha1(out, BINARY_SIZE, (unsigned char*)"Client Key", 10, out, BINARY_SIZE);
SHA1_Init(&ctx);
SHA1_Update(&ctx, out, BINARY_SIZE);
SHA1_Final((unsigned char*)crypt_out[index], &ctx);
#else
SHA_CTX ctx;
int i;
unsigned char *pin[SIMD_KEYS];
int lens[SIMD_KEYS];
unsigned char out_[SIMD_KEYS][BINARY_SIZE], *out[SIMD_KEYS];
for (i = 0; i < SIMD_KEYS; ++i) {
pin[i] = (unsigned char*)saved_key[index+i];
lens[i] = strlen(saved_key[index+i]);
out[i] = out_[i];
}
pbkdf2_sha1_sse((const unsigned char **)pin, lens, cur_salt->salt,
cur_salt->saltlen, cur_salt->iterations, out,
BINARY_SIZE, 0);
for (i = 0; i < SIMD_KEYS; ++i) {
hmac_sha1(out[i], BINARY_SIZE, (unsigned char*)"Client Key", 10, out[i], BINARY_SIZE);
SHA1_Init(&ctx);
SHA1_Update(&ctx, out[i], BINARY_SIZE);
SHA1_Final((unsigned char*)crypt_out[index+i], &ctx);
}
#endif
}
return count;
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
static void set_key(char *key, int index)
{
strnzcpy(saved_key[index], key, sizeof(*saved_key));
}
static char *get_key(int index)
{
return saved_key[index];
}
struct fmt_main fmt_xmpp_scram = {
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
{ NULL },
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
{ NULL },
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
fmt_default_salt_hash,
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
