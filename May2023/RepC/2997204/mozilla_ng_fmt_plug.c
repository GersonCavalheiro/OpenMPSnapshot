#if AC_BUILT
#include "autoconfig.h"
#endif
#if HAVE_LIBCRYPTO
#if FMT_EXTERNS_H
extern struct fmt_main fmt_mozilla;
#elif FMT_REGISTERS_H
john_register_one(&fmt_mozilla);
#else
#include <string.h>
#include <stdint.h>
#include <openssl/des.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "arch.h"
#include "md5.h"
#include "misc.h"
#include "common.h"
#include "formats.h"
#include "johnswap.h"
#include "params.h"
#include "options.h"
#include "sha.h"
#define FORMAT_LABEL            "Mozilla"
#define FORMAT_NAME             "Mozilla key3.db"
#define FORMAT_TAG              "$mozilla$"
#define TAG_LENGTH              (sizeof(FORMAT_TAG) - 1)
#define ALGORITHM_NAME          "SHA1 3DES 32/" ARCH_BITS_STR
#define BENCHMARK_COMMENT       ""
#define BENCHMARK_LENGTH        7
#define PLAINTEXT_LENGTH        125
#define BINARY_SIZE             16
#define BINARY_ALIGN            sizeof(uint32_t)
#define SALT_SIZE               sizeof(struct custom_salt)
#define SALT_ALIGN              sizeof(int)
#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      128
#ifndef OMP_SCALE
#define OMP_SCALE 2 
#endif
static struct fmt_tests tests[] = {
{"$mozilla$*3*20*1*5199adfab24e85e3f308bacf692115f23dcd4f8f*11*2a864886f70d010c050103*16*9debdebd4596b278de029b2b2285ce2e*20*2c4d938ccb3f7f1551262185ccee947deae3b8ae", "12345678"},
{"$mozilla$*3*20*1*4f184f0d3c91cf52ee9190e65389b4d4c8fc66f2*11*2a864886f70d010c050103*16*590d1771368107d6be64844780707787*20*b8458c712ffcc2ff938409804cf3805e4bb7d722", "openwall"},
{"$mozilla$*3*20*1*897f35ff10348f0d3a7739dbf0abddc62e2e64c3*11*2a864886f70d010c050103*16*1851b917997b3119f82b8841a764db62*20*197958dd5e114281f59f9026ad8b7cfe3de7196a", "password"},
{NULL}
};
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static int *saved_len;
static uint32_t (*crypt_out)[BINARY_SIZE / sizeof(uint32_t)];
static  struct custom_salt {
SHA_CTX pctx;
int global_salt_length;
unsigned char global_salt[20];
int local_salt_length;  
unsigned char local_salt[20];
} *cur_salt;
static void init(struct fmt_main *self)
{
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*saved_key));
saved_len = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*saved_len));
crypt_out = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*crypt_out));
}
static void done(void)
{
MEM_FREE(crypt_out);
MEM_FREE(saved_len);
MEM_FREE(saved_key);
}
static int valid(char *ciphertext, struct fmt_main *self)
{
char *p, *keepptr;
int res;
if (strncmp(ciphertext, FORMAT_TAG, TAG_LENGTH))
return 0;
keepptr=xstrdup(ciphertext);
p = &keepptr[TAG_LENGTH];
if (*p != '*')
goto err;
++p;
if ((p = strtokm(p, "*")) == NULL) 
goto err;
if (!isdec(p))
goto err;
res = atoi(p);
if (res != 3)  
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (!isdec(p))
goto err;
res = atoi(p);
if (res > 20)
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (strlen(p) /2 != res)
goto err;
if (!ishexlc(p))
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (!isdec(p))
goto err;
res = atoi(p);
if (res > 20)
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (strlen(p) / 2 != res)
goto err;
if (!ishexlc(p))
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (!isdec(p))
goto err;
res = atoi(p);
if (res > 20)
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (strlen(p) / 2 != res)
goto err;
if (!ishexlc(p))
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (!isdec(p))
goto err;
res = atoi(p);
if (res > 20)
goto err;
if ((p = strtokm(NULL, "*")) == NULL) 
goto err;
if (strlen(p) / 2 != res)
goto err;
if (!ishexlc(p))
goto err;
MEM_FREE(keepptr);
return 1;
err:
MEM_FREE(keepptr);
return 0;
}
static void *get_salt(char *ciphertext)
{
int i;
static struct custom_salt cs;
char *p, *q;
memset(&cs, 0, SALT_SIZE);  
p = ciphertext + TAG_LENGTH;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
cs.local_salt_length = atoi(p);
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
for (i = 0; i < cs.local_salt_length; i++)
cs.local_salt[i] = (atoi16[ARCH_INDEX(p[2 * i])] << 4) |
atoi16[ARCH_INDEX(p[2 * i + 1])];
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
cs.global_salt_length = atoi(p);
q = strchr(p, '*'); 
p = q + 1;
for (i = 0; i < cs.global_salt_length; i++)
cs.global_salt[i] = atoi16[ARCH_INDEX(p[i * 2])]
* 16 + atoi16[ARCH_INDEX(p[i * 2 + 1])];
SHA1_Init(&cs.pctx);
SHA1_Update(&cs.pctx, cs.global_salt, cs.global_salt_length);
return (void *)&cs;
}
static void *get_binary(char *ciphertext)
{
static union {
unsigned char c[BINARY_SIZE];
ARCH_WORD dummy;
} buf;
unsigned char *out = buf.c;
char *p, *q;
int i;
p = ciphertext + TAG_LENGTH;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
q = strchr(p, '*'); 
p = q + 1;
for (i = 0; i < BINARY_SIZE; i++) {
out[i] =
(atoi16[ARCH_INDEX(*p)] << 4) |
atoi16[ARCH_INDEX(p[1])];
p += 2;
}
return out;
}
#define COMMON_GET_HASH_VAR crypt_out
#include "common-get-hash.h"
static void set_salt(void *salt)
{
cur_salt = (struct custom_salt *)salt;
}
static int crypt_all(int *pcount, struct db_salt *salt)
{
const int count = *pcount;
int index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (index = 0; index < count; index++) {
SHA_CTX ctx, ctxi, ctxo;
int i;
union {
unsigned char uc[64];
uint32_t ui[64/4];
} pad;
unsigned char buffer[20];
unsigned char tk[20];
unsigned char key[40];
DES_cblock ivec;
DES_key_schedule ks1, ks2, ks3;
memcpy(&ctx, &cur_salt->pctx, sizeof(SHA_CTX));
SHA1_Update(&ctx, saved_key[index], saved_len[index]);
SHA1_Final(buffer, &ctx);
SHA1_Init(&ctx);
SHA1_Update(&ctx, buffer, 20);
SHA1_Update(&ctx, cur_salt->local_salt, cur_salt->local_salt_length);
SHA1_Final(buffer, &ctx);
SHA1_Init(&ctxi);
SHA1_Init(&ctxo);
memset(pad.uc, 0x36, 64);
for (i = 0; i < 20; ++i)
pad.uc[i] ^= buffer[i];
SHA1_Update(&ctxi, pad.uc, 64);
for (i = 0; i < 64/4; ++i)
pad.ui[i] ^= 0x36363636^0x5c5c5c5c;
SHA1_Update(&ctxo, pad.uc, 64);
memcpy(&ctx, &ctxi, sizeof(ctx));
SHA1_Update(&ctx, cur_salt->local_salt, 20);
SHA1_Update(&ctx, cur_salt->local_salt, cur_salt->local_salt_length);
SHA1_Final(buffer, &ctx);
memcpy(&ctx, &ctxo, sizeof(ctx));
SHA1_Update(&ctx, buffer, 20);
SHA1_Final(key, &ctx);
memcpy(&ctx, &ctxi, sizeof(ctx));
SHA1_Update(&ctx, cur_salt->local_salt, 20);
SHA1_Final(buffer, &ctx);
memcpy(&ctx, &ctxo, sizeof(ctx));
SHA1_Update(&ctx, buffer, 20);
SHA1_Final(tk, &ctx);
SHA1_Update(&ctxi, tk, 20);
SHA1_Update(&ctxi, cur_salt->local_salt, cur_salt->local_salt_length);
SHA1_Final(buffer, &ctxi);
SHA1_Update(&ctxo, buffer, 20);
SHA1_Final(key+20, &ctxo);
DES_set_key_unchecked((DES_cblock *) key, &ks1);
DES_set_key_unchecked((DES_cblock *) (key+8), &ks2);
DES_set_key_unchecked((DES_cblock *) (key+16), &ks3);
memcpy(ivec, key + 32, 8);  
DES_ede3_cbc_encrypt((unsigned char*)"password-check\x02\x02", (unsigned char*)crypt_out[index], 16, &ks1, &ks2, &ks3, &ivec, DES_ENCRYPT);
}
return count;
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
return !memcmp(binary, crypt_out[index], BINARY_SIZE);
}
static int cmp_exact(char *source, int index)
{
return 1;
}
static void mozilla_set_key(char *key, int index)
{
saved_len[index] = strnzcpyn(saved_key[index], key, sizeof(*saved_key));
}
static char *get_key(int index)
{
return saved_key[index];
}
struct fmt_main fmt_mozilla = {
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
BINARY_ALIGN,
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
mozilla_set_key,
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
#endif 
